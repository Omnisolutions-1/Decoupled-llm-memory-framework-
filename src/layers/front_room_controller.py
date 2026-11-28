import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# Ensure repo root is importable when running src/layers/front_room_controller.py directly
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from semantic_ingestion_gateway import (
    AdaptiveHybridRetriever,
    CompactionReport,
    HierarchicalCompactor,
    LatencyTracker,
    SearchHit,
    SemanticIngestionGateway,
    TokenBudgetedFrontRoom,
    cosine_similarity as _gateway_cosine_similarity,
    count_tokens,
    default_embed_text,
)

# Optional heavy ML dependencies; degrade gracefully to zero-dependency v2.1 utilities
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None  # type: ignore
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity as _sklearn_cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    _sklearn_cosine_similarity = None  # type: ignore
    SKLEARN_AVAILABLE = False

# Try importing Qdrant for production environments; degrade gracefully if missing
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


@dataclass
class SemanticPointer:
    """Lightweight metadata fingerprint for a conversation turn."""
    turn_id: str
    timestamp: float
    topic_label: str       # e.g., "technical_issue", "billing"
    entities: List[str]    # Extracted named entities
    embedding: np.ndarray  # Dense vector representation (384-dim for MiniLM, 64-dim fallback)
    token_count: int


@dataclass
class ConversationTurn:
    """A single turn in the active window."""
    turn_id: str
    user_input: str
    assistant_response: str
    pointer: SemanticPointer


@dataclass
class ArchivedBundle:
    """A conversation turn archived in the warehouse."""
    turn_id: str
    user_input: str
    assistant_response: str
    pointer: SemanticPointer


class FrontRoomController:
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        active_window_size: int = 2,
        drift_threshold: float = 0.65,
        max_retrieval_bundles: int = 3,
        max_context_tokens: int = 1764,
        warehouse_query_fn: Optional[Callable] = None,
        warehouse_archive_fn: Optional[Callable] = None,
        gateway: Optional[SemanticIngestionGateway] = None,
        retriever: Optional[AdaptiveHybridRetriever] = None,
        compactor: Optional[HierarchicalCompactor] = None,
        use_local_embeddings: bool = False,
    ):
        self.embedding_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE and not use_local_embeddings:
            try:
                self.embedding_model = SentenceTransformer(embedding_model_name)
            except Exception:
                self.embedding_model = None

        self.active_window_size = active_window_size
        self.drift_threshold = drift_threshold
        self.max_retrieval_bundles = max_retrieval_bundles
        self.max_context_tokens = max_context_tokens

        self.warehouse_query = warehouse_query_fn or self._extract_placeholder_query()
        self.warehouse_archive = warehouse_archive_fn or self._mock_warehouse_archive

        self.gateway = gateway or SemanticIngestionGateway(embed_fn=self._embed_list)
        self.retriever = retriever or AdaptiveHybridRetriever(
            embed_fn=self._embed_list,
            drift_threshold=drift_threshold,
        )
        self.compactor = compactor or HierarchicalCompactor(embed_fn=self._embed_list)
        self.token_budgeted_front_room = TokenBudgetedFrontRoom(
            gateway=self.gateway,
            retriever=self.retriever,
            active_window_size=active_window_size,
            max_context_tokens=max_context_tokens,
            max_retrieval_bundles=max_retrieval_bundles,
        )
        self.latency_tracker: LatencyTracker = self.token_budgeted_front_room.tracker

        self.active_window: List[ConversationTurn] = []
        self.turn_counter = 0

    def _embed_list(self, text: str) -> List[float]:
        if self.embedding_model is not None:
            vec = self.embedding_model.encode(text, convert_to_numpy=True)
            return vec.tolist()
        return default_embed_text(text)

    def _embed_numpy(self, text: str) -> np.ndarray:
        if self.embedding_model is not None:
            return self.embedding_model.encode(text, convert_to_numpy=True)
        return np.array(default_embed_text(text), dtype=np.float32)

    def _extract_placeholder_query(self):
        return self._mock_warehouse_query

    def extract_semantic_pointer(self, text: str, turn_id: str) -> SemanticPointer:
        embedding = self._embed_numpy(text)
        entities = self._extract_entities_simple(text)
        topic_label = self._classify_topic_simple(text)
        token_count = count_tokens(text)

        return SemanticPointer(
            turn_id=turn_id,
            timestamp=time.time(),
            topic_label=topic_label,
            entities=entities,
            embedding=embedding,
            token_count=token_count,
        )

    def detect_semantic_drift(self, new_embedding: np.ndarray) -> bool:
        if not self.active_window:
            return False

        if SKLEARN_AVAILABLE and _sklearn_cosine_similarity is not None:
            active_embeddings = np.array([
                turn.pointer.embedding for turn in self.active_window
            ])
            similarities = _sklearn_cosine_similarity(
                new_embedding.reshape(1, -1),
                active_embeddings,
            )[0]
            return float(similarities.max()) < self.drift_threshold

        new_list = new_embedding.tolist()
        max_sim = max(
            _gateway_cosine_similarity(new_list, turn.pointer.embedding.tolist())
            for turn in self.active_window
        )
        return max_sim < self.drift_threshold

    def retrieve_relevant_history(self, current_pointer: SemanticPointer) -> List[ArchivedBundle]:
        return self.warehouse_query(
            turn_id=current_pointer.turn_id,
            embedding=current_pointer.embedding,
            k=self.max_retrieval_bundles,
        )

    def compact_memory(self) -> CompactionReport:
        return self.compactor.compact(self.gateway)

    def get_telemetry(self) -> Dict[str, Any]:
        return self.latency_tracker.summary()

    def process_turn(self, user_input: str, llm_engine: Callable[[str], str]) -> str:
        self.turn_counter += 1
        turn_id = f"turn_{self.turn_counter:05d}"

        current_pointer = self.extract_semantic_pointer(user_input, turn_id)
        assembly = self.token_budgeted_front_room.process_turn(user_input)

        drift_detected = assembly.drifted or self.detect_semantic_drift(current_pointer.embedding)
        injected_history: List[ArchivedBundle] = []
        if drift_detected:
            injected_history = self.retrieve_relevant_history(current_pointer)
            if injected_history or assembly.retrieved_memories:
                print(
                    f"[FrontRoom] Drift detected (sim={assembly.drift_similarity}). "
                    f"Injected {len(injected_history)} warehouse bundles & "
                    f"{len(assembly.retrieved_memories)} hybrid hits."
                )

        context = self._build_llm_context(
            user_input,
            injected_history,
            retrieved_memories=assembly.retrieved_memories,
        )

        assistant_response = llm_engine(context)

        self.gateway.ingest(
            raw_text=user_input,
            turn_id=turn_id,
            topic_label=current_pointer.topic_label,
            entities=current_pointer.entities,
        )

        current_turn = ConversationTurn(
            turn_id=turn_id,
            user_input=user_input,
            assistant_response=assistant_response,
            pointer=current_pointer,
        )
        self.active_window.append(current_turn)

        while len(self.active_window) > self.active_window_size:
            archived_turn = self.active_window.pop(0)
            bundle = ArchivedBundle(
                turn_id=archived_turn.turn_id,
                user_input=archived_turn.user_input,
                assistant_response=archived_turn.assistant_response,
                pointer=archived_turn.pointer,
            )
            self.warehouse_archive(bundle)
            print(f"[FrontRoom] Archived {bundle.turn_id} to warehouse.")

        return assistant_response

    def _build_llm_context(
        self,
        current_input: str,
        injected_history: List[ArchivedBundle],
        retrieved_memories: Optional[List[SearchHit]] = None,
    ) -> str:
        context_parts: List[str] = []
        prompt_tail = f"User: {current_input}\nAssistant:"
        used_tokens = count_tokens(prompt_tail)

        if retrieved_memories:
            mem_lines = ["=== RELEVANT CONSOLIDATED MEMORY ==="]
            for hit in retrieved_memories:
                line = f"[{hit.item_type}] {hit.text}"
                line_tokens = count_tokens(line)
                if used_tokens + line_tokens <= self.max_context_tokens:
                    mem_lines.append(line)
                    used_tokens += line_tokens
            if len(mem_lines) > 1:
                mem_lines.append("=== END CONSOLIDATED MEMORY ===\n")
                context_parts.extend(mem_lines)

        if injected_history:
            hist_lines = ["=== RELEVANT PAST CONTEXT ==="]
            for bundle in injected_history:
                block = f"User: {bundle.user_input}\nAssistant: {bundle.assistant_response}"
                block_tokens = count_tokens(block)
                if used_tokens + block_tokens <= self.max_context_tokens:
                    hist_lines.append(block)
                    used_tokens += block_tokens
            if len(hist_lines) > 1:
                hist_lines.append("=== END PAST CONTEXT ===\n")
                context_parts.extend(hist_lines)

        if self.active_window:
            recent_lines = ["=== RECENT CONVERSATION ==="]
            packed_turns: List[str] = []
            for turn in reversed(self.active_window):
                block = f"User: {turn.user_input}\nAssistant: {turn.assistant_response}"
                block_tokens = count_tokens(block)
                if used_tokens + block_tokens <= self.max_context_tokens:
                    packed_turns.insert(0, block)
                    used_tokens += block_tokens
            if packed_turns:
                recent_lines.extend(packed_turns)
                recent_lines.append("=== END RECENT CONVERSATION ===\n")
                context_parts.extend(recent_lines)

        context_parts.append(prompt_tail)
        return "\n".join(context_parts)

    def _extract_entities_simple(self, text: str) -> List[str]:
        return SemanticIngestionGateway._extract_entities(text)[:5]

    def _classify_topic_simple(self, text: str) -> str:
        text_lower = text.lower()
        if any(kw in text_lower for kw in ["error", "bug", "broken", "fail", "authentication"]):
            return "technical_issue"
        elif any(kw in text_lower for kw in ["price", "cost", "billing", "payment", "pricing"]):
            return "billing"
        elif any(kw in text_lower for kw in ["how", "what", "explain", "tutorial"]):
            return "information_request"
        return "general"

    def _mock_warehouse_query(self, turn_id: str, embedding: np.ndarray, k: int) -> List[ArchivedBundle]:
        print(f"[Warehouse] Mock query for turn {turn_id} (k={k})")
        return []

    def _mock_warehouse_archive(self, bundle: ArchivedBundle) -> None:
        print(f"[Warehouse] Mock archive: {bundle.turn_id}")


if QDRANT_AVAILABLE:
    class QdrantWarehouse:
        def __init__(self, host: str = "localhost", port: int = 6333, vector_size: int = 384):
            self.client = QdrantClient(host=host, port=port)
            self.collection_name = "conversation_history"
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
            except Exception:
                pass

        def archive_bundle(self, bundle: ArchivedBundle) -> None:
            point = PointStruct(
                id=bundle.turn_id,
                vector=bundle.pointer.embedding.tolist(),
                payload={
                    "user_input": bundle.user_input,
                    "assistant_response": bundle.assistant_response,
                    "topic_label": bundle.pointer.topic_label,
                    "entities": bundle.pointer.entities,
                    "timestamp": bundle.pointer.timestamp,
                    "token_count": bundle.pointer.token_count,
                },
            )
            self.client.upsert(collection_name=self.collection_name, points=[point])

        def query_similar_bundles(self, turn_id: str, embedding: np.ndarray, k: int) -> List[ArchivedBundle]:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=embedding.tolist(),
                limit=k,
                score_threshold=0.5,
            )
            bundles = []
            for result in results:
                payload = result.payload
                pointer = SemanticPointer(
                    turn_id=str(result.id),
                    timestamp=payload["timestamp"],
                    topic_label=payload["topic_label"],
                    entities=payload["entities"],
                    embedding=np.array(result.vector if result.vector else []),
                    token_count=payload["token_count"],
                )
                bundles.append(ArchivedBundle(
                    turn_id=str(result.id),
                    user_input=payload["user_input"],
                    assistant_response=payload["assistant_response"],
                    pointer=pointer,
                ))
            return bundles


def mock_llm_engine(context: str) -> str:
    return f"[Mock LLM Response - received {len(context)} chars ({count_tokens(context)} tokens) of context]"


if __name__ == "__main__":
    controller = FrontRoomController(
        active_window_size=2,
        drift_threshold=0.65,
        max_retrieval_bundles=3,
        max_context_tokens=1764,
        use_local_embeddings=True,
    )
    turns = [
        "I'm having trouble logging into the system",
        "It says 'authentication failed' when I enter my password",
        "Okay, I reset it and now it works. Thanks!",
        "By the way, what's the pricing for the enterprise plan?",
        "And does that include priority support?",
    ]
    for user_input in turns:
        print(f"\n{'='*60}")
        print(f"USER: {user_input}")
        response = controller.process_turn(user_input, mock_llm_engine)
        print(f"ASSISTANT: {response}")
        print(f"Active Window Size: {len(controller.active_window)}")

    print(f"\n{'='*60}")
    print("Compaction report:", controller.compact_memory())
    print("Telemetry summary:", controller.get_telemetry())
