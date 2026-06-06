import time
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Try importing Qdrant for production environments; degrade gracefully if missing
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

# ========================================================================
# CORE DATA STORAGE OBJECTS
# ========================================================================

@dataclass
class SemanticPointer:
    """Lightweight metadata fingerprint for a conversation turn."""
    turn_id: str
    timestamp: float
    topic_label: str       # e.g., "technical_issue", "billing"
    entities: List[str]    # Extracted named entities
    embedding: np.ndarray  # Dense vector representation (384-dim for MiniLM)
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

# ========================================================================
# FRONT ROOM CONTROLLER CORE ENGINE
# ========================================================================

class FrontRoomController:
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        active_window_size: int = 2,
        drift_threshold: float = 0.65,  # Cosine similarity threshold
        max_retrieval_bundles: int = 3,
        warehouse_query_fn: Optional[Callable] = None,
        warehouse_archive_fn: Optional[Callable] = None
    ):
        """
        Manages the fast active window context and triggers warehouse telemetry syncs.
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.active_window_size = active_window_size
        self.drift_threshold = drift_threshold
        self.max_retrieval_bundles = max_retrieval_bundles

        # Inject infrastructure functions or fall back to local testing mocks
        self.warehouse_query = warehouse_query_fn or self._extract_placeholder_query()
        self.warehouse_archive = warehouse_archive_fn or self._mock_warehouse_archive

        # Session tracking state
        self.active_window: List[ConversationTurn] = []
        self.turn_counter = 0

    def _extract_placeholder_query(self):
        """Internal selector to bind default fallback mocks safely."""
        return self._mock_warehouse_query

    def extract_semantic_pointer(self, text: str, turn_id: str) -> SemanticPointer:
        """Generates a lightweight semantic fingerprint for a conversation turn."""
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        entities = self._extract_entities_simple(text)
        topic_label = self._classify_topic_simple(text)
        token_count = len(text.split())

        return SemanticPointer(
            turn_id=turn_id,
            timestamp=time.time(),
            topic_label=topic_label,
            entities=entities,
            embedding=embedding,
            token_count=token_count
        )

    def detect_semantic_drift(self, new_embedding: np.ndarray) -> bool:
        """Determines if a new input drifts from active window context."""
        if not self.active_window:
            return False  # Empty state cannot drift

        active_embeddings = np.array([
            turn.pointer.embedding for turn in self.active_window
        ])

        similarities = cosine_similarity(
            new_embedding.reshape(1, -1),
            active_embeddings
        )[0]

        return similarities.max() < self.drift_threshold

    def retrieve_relevant_history(self, current_pointer: SemanticPointer) -> List[ArchivedBundle]:
        """Queries the telemetry warehouse using injected query interfaces."""
        return self.warehouse_query(
            turn_id=current_pointer.turn_id,
            embedding=current_pointer.embedding,
            k=self.max_retrieval_bundles
        )

    def process_turn(self, user_input: str, llm_engine: Callable[[str], str]) -> str:
        """Main execution framework loop."""
        self.turn_counter += 1
        turn_id = f"turn_{self.turn_counter:05d}"

        # 1. Pipeline Fingerprinting
        current_pointer = self.extract_semantic_pointer(user_input, turn_id)

        # 2. Evaluation & Retrieval Gateways
        drift_detected = self.detect_semantic_drift(current_pointer.embedding)
        injected_history = []
        if drift_detected:
            injected_history = self.retrieve_relevant_history(current_pointer)
            print(f"[FrontRoom] Drift detected. Injected {len(injected_history)} historical bundles.")

        # 3. Compile Context String
        context = self._build_llm_context(user_input, injected_history)

        # 4. Process Model Inference Response
        assistant_response = llm_engine(context)

        # 5. Commit to Active Context Window
        current_turn = ConversationTurn(
            turn_id=turn_id,
            user_input=user_input,
            assistant_response=assistant_response,
            pointer=current_pointer
        )
        self.active_window.append(current_turn)

        # 6. Apply Strict O(1) Telemetry Archival
        while len(self.active_window) > self.active_window_size:
            archived_turn = self.active_window.pop(0)
            bundle = ArchivedBundle(
                turn_id=archived_turn.turn_id,
                user_input=archived_turn.user_input,
                assistant_response=archived_turn.assistant_response,
                pointer=archived_turn.pointer
            )
            self.warehouse_archive(bundle)
            print(f"[FrontRoom] Archived {bundle.turn_id} to warehouse.")

        return assistant_response

    def _build_llm_context(self, current_input: str, injected_history: List[ArchivedBundle]) -> str:
        """Constructs structural prompt formatting tags for execution safety."""
        context_parts = []

        if injected_history:
            context_parts.append("=== RELEVANT PAST CONTEXT ===")
            for bundle in injected_history:
                context_parts.append(f"User: {bundle.user_input}")
                context_parts.append(f"Assistant: {bundle.assistant_response}")
            context_parts.append("=== END PAST CONTEXT ===\n")

        if self.active_window:
            context_parts.append("=== RECENT CONVERSATION ===")
            for turn in self.active_window:
                context_parts.append(f"User: {turn.user_input}")
                context_parts.append(f"Assistant: {turn.assistant_response}")
            context_parts.append("=== END RECENT CONVERSATION ===\n")

        context_parts.append(f"User: {current_input}")
        context_parts.append("Assistant:")
        return "\n".join(context_parts)

    # ========================================================================
    # PLACEHOLDER LOCAL TESTING METHODS
    # ========================================================================

    def _extract_entities_simple(self, text: str) -> List[str]:
        words = text.split()
        if not words:
            return []
        entities = [w.strip("?,.:;!") for w in words if w and w[0].isupper() and len(w) > 2]
        return list(set(entities))[:5]

    def _classify_topic_simple(self, text: str) -> str:
        text_lower = text.lower()
        if any(kw in text_lower for kw in ["error", "bug", "broken", "fail"]):
            return "technical_issue"
        elif any(kw in text_lower for kw in ["price", "cost", "billing", "payment"]):
            return "billing"
        elif any(kw in text_lower for kw in ["how", "what", "explain", "tutorial"]):
            return "information_request"
        return "general"

    def _mock_warehouse_query(self, turn_id: str, embedding: np.ndarray, k: int) -> List[ArchivedBundle]:
        print(f"[Warehouse] Mock query for turn {turn_id} (k={k})")
        return []

    def _mock_warehouse_archive(self, bundle: ArchivedBundle) -> None:
        print(f"[Warehouse] Mock archive: {bundle.turn_id}")

# ========================================================================
# QDRANT DISTRIBUTED ECOSYSTEM INTEGRATION
# ========================================================================

if QDRANT_AVAILABLE:
    class QdrantWarehouse:
        def __init__(self, host: str = "localhost", port: int = 6333):
            self.client = QdrantClient(host=host, port=port)
            self.collection_name = "conversation_history"
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
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
                    "token_count": bundle.pointer.token_count
                }
            )
            self.client.upsert(collection_name=self.collection_name, points=[point])

        def query_similar_bundles(self, turn_id: str, embedding: np.ndarray, k: int) -> List[ArchivedBundle]:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=embedding.tolist(),
                limit=k,
                score_threshold=0.5
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
                    token_count=payload["token_count"]
                )
                bundles.append(ArchivedBundle(
                    turn_id=str(result.id),
                    user_input=payload["user_input"],
                    assistant_response=payload["assistant_response"],
                    pointer=pointer
                ))
            return bundles

# ========================================================================
# SIMULATION LOCAL RUNTIME EXECUTION
# ========================================================================

def mock_llm_engine(context: str) -> str:
    return f"[Mock LLM Response - received {len(context)} chars of context]"


if __name__ == "__main__":
    controller = FrontRoomController(
        active_window_size=2,
        drift_threshold=0.65,
        max_retrieval_bundles=3
    )
    turns = [
        "I'm having trouble logging into the system",
        "It says 'authentication failed' when I enter my password",
        "Okay, I reset it and now it works. Thanks!",
        "By the way, what's the pricing for the enterprise plan?",  # Topic drift!
        "And does that include priority support?"
    ]
    for user_input in turns:
        print(f"\n{'='*60}")
        print(f"USER: {user_input}")
        response = controller.process_turn(user_input, mock_llm_engine)
        print(f"ASSISTANT: {response}")
        print(f"Active Window Size: {len(controller.active_window)}")
