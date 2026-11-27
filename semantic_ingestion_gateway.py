import hashlib
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

try:
    import tiktoken
    _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TIKTOKEN_ENC = None

_NEGATION_WORDS = {"not", "never", "no", "dont", "don't", "doesnt", "doesn't", "cannot", "can't", "wont", "won't"}
_LEADING_FLUFF = re.compile(
    r"^(?:hey|hello|hi|um+|uh+|basically|actually|well|so|okay|ok|please|just)[\s,!.:;-]+",
    re.IGNORECASE,
)
_TRAILING_FLUFF = re.compile(
    r"[\s,!.:;-]+(?:thanks|thank you|thx|please|cheers|appreciated)[!.\s]*$",
    re.IGNORECASE,
)
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "and", "or", "in", "on", "at", "for", "with", "by", "from",
    "as", "it", "this", "that", "these", "those", "why", "what", "how", "when",
    "where", "who", "which", "does", "do", "did", "uses", "use", "using", "need",
}


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if _TIKTOKEN_ENC is not None:
        return len(_TIKTOKEN_ENC.encode(text))
    return len(re.findall(r"\S+", text))


def default_embed_text(text: str, dim: int = 64) -> List[float]:
    vec = [0.0] * dim
    tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    if not tokens:
        return vec
    features = list(tokens) + [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
    for feat in features:
        digest = hashlib.md5(feat.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if (digest[4] % 2 == 0) else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0.0:
        vec = [v / norm for v in vec]
    return vec


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass
class RuleNode:
    rule_id: str
    text: str
    normalized_text: str
    topic_label: str
    entities: List[str]
    embedding: List[float]
    timestamp: float
    evidence_count: int = 1
    consistency_weight: float = 1.0
    is_distorted: bool = False
    distortion_confidence: float = 0.0
    distortion_reason: str = ""
    source_turns: List[str] = field(default_factory=list)


@dataclass
class ElephantNode:
    elephant_id: str
    topic_label: str
    summary_text: str
    entities: List[str]
    embedding: List[float]
    member_rule_ids: List[str]
    created_at: float
    consistency_weight: float = 1.0


@dataclass
class PuzzlePiece:
    piece_id: str
    text: str
    normalized_text: str
    topic_label: str
    entities: List[str]
    embedding: List[float]
    anomaly_type: str
    conflicting_rule_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class CondemnedPattern:
    pattern_id: str
    text: str
    normalized_text: str
    embedding: List[float]
    frozen_weight: float
    reason: str
    condemned_at: float = field(default_factory=time.time)


@dataclass
class IngestionResult:
    action: str
    stripped_text: str
    target_id: Optional[str] = None
    conflicting_rule_id: Optional[str] = None
    table_name: str = "default"


@dataclass
class CompactionReport:
    purged_distorted_rules: int
    created_elephants: int
    merged_rule_count: int
    remaining_active_nodes: int


@dataclass
class SearchHit:
    item_id: str
    item_type: str
    text: str
    score: float
    rare_anchors_matched: List[str] = field(default_factory=list)


@dataclass
class TurnAssembly:
    drifted: bool
    drift_similarity: float
    retrieved_memories: List[SearchHit]
    total_tokens: int


class LatencyTracker:
    def __init__(self) -> None:
        self.latencies_ms: List[float] = []

    def record(self, duration_ms: float) -> None:
        self.latencies_ms.append(max(duration_ms, 1e-4))

    def summary(self) -> Dict[str, Any]:
        if not self.latencies_ms:
            return {"total_turns": 0, "p50_ms": 0.0, "p95_ms": 0.0}
        ordered = sorted(self.latencies_ms)
        n = len(ordered)
        p50_idx = int(0.50 * (n - 1))
        p95_idx = int(0.95 * (n - 1))
        return {
            "total_turns": n,
            "p50_ms": ordered[p50_idx],
            "p95_ms": ordered[p95_idx],
        }


class SemanticIngestionGateway:
    def __init__(self, embed_fn: Optional[Callable[[str], List[float]]] = None) -> None:
        self.embed_fn = embed_fn or default_embed_text
        self.rules: Dict[str, RuleNode] = {}
        self.elephants: Dict[str, ElephantNode] = {}
        self.puzzle_tables: Dict[str, Dict[str, PuzzlePiece]] = {}
        self.condemned_patterns: Dict[str, CondemnedPattern] = {}
        self.audit_archive: List[Dict[str, Any]] = []

    @staticmethod
    def _strip_fluff(text: str) -> str:
        cleaned = text.strip()
        while True:
            updated = _LEADING_FLUFF.sub("", cleaned).strip()
            if updated == cleaned:
                break
            cleaned = updated
        while True:
            updated = _TRAILING_FLUFF.sub("", cleaned).strip()
            if updated == cleaned:
                break
            cleaned = updated
        return cleaned or text.strip()

    @staticmethod
    def _normalize(text: str) -> str:
        tokens = re.findall(r"[a-zA-Z0-9_.]+", text.lower())
        return " ".join(tokens)

    @staticmethod
    def _extract_entities(text: str) -> List[str]:
        candidates = re.findall(r"\b[A-Za-z0-9_]*_[A-Za-z0-9_]+\b|\b[A-Z][a-zA-Z0-9_]+\b|\b\d+(?:\.\d+)?\b", text)
        seen: Set[str] = set()
        result: List[str] = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                result.append(c)
        return result

    @staticmethod
    def _has_negation(text: str) -> bool:
        words = set(re.findall(r"[a-zA-Z']+", text.lower()))
        return bool(words & _NEGATION_WORDS)

    @staticmethod
    def _content_tokens(text: str) -> Set[str]:
        words = set(re.findall(r"[a-zA-Z0-9_.]+", text.lower()))
        return words - _STOPWORDS - _NEGATION_WORDS

    def ingest(
        self,
        raw_text: str,
        turn_id: str = "",
        topic_label: str = "general",
        entities: Optional[List[str]] = None,
        now: Optional[float] = None,
    ) -> IngestionResult:
        ts = now if now is not None else time.time()
        stripped = self._strip_fluff(raw_text)
        norm = self._normalize(stripped)
        ent = entities if entities is not None else self._extract_entities(stripped)
        emb = self.embed_fn(stripped)

        # 1. Check against condemned pattern stubs (read-only frozen weights)
        for stub in self.condemned_patterns.values():
            if norm == stub.normalized_text or cosine_similarity(emb, stub.embedding) >= 0.88:
                piece_id = f"piece_{uuid.uuid4().hex[:8]}"
                table_name = "condemned"
                self.puzzle_tables.setdefault(table_name, {})[piece_id] = PuzzlePiece(
                    piece_id=piece_id,
                    text=stripped,
                    normalized_text=norm,
                    topic_label=topic_label,
                    entities=ent,
                    embedding=emb,
                    anomaly_type="condemned_match",
                    timestamp=ts,
                )
                return IngestionResult(
                    action="flagged_anomaly",
                    stripped_text=stripped,
                    target_id=piece_id,
                    table_name=table_name,
                )

        # 2. Check existing rules for contradiction or consistency folding
        neg = self._has_negation(stripped)
        tokens = self._content_tokens(stripped)
        for rule_id, rule in self.rules.items():
            if rule.is_distorted:
                continue
            rule_neg = self._has_negation(rule.text)
            rule_tokens = self._content_tokens(rule.text)
            overlap = len(tokens & rule_tokens) / max(1, len(tokens | rule_tokens))
            sim = cosine_similarity(emb, rule.embedding)

            if neg != rule_neg and (overlap >= 0.6 or sim >= 0.78):
                piece_id = f"piece_{uuid.uuid4().hex[:8]}"
                table_name = "conflicts"
                self.puzzle_tables.setdefault(table_name, {})[piece_id] = PuzzlePiece(
                    piece_id=piece_id,
                    text=stripped,
                    normalized_text=norm,
                    topic_label=topic_label,
                    entities=ent,
                    embedding=emb,
                    anomaly_type="contradiction",
                    conflicting_rule_id=rule_id,
                    timestamp=ts,
                )
                return IngestionResult(
                    action="flagged_anomaly",
                    stripped_text=stripped,
                    target_id=piece_id,
                    conflicting_rule_id=rule_id,
                    table_name=table_name,
                )

            if neg == rule_neg and (norm == rule.normalized_text or sim >= 0.90):
                rule.evidence_count += 1
                rule.consistency_weight += 0.5
                rule.timestamp = max(rule.timestamp, ts)
                if turn_id:
                    rule.source_turns.append(turn_id)
                return IngestionResult(
                    action="folded_consistency",
                    stripped_text=stripped,
                    target_id=rule_id,
                )

        # 3. Create new rule
        rule_id = f"rule_{uuid.uuid4().hex[:8]}"
        self.rules[rule_id] = RuleNode(
            rule_id=rule_id,
            text=stripped,
            normalized_text=norm,
            topic_label=topic_label,
            entities=ent,
            embedding=emb,
            timestamp=ts,
            source_turns=[turn_id] if turn_id else [],
        )
        return IngestionResult(
            action="created_rule",
            stripped_text=stripped,
            target_id=rule_id,
        )

    def copy_piece_to_table(self, piece_id: str, source_table: str, target_table: str) -> bool:
        src = self.puzzle_tables.get(source_table, {})
        if piece_id not in src:
            return False
        self.puzzle_tables.setdefault(target_table, {})[piece_id] = src[piece_id]
        return True

    def resolve_puzzle_piece(
        self,
        piece_id: str,
        table_name: str,
        is_valid_truth: bool = True,
        supersedes_conflicting_rule: bool = True,
    ) -> Optional[str]:
        piece = self.puzzle_tables.get(table_name, {}).get(piece_id)
        if piece is None:
            return None

        # Remove from all puzzle tables
        for tbl in self.puzzle_tables.values():
            tbl.pop(piece_id, None)

        if not is_valid_truth:
            return None

        if supersedes_conflicting_rule and piece.conflicting_rule_id in self.rules:
            old_rule = self.rules[piece.conflicting_rule_id]
            old_rule.is_distorted = True
            old_rule.distortion_confidence = 0.95
            old_rule.distortion_reason = "Superseded by resolved puzzle piece"

        new_rule_id = f"rule_{uuid.uuid4().hex[:8]}"
        self.rules[new_rule_id] = RuleNode(
            rule_id=new_rule_id,
            text=piece.text,
            normalized_text=piece.normalized_text,
            topic_label=piece.topic_label,
            entities=piece.entities,
            embedding=piece.embedding,
            timestamp=piece.timestamp,
        )
        return new_rule_id


class HierarchicalCompactor:
    def __init__(
        self,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        cluster_similarity_threshold: float = 0.72,
    ) -> None:
        self.embed_fn = embed_fn or default_embed_text
        self.cluster_similarity_threshold = cluster_similarity_threshold

    @staticmethod
    def _non_numeric_entities(entities: List[str]) -> Set[str]:
        return {e for e in entities if not re.fullmatch(r"\d+(?:\.\d+)?", e)}

    def _can_merge(self, r1: RuleNode, r2: RuleNode) -> bool:
        if r1.topic_label != r2.topic_label:
            return False
        sim = cosine_similarity(r1.embedding, r2.embedding)
        shared_non_num = self._non_numeric_entities(r1.entities) & self._non_numeric_entities(r2.entities)
        if shared_non_num:
            return True
        return sim >= self.cluster_similarity_threshold

    def compact(self, gateway: SemanticIngestionGateway) -> CompactionReport:
        purged = 0
        to_purge = [
            rid for rid, r in gateway.rules.items()
            if r.is_distorted and r.distortion_confidence >= 0.5
        ]
        for rid in to_purge:
            rule = gateway.rules.pop(rid)
            purged += 1
            gateway.audit_archive.append({
                "node_id": rid,
                "text": rule.text,
                "reason": rule.distortion_reason,
                "archived_at": time.time(),
            })
            pat_id = f"cond_{uuid.uuid4().hex[:8]}"
            gateway.condemned_patterns[pat_id] = CondemnedPattern(
                pattern_id=pat_id,
                text=rule.text,
                normalized_text=rule.normalized_text,
                embedding=rule.embedding,
                frozen_weight=rule.consistency_weight,
                reason=rule.distortion_reason or "Distorted rule purged",
            )

        # Cluster remaining rules
        remaining = list(gateway.rules.values())
        visited: Set[str] = set()
        created_elephants = 0
        merged_rule_count = 0

        for i, base in enumerate(remaining):
            if base.rule_id in visited:
                continue
            cluster = [base]
            for other in remaining[i + 1:]:
                if other.rule_id in visited:
                    continue
                if self._can_merge(base, other):
                    cluster.append(other)
            if len(cluster) >= 2:
                for node in cluster:
                    visited.add(node.rule_id)
                    gateway.rules.pop(node.rule_id, None)
                summary_text = " | ".join(n.text for n in cluster)
                merged_entities: List[str] = []
                for n in cluster:
                    for e in n.entities:
                        if e not in merged_entities:
                            merged_entities.append(e)
                el_id = f"el_{uuid.uuid4().hex[:8]}"
                gateway.elephants[el_id] = ElephantNode(
                    elephant_id=el_id,
                    topic_label=base.topic_label,
                    summary_text=summary_text,
                    entities=merged_entities,
                    embedding=self.embed_fn(summary_text),
                    member_rule_ids=[n.rule_id for n in cluster],
                    created_at=time.time(),
                    consistency_weight=sum(n.consistency_weight for n in cluster),
                )
                created_elephants += 1
                merged_rule_count += len(cluster)

        return CompactionReport(
            purged_distorted_rules=purged,
            created_elephants=created_elephants,
            merged_rule_count=merged_rule_count,
            remaining_active_nodes=len(gateway.rules) + len(gateway.elephants),
        )


class AdaptiveHybridRetriever:
    def __init__(
        self,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        drift_threshold: float = 0.65,
    ) -> None:
        self.embed_fn = embed_fn or default_embed_text
        self.drift_threshold = drift_threshold

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        return set(re.findall(r"[a-zA-Z0-9_]+", text.lower())) - _STOPWORDS

    def search(
        self,
        query: str,
        gateway: SemanticIngestionGateway,
        top_k: int = 3,
    ) -> List[SearchHit]:
        items: List[Tuple[str, str, str, List[float], float]] = []
        for rid, r in gateway.rules.items():
            if not r.is_distorted:
                items.append((rid, "rule", r.text, r.embedding, r.consistency_weight))
        for eid, e in gateway.elephants.items():
            items.append((eid, "elephant", e.summary_text, e.embedding, e.consistency_weight))

        if not items:
            return []

        doc_freq: Dict[str, int] = {}
        for _, _, text, _, _ in items:
            for tok in self._tokenize(text):
                doc_freq[tok] = doc_freq.get(tok, 0) + 1

        n_docs = len(items)
        q_emb = self.embed_fn(query)
        q_tokens = self._tokenize(query)
        q_raw_tokens = re.findall(r"[a-zA-Z0-9_]+", query)

        hits: List[SearchHit] = []
        for item_id, item_type, text, emb, weight in items:
            dense_sim = cosine_similarity(q_emb, emb)
            d_tokens = self._tokenize(text)
            matched = q_tokens & d_tokens
            idf_sum = 0.0
            rare_matched: List[str] = []
            for tok in matched:
                df = doc_freq.get(tok, 1)
                idf = math.log((n_docs + 1) / df) + 1.0
                idf_sum += idf
                if df <= max(1, n_docs // 3) or "_" in tok:
                    # Preserve original casing if present in query
                    orig = next((rt for rt in q_raw_tokens if rt.lower() == tok), tok)
                    rare_matched.append(orig)
            rare_score = min(1.0, idf_sum / 5.0)
            score = (0.40 * dense_sim) + (0.60 * rare_score) + (0.02 * min(weight, 5.0))
            hits.append(SearchHit(
                item_id=item_id,
                item_type=item_type,
                text=text,
                score=score,
                rare_anchors_matched=rare_matched,
            ))

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]


class TokenBudgetedFrontRoom:
    def __init__(
        self,
        gateway: SemanticIngestionGateway,
        retriever: Optional[AdaptiveHybridRetriever] = None,
        active_window_size: int = 2,
        max_context_tokens: int = 1764,
        max_retrieval_bundles: int = 3,
    ) -> None:
        self.gateway = gateway
        self.retriever = retriever or AdaptiveHybridRetriever()
        self.active_window_size = active_window_size
        self.max_context_tokens = max_context_tokens
        self.max_retrieval_bundles = max_retrieval_bundles
        self.active_turns: List[str] = []
        self.tracker = LatencyTracker()

    def process_turn(self, user_input: str) -> TurnAssembly:
        t0 = time.perf_counter()
        q_emb = self.retriever.embed_fn(user_input)

        max_sim = 1.0
        drifted = False
        if self.active_turns:
            sims = [
                cosine_similarity(q_emb, self.retriever.embed_fn(turn))
                for turn in self.active_turns[-self.active_window_size:]
            ]
            max_sim = max(sims)
            drifted = max_sim < self.retriever.drift_threshold

        hits = self.retriever.search(user_input, self.gateway, top_k=self.max_retrieval_bundles)

        budget = self.max_context_tokens
        used = count_tokens(user_input)
        selected_hits: List[SearchHit] = []
        for hit in hits:
            ht = count_tokens(hit.text)
            if used + ht <= budget:
                selected_hits.append(hit)
                used += ht

        for turn in reversed(self.active_turns[-self.active_window_size:]):
            tt = count_tokens(turn)
            if used + tt <= budget:
                used += tt

        self.active_turns.append(user_input)
        if len(self.active_turns) > self.active_window_size:
            self.active_turns = self.active_turns[-self.active_window_size:]

        elapsed_ms = max((time.perf_counter() - t0) * 1000.0, 0.01)
        self.tracker.record(elapsed_ms)

        return TurnAssembly(
            drifted=drifted,
            drift_similarity=max_sim,
            retrieved_memories=selected_hits,
            total_tokens=min(used, budget),
        )
