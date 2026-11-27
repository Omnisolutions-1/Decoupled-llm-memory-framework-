#!/usr/bin/env python3
"""
Decoupled LLM Memory Framework v2.1 — Semantic Ingestion Gateway & Upgrade Suite
================================================================================
Compatible with `src/layers/front_room_controller.py` and `warehouse.py` in:
https://github.com/Omnisolutions-1/Decoupled-llm-memory-framework-

Implements the 3-part v2.1 architectural upgrade:
1. SemanticIngestionGateway (Consistencies vs. Anomalies):
   - Strips conversational surface fluff while preserving core assertions & rare terms.
   - Folds repeated consistencies into a single weighted RuleNode instead of
     appending duplicate entries ("truth is definitive").
   - Routes contradictions and unplaced signals to open PuzzleTables, supporting
     non-destructive cross-table copy-linking and 100%-solid resolution.
2. HierarchicalCompactor ("As I carry more, I carry less"):
   - Zero blind TTL: valid RuleNodes and ConsolidatedElephants are never evicted
     solely for sitting idle.
   - Distortion-only purging: removes only proven-distorted/contradicted items.
   - Upward consolidation: merges related loose RuleNodes into fewer, denser
     ConsolidatedElephant macro-nodes.
3. AdaptiveHybridRetriever, TokenBudgetedFrontRoom, & LatencyTracker:
   - Combines corpus-wide BM25/IDF rare-term weighting with dense vector cosine
     similarity, dynamically adapting anchor selection & fusion weights per query.
   - Preserves conditional semantic drift gating (< 0.65 similarity triggers lookup).
   - Enforces a hard token ceiling (default 1,764 tokens) using BPE-aligned
     tokenization (tiktoken if installed, subword-aware fallback otherwise).
   - Instruments per-turn p50/p95 latency and drift-skip telemetry.
"""

from __future__ import annotations

import hashlib
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple


_STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "could", "did", "do", "does", "for", "from", "had", "has", "have", "he",
    "her", "here", "him", "his", "how", "i", "if", "in", "into", "is", "it",
    "its", "just", "me", "my", "no", "not", "of", "on", "or", "our", "out",
    "she", "so", "some", "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "to", "too", "up", "us", "was", "we", "were",
    "what", "when", "where", "which", "who", "will", "with", "would", "you",
    "your",
}

_FLUFF_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(hey|hi|hello|good morning|good afternoon|good evening|ok|okay|so|well|um|uh|like|basically|actually|honestly|literally|you know|i mean|to be honest|kind of|sort of)[,!\.\s]+", re.IGNORECASE),
    re.compile(r"\b(um+|uh+|like|basically|actually|honestly|literally|kind of|sort of|you know)\b", re.IGNORECASE),
    re.compile(r"\s*(thanks|thank you|appreciate it|lol|haha|cheers)[!.\s]*$", re.IGNORECASE),
)

_NEGATION_TOKENS: Set[str] = {
    "not", "never", "no", "dont", "don't", "doesnt", "doesn't", "didnt",
    "didn't", "isnt", "isn't", "arent", "aren't", "wasnt", "wasn't",
    "cant", "can't", "cannot", "wont", "won't", "false", "disabled",
    "deprecated", "removed", "avoid", "forbidden",
}


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """
    Accurate token counter replacing `len(text.split())`.
    Uses `tiktoken` if available; otherwise uses a subword-aware estimator
    that closely tracks BPE token counts across words and punctuation.
    """
    if not text:
        return 0
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding(model)
        return len(enc.encode(text))
    except Exception:
        pieces = re.findall(r"[A-Za-z0-9_]+|[^\w\s]", text)
        total = 0
        for p in pieces:
            if len(p) <= 6:
                total += 1
            else:
                total += math.ceil(len(p) / 5.0)
        return total


def tokenize_terms(text: str, drop_stopwords: bool = True) -> List[str]:
    """Extracts normalized lowercase terms for BM25/IDF and signature matching."""
    raw = re.findall(r"[a-z0-9_'-]+", text.lower())
    if not drop_stopwords:
        return raw
    return [t for t in raw if t not in _STOPWORDS and len(t) > 1]


def default_embed_text(text: str, dim: int = 64) -> List[float]:
    """
    Deterministic, zero-dependency feature-hashed dense vector (L2-normalized).
    Drop-in compatible with `all-MiniLM-L6-v2` embeddings via `embed_fn`.
    """
    vec = [0.0] * dim
    terms = tokenize_terms(text, drop_stopwords=False)
    if not terms:
        return vec
    for term in terms:
        digest = hashlib.sha256(term.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        weight = 1.0 if term in _STOPWORDS else 2.0
        vec[idx] += sign * weight
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0.0:
        vec = [x / norm for x in vec]
    return vec


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Computes cosine similarity between two dense vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


@dataclass
class SemanticPointer:
    """Compatible with `SemanticPointer` in `warehouse.py`."""
    turn_id: str
    timestamp: float
    topic_label: str
    entities: List[str]
    embedding: List[float]
    token_count: int


@dataclass
class ArchivedBundle:
    """Compatible with `ArchivedBundle` in `warehouse.py`."""
    pointer: SemanticPointer
    raw_content: str
    distilled_fact: Optional[str] = None


@dataclass
class RuleNode:
    """
    A weighted consistency node ("truth is definitive").
    Repeated consistent observations fold into this single node rather than
    creating duplicate records. Never evicted by blind TTL.
    """
    node_id: str
    canonical_text: str
    topic_label: str
    entities: List[str]
    embedding: List[float]
    token_count: int
    consistency_weight: float = 1.0
    evidence_count: int = 1
    source_turn_ids: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_reinforced_at: float = field(default_factory=time.time)
    is_distorted: bool = False
    distortion_confidence: float = 0.0
    distortion_reason: Optional[str] = None

    def reinforce(self, turn_id: str, weight_delta: float = 1.0, now: Optional[float] = None) -> None:
        self.consistency_weight += weight_delta
        self.evidence_count += 1
        if turn_id not in self.source_turn_ids:
            self.source_turn_ids.append(turn_id)
        self.last_reinforced_at = now if now is not None else time.time()


@dataclass
class PuzzlePiece:
    """
    An unresolved anomaly, contradiction, or unplaced signal held on an open
    PuzzleTable until verified (100% solid) or purged as proven distortion.
    Supports non-destructive copy-linking across multiple tables.
    """
    piece_id: str
    content: str
    topic_label: str
    entities: List[str]
    embedding: List[float]
    token_count: int
    anomaly_type: str  # "contradiction" | "unplaced" | "ambiguous"
    conflicting_rule_id: Optional[str] = None
    table_names: List[str] = field(default_factory=lambda: ["default"])
    source_turn_ids: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    confidence: float = 0.5
    status: str = "open"  # "open" | "resolved_to_rule" | "purged_distortion"


@dataclass
class ConsolidatedElephant:
    """
    A higher-order macro-node formed by merging related loose RuleNodes upward
    ("As I carry more, I carry less").
    """
    elephant_id: str
    summary_text: str
    topic_label: str
    entities: List[str]
    embedding: List[float]
    token_count: int
    child_rule_ids: List[str]
    combined_weight: float
    created_at: float = field(default_factory=time.time)
    last_updated_at: float = field(default_factory=time.time)
    is_distorted: bool = False


@dataclass
class CondemnedPattern:
    """
    Frozen-weight "wanted poster" stub left behind when a proven-distorted
    RuleNode is purged and moved to the audit archive. Incoming claims that
    match this shape are flagged as anomalies rather than re-ingested as truth.
    """
    pattern_id: str
    source_rule_id: str
    canonical_text: str
    topic_label: str
    entities: List[str]
    embedding: List[float]
    frozen_weight: float
    distortion_confidence: float
    distortion_reason: Optional[str] = None
    condemned_at: float = field(default_factory=time.time)


@dataclass
class IngestionOutcome:
    action: str  # "folded_consistency" | "created_rule" | "flagged_anomaly" | "dropped_fluff"
    stripped_text: str
    target_id: Optional[str] = None
    table_name: Optional[str] = None
    conflicting_rule_id: Optional[str] = None


class SemanticIngestionGateway:
    """
    Upgrades the append-only ingestion pipeline:
      1. Strips conversational surface fluff.
      2. Detects contradictions against existing RuleNodes.
      3. Folds repeated consistencies into one weighted RuleNode.
      4. Routes anomalies & unplaced pieces onto open PuzzleTables.
    """

    def __init__(
        self,
        embed_fn: Callable[[str], List[float]] = default_embed_text,
        consistency_threshold: float = 0.78,
        contradiction_topic_threshold: float = 0.45,
        min_signal_tokens: int = 3,
    ) -> None:
        self.embed_fn = embed_fn
        self.consistency_threshold = consistency_threshold
        self.contradiction_topic_threshold = contradiction_topic_threshold
        self.min_signal_tokens = min_signal_tokens

        self.rules: Dict[str, RuleNode] = {}
        self.puzzle_tables: Dict[str, Dict[str, PuzzlePiece]] = defaultdict(dict)
        self.elephants: Dict[str, ConsolidatedElephant] = {}
        self.condemned_patterns: Dict[str, CondemnedPattern] = {}
        self.audit_archive: List[Dict[str, Any]] = []

    @staticmethod
    def strip_surface_fluff(raw_text: str) -> str:
        """Removes greetings, filler hedges, and trailing conversational noise."""
        cleaned = raw_text.strip()
        for pattern in _FLUFF_PATTERNS:
            cleaned = pattern.sub(" ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;-")
        return cleaned

    @staticmethod
    def _extract_entities(text: str) -> List[str]:
        """Extracts capitalized identifiers, snake_case symbols, and numbers."""
        caps = re.findall(r"\b[A-Z][A-Za-z0-9_-]{2,}\b", text)
        symbols = re.findall(r"\b[a-z]+_[a-z0-9_]+\b", text)
        nums = re.findall(r"\b\d+(?:\.\d+)?\b", text)
        seen: Set[str] = set()
        entities: List[str] = []
        for item in caps + symbols + nums:
            if item not in seen:
                seen.add(item)
                entities.append(item)
        return entities

    @staticmethod
    def _has_negation(text: str) -> bool:
        tokens = set(re.findall(r"[a-z0-9_']+", text.lower()))
        return bool(tokens & _NEGATION_TOKENS)

    @staticmethod
    def _extract_numeric_claims(text: str) -> List[str]:
        return re.findall(r"\b\d+(?:\.\d+)?\b", text)

    def _is_contradiction(self, text: str, emb: List[float], rule: RuleNode) -> bool:
        """
        Detects semantic contradiction against an existing RuleNode:
        - High topical overlap (shared entities or cosine >= contradiction_topic_threshold)
        - Opposing polarity (negation mismatch) OR conflicting numeric values on same subject.
        """
        if rule.is_distorted:
            return False
        sim = cosine_similarity(emb, rule.embedding)
        shared_entities = set(self._extract_entities(text)) & set(rule.entities)
        if sim < self.contradiction_topic_threshold and not shared_entities:
            return False

        neg_new = self._has_negation(text)
        neg_old = self._has_negation(rule.canonical_text)
        if neg_new != neg_old and (sim >= 0.35 or len(shared_entities) >= 1):
            return True

        nums_new = self._extract_numeric_claims(text)
        nums_old = self._extract_numeric_claims(rule.canonical_text)
        if nums_new and nums_old and set(nums_new) != set(nums_old):
            terms_new = set(tokenize_terms(text)) - set(nums_new)
            terms_old = set(tokenize_terms(rule.canonical_text)) - set(nums_old)
            overlap = len(terms_new & terms_old)
            if overlap >= 2 or len(shared_entities) >= 1:
                return True

        return False

    def ingest(
        self,
        raw_text: str,
        turn_id: str,
        topic_label: str = "general",
        entities: Optional[List[str]] = None,
        is_unplaced: bool = False,
        table_name: Optional[str] = None,
        now: Optional[float] = None,
    ) -> IngestionOutcome:
        """
        Ingests a raw conversational turn or memory candidate:
        - Drops pure surface fluff.
        - Checks against frozen-weight CondemnedPattern stubs -> routes to PuzzleTable.
        - Checks for contradictions -> routes to PuzzleTable.
        - Checks for consistent match -> folds into existing RuleNode.
        - Otherwise creates a new RuleNode (or PuzzlePiece if `is_unplaced=True`).
        """
        ts = now if now is not None else time.time()
        stripped = self.strip_surface_fluff(raw_text)
        tok_count = count_tokens(stripped)
        if tok_count < self.min_signal_tokens:
            return IngestionOutcome(action="dropped_fluff", stripped_text=stripped)

        emb = self.embed_fn(stripped)
        ent_list = entities if entities is not None else self._extract_entities(stripped)

        # 0. Check against frozen-weight CondemnedPattern ("wanted poster") stubs
        terms_new_condemned = set(tokenize_terms(stripped))
        for pattern in self.condemned_patterns.values():
            sim = cosine_similarity(emb, pattern.embedding)
            terms_old = set(tokenize_terms(pattern.canonical_text))
            jaccard = len(terms_new_condemned & terms_old) / max(1, len(terms_new_condemned | terms_old))
            combined = 0.7 * sim + 0.3 * jaccard
            if sim >= self.consistency_threshold or combined >= 0.75:
                t_name = table_name or f"condemned_{topic_label}"
                piece_id = f"pz_{hashlib.md5((turn_id + stripped).encode()).hexdigest()[:10]}"
                piece = PuzzlePiece(
                    piece_id=piece_id,
                    content=stripped,
                    topic_label=topic_label,
                    entities=ent_list,
                    embedding=emb,
                    token_count=tok_count,
                    anomaly_type="condemned_match",
                    conflicting_rule_id=pattern.source_rule_id,
                    table_names=[t_name],
                    source_turn_ids=[turn_id],
                    created_at=ts,
                    confidence=pattern.distortion_confidence,
                )
                self.puzzle_tables[t_name][piece_id] = piece
                return IngestionOutcome(
                    action="flagged_anomaly",
                    stripped_text=stripped,
                    target_id=piece_id,
                    table_name=t_name,
                    conflicting_rule_id=pattern.source_rule_id,
                )

        # 1. Check for contradiction against active RuleNodes
        for rule in self.rules.values():
            if self._is_contradiction(stripped, emb, rule):
                t_name = table_name or f"anomaly_{topic_label}"
                piece_id = f"pz_{hashlib.md5((turn_id + stripped).encode()).hexdigest()[:10]}"
                piece = PuzzlePiece(
                    piece_id=piece_id,
                    content=stripped,
                    topic_label=topic_label,
                    entities=ent_list,
                    embedding=emb,
                    token_count=tok_count,
                    anomaly_type="contradiction",
                    conflicting_rule_id=rule.node_id,
                    table_names=[t_name],
                    source_turn_ids=[turn_id],
                    created_at=ts,
                )
                self.puzzle_tables[t_name][piece_id] = piece
                return IngestionOutcome(
                    action="flagged_anomaly",
                    stripped_text=stripped,
                    target_id=piece_id,
                    table_name=t_name,
                    conflicting_rule_id=rule.node_id,
                )

        # 2. If explicitly unplaced/speculative, put on an open PuzzleTable
        if is_unplaced:
            t_name = table_name or f"open_{topic_label}"
            piece_id = f"pz_{hashlib.md5((turn_id + stripped).encode()).hexdigest()[:10]}"
            piece = PuzzlePiece(
                piece_id=piece_id,
                content=stripped,
                topic_label=topic_label,
                entities=ent_list,
                embedding=emb,
                token_count=tok_count,
                anomaly_type="unplaced",
                table_names=[t_name],
                source_turn_ids=[turn_id],
                created_at=ts,
            )
            self.puzzle_tables[t_name][piece_id] = piece
            return IngestionOutcome(
                action="flagged_anomaly",
                stripped_text=stripped,
                target_id=piece_id,
                table_name=t_name,
            )

        # 3. Check for consistency folding into an existing RuleNode
        best_rule: Optional[RuleNode] = None
        best_sim = -1.0
        terms_new = set(tokenize_terms(stripped))
        for rule in self.rules.values():
            if rule.is_distorted:
                continue
            sim = cosine_similarity(emb, rule.embedding)
            terms_old = set(tokenize_terms(rule.canonical_text))
            jaccard = (
                len(terms_new & terms_old) / max(1, len(terms_new | terms_old))
            )
            combined = 0.7 * sim + 0.3 * jaccard
            if (sim >= self.consistency_threshold or combined >= 0.75) and combined > best_sim:
                best_sim = combined
                best_rule = rule

        if best_rule is not None:
            best_rule.reinforce(turn_id=turn_id, weight_delta=1.0, now=ts)
            for e in ent_list:
                if e not in best_rule.entities:
                    best_rule.entities.append(e)
            return IngestionOutcome(
                action="folded_consistency",
                stripped_text=stripped,
                target_id=best_rule.node_id,
            )

        # 4. Otherwise create a new RuleNode
        rule_id = f"rule_{hashlib.md5((turn_id + stripped).encode()).hexdigest()[:10]}"
        new_rule = RuleNode(
            node_id=rule_id,
            canonical_text=stripped,
            topic_label=topic_label,
            entities=ent_list,
            embedding=emb,
            token_count=tok_count,
            source_turn_ids=[turn_id],
            created_at=ts,
            last_reinforced_at=ts,
        )
        self.rules[rule_id] = new_rule
        return IngestionOutcome(
            action="created_rule",
            stripped_text=stripped,
            target_id=rule_id,
        )

    def copy_piece_to_table(self, piece_id: str, source_table: str, target_table: str) -> bool:
        """
        Copies a PuzzlePiece onto another PuzzleTable WITHOUT removing it from
        its original table, preserving cross-pattern links.
        """
        src = self.puzzle_tables.get(source_table, {})
        piece = src.get(piece_id)
        if piece is None:
            return False
        if target_table not in piece.table_names:
            piece.table_names.append(target_table)
        self.puzzle_tables[target_table][piece_id] = piece
        return True

    def resolve_puzzle_piece(
        self,
        piece_id: str,
        table_name: str,
        is_valid_truth: bool,
        supersedes_conflicting_rule: bool = False,
        now: Optional[float] = None,
    ) -> Optional[str]:
        """
        Resolves an open PuzzlePiece once 100% solid:
        - If `is_valid_truth=True`, promotes it to a RuleNode (and optionally
          marks the contradicted rule as distorted).
        - If `is_valid_truth=False`, marks it as `purged_distortion` and removes
          it from open puzzle tables.
        """
        ts = now if now is not None else time.time()
        table = self.puzzle_tables.get(table_name, {})
        piece = table.get(piece_id)
        if piece is None:
            return None

        if not is_valid_truth:
            piece.status = "purged_distortion"
            for t in list(piece.table_names):
                self.puzzle_tables.get(t, {}).pop(piece_id, None)
            return None

        if supersedes_conflicting_rule and piece.conflicting_rule_id:
            old_rule = self.rules.get(piece.conflicting_rule_id)
            if old_rule is not None:
                old_rule.is_distorted = True
                old_rule.distortion_confidence = 1.0
                old_rule.distortion_reason = f"Superseded by verified piece {piece_id}"

        rule_id = f"rule_{hashlib.md5((piece.piece_id + piece.content).encode()).hexdigest()[:10]}"
        self.rules[rule_id] = RuleNode(
            node_id=rule_id,
            canonical_text=piece.content,
            topic_label=piece.topic_label,
            entities=list(piece.entities),
            embedding=list(piece.embedding),
            token_count=piece.token_count,
            consistency_weight=1.5,
            evidence_count=len(piece.source_turn_ids),
            source_turn_ids=list(piece.source_turn_ids),
            created_at=ts,
            last_reinforced_at=ts,
        )
        piece.status = "resolved_to_rule"
        for t in list(piece.table_names):
            self.puzzle_tables.get(t, {}).pop(piece_id, None)
        return rule_id


@dataclass
class CompactionReport:
    purged_distorted_rules: int
    purged_distorted_elephants: int
    merged_rule_count: int
    created_elephants: int
    remaining_active_nodes: int


class HierarchicalCompactor:
    """
    Replaces blind TTL eviction with distortion-only purging and upward
    hierarchical consolidation ("100 loose pieces -> fewer, tighter elephants").
    """

    def __init__(
        self,
        embed_fn: Callable[[str], List[float]] = default_embed_text,
        distortion_threshold: float = 0.8,
        min_cluster_size: int = 2,
        cluster_similarity_threshold: float = 0.35,
    ) -> None:
        self.embed_fn = embed_fn
        self.distortion_threshold = distortion_threshold
        self.min_cluster_size = min_cluster_size
        self.cluster_similarity_threshold = cluster_similarity_threshold

    def compact(self, gateway: SemanticIngestionGateway, now: Optional[float] = None) -> CompactionReport:
        """
        1. Never deletes valid nodes solely for sitting idle (no blind TTL).
        2. Purges only proven-distorted nodes (`is_distorted=True` & confidence >= threshold).
        3. Clusters remaining RuleNodes by topic/entity/semantic similarity and
           merges clusters upward into ConsolidatedElephant macro-nodes.
        """
        ts = now if now is not None else time.time()

        # Step 1: Distortion-only purge -> audit archive + frozen-weight CondemnedPattern stub
        purged_rules = 0
        for rid in list(gateway.rules.keys()):
            rule = gateway.rules[rid]
            if rule.is_distorted and rule.distortion_confidence >= self.distortion_threshold:
                gateway.audit_archive.append({
                    "node_id": rule.node_id,
                    "canonical_text": rule.canonical_text,
                    "topic_label": rule.topic_label,
                    "entities": list(rule.entities),
                    "consistency_weight": rule.consistency_weight,
                    "evidence_count": rule.evidence_count,
                    "source_turn_ids": list(rule.source_turn_ids),
                    "distortion_confidence": rule.distortion_confidence,
                    "distortion_reason": rule.distortion_reason,
                    "archived_at": ts,
                })
                pattern_id = f"cond_{rule.node_id}"
                gateway.condemned_patterns[pattern_id] = CondemnedPattern(
                    pattern_id=pattern_id,
                    source_rule_id=rule.node_id,
                    canonical_text=rule.canonical_text,
                    topic_label=rule.topic_label,
                    entities=list(rule.entities),
                    embedding=list(rule.embedding),
                    frozen_weight=rule.consistency_weight,
                    distortion_confidence=rule.distortion_confidence,
                    distortion_reason=rule.distortion_reason,
                    condemned_at=ts,
                )
                gateway.rules.pop(rid, None)
                purged_rules += 1

        purged_elephants = 0
        for eid in list(gateway.elephants.keys()):
            if gateway.elephants[eid].is_distorted:
                del gateway.elephants[eid]
                purged_elephants += 1

        # Step 2: Upward consolidation of remaining RuleNodes into Elephants
        by_topic: Dict[str, List[RuleNode]] = defaultdict(list)
        for rule in gateway.rules.values():
            by_topic[rule.topic_label].append(rule)

        merged_count = 0
        created_elephants = 0

        for topic, nodes in by_topic.items():
            if len(nodes) < self.min_cluster_size:
                continue

            used: Set[str] = set()
            for i, seed in enumerate(nodes):
                if seed.node_id in used:
                    continue
                cluster = [seed]
                used.add(seed.node_id)
                seed_entities = set(seed.entities)

                for other in nodes[i + 1:]:
                    if other.node_id in used:
                        continue
                    sim = cosine_similarity(seed.embedding, other.embedding)
                    shared_ent = bool(seed_entities & set(other.entities))
                    non_num_seed = {e for e in seed.entities if not re.fullmatch(r"\d+(?:\.\d+)?", e)}
                    non_num_other = {e for e in other.entities if not re.fullmatch(r"\d+(?:\.\d+)?", e)}
                    shared_non_numeric = non_num_seed & non_num_other
                    if (sim >= self.cluster_similarity_threshold and shared_ent) or len(shared_non_numeric) >= 2:
                        cluster.append(other)
                        used.add(other.node_id)

                if len(cluster) >= self.min_cluster_size:
                    ordered = sorted(cluster, key=lambda r: r.consistency_weight, reverse=True)
                    summary_parts = [r.canonical_text.rstrip(".") for r in ordered]
                    summary_text = f"[{topic}] " + "; ".join(summary_parts) + "."
                    combined_entities: List[str] = []
                    for r in ordered:
                        for e in r.entities:
                            if e not in combined_entities:
                                combined_entities.append(e)

                    elephant_id = f"ele_{hashlib.md5(summary_text.encode()).hexdigest()[:10]}"
                    elephant = ConsolidatedElephant(
                        elephant_id=elephant_id,
                        summary_text=summary_text,
                        topic_label=topic,
                        entities=combined_entities,
                        embedding=self.embed_fn(summary_text),
                        token_count=count_tokens(summary_text),
                        child_rule_ids=[r.node_id for r in ordered],
                        combined_weight=sum(r.consistency_weight for r in ordered),
                        created_at=ts,
                        last_updated_at=ts,
                    )
                    gateway.elephants[elephant_id] = elephant
                    for r in ordered:
                        gateway.rules.pop(r.node_id, None)
                    merged_count += len(ordered)
                    created_elephants += 1

        remaining = len(gateway.rules) + len(gateway.elephants)
        return CompactionReport(
            purged_distorted_rules=purged_rules,
            purged_distorted_elephants=purged_elephants,
            merged_rule_count=merged_count,
            created_elephants=created_elephants,
            remaining_active_nodes=remaining,
        )


@dataclass
class SearchHit:
    item_id: str
    item_type: str  # "elephant" | "rule" | "puzzle"
    text: str
    token_count: int
    hybrid_score: float
    bm25_score: float
    dense_score: float
    rare_anchors_matched: List[str]


class AdaptiveHybridRetriever:
    """
    Replaces hardcoded keyword matching & dense-only search with:
      - Corpus-wide IDF rare-term weighting + BM25 scoring.
      - Dense vector cosine similarity.
      - Per-query adaptive fusion: boosts rare-anchor BM25 weight when query
        contains high-IDF distinctive terms; leans on dense shape matching for
        broad conceptual queries.
    """

    def __init__(
        self,
        embed_fn: Callable[[str], List[float]] = default_embed_text,
        k1: float = 1.5,
        b: float = 0.75,
        drift_threshold: float = 0.65,
    ) -> None:
        self.embed_fn = embed_fn
        self.k1 = k1
        self.b = b
        self.drift_threshold = drift_threshold

    def detect_semantic_drift(
        self,
        query: str,
        active_context_text: str,
    ) -> Tuple[bool, float]:
        """
        Preserves v2.0 conditional demand-paging gate:
        Returns (drifted, similarity). Lookup is triggered only if sim < 0.65.
        """
        if not active_context_text.strip():
            return True, 0.0
        q_emb = self.embed_fn(query)
        c_emb = self.embed_fn(active_context_text)
        sim = cosine_similarity(q_emb, c_emb)
        return sim < self.drift_threshold, sim

    def search(
        self,
        query: str,
        gateway: SemanticIngestionGateway,
        top_k: int = 5,
        include_puzzles: bool = True,
        min_score: float = 0.05,
    ) -> List[SearchHit]:
        corpus: List[Tuple[str, str, str, List[float], int, float]] = []
        for ele in gateway.elephants.values():
            if not ele.is_distorted:
                corpus.append((
                    ele.elephant_id, "elephant", ele.summary_text,
                    ele.embedding, ele.token_count, ele.combined_weight,
                ))
        for rule in gateway.rules.values():
            if not rule.is_distorted:
                corpus.append((
                    rule.node_id, "rule", rule.canonical_text,
                    rule.embedding, rule.token_count, rule.consistency_weight,
                ))
        if include_puzzles:
            seen_puzzles: Set[str] = set()
            for table in gateway.puzzle_tables.values():
                for piece in table.values():
                    if piece.piece_id not in seen_puzzles and piece.status == "open":
                        seen_puzzles.add(piece.piece_id)
                        corpus.append((
                            piece.piece_id, "puzzle", piece.content,
                            piece.embedding, piece.token_count, 0.5,
                        ))

        if not corpus:
            return []

        n_docs = len(corpus)
        doc_terms: List[List[str]] = [tokenize_terms(c[2]) for c in corpus]
        df: Counter[str] = Counter()
        for terms in doc_terms:
            for t in set(terms):
                df[t] += 1

        avgdl = sum(len(t) for t in doc_terms) / max(1, n_docs)

        def idf(term: str) -> float:
            freq = df.get(term, 0)
            return math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1.0)

        q_terms = tokenize_terms(query)
        q_idfs = {t: idf(t) for t in set(q_terms)}

        max_possible_idf = math.log((n_docs + 0.5) / 0.5 + 1.0)
        rare_threshold = max_possible_idf * 0.55
        rare_anchors = [t for t, val in sorted(q_idfs.items(), key=lambda kv: -kv[1]) if val >= rare_threshold]

        if rare_anchors:
            bm25_weight = 0.65
            dense_weight = 0.35
        else:
            bm25_weight = 0.35
            dense_weight = 0.65

        q_emb = self.embed_fn(query)
        raw_bm25: List[float] = []
        raw_dense: List[float] = []
        matched_anchors_per_doc: List[List[str]] = []

        for idx, (item_id, item_type, text, emb, tok_cnt, weight) in enumerate(corpus):
            terms = doc_terms[idx]
            tf = Counter(terms)
            dl = max(1, len(terms))
            score_bm25 = 0.0
            matched: List[str] = []
            for t in set(q_terms):
                if tf[t] > 0:
                    term_idf = q_idfs[t]
                    denom = tf[t] + self.k1 * (1.0 - self.b + self.b * (dl / max(1.0, avgdl)))
                    score_bm25 += term_idf * ((tf[t] * (self.k1 + 1.0)) / denom)
                    if t in rare_anchors:
                        matched.append(t)
            raw_bm25.append(score_bm25)
            raw_dense.append(max(0.0, cosine_similarity(q_emb, emb)))
            matched_anchors_per_doc.append(matched)

        max_bm25 = max(raw_bm25) if max(raw_bm25) > 0 else 1.0
        hits: List[SearchHit] = []
        for idx, (item_id, item_type, text, emb, tok_cnt, weight) in enumerate(corpus):
            norm_bm25 = raw_bm25[idx] / max_bm25 if max_bm25 > 0 else 0.0
            dense_s = raw_dense[idx]
            weight_boost = min(0.15, 0.03 * math.log1p(weight))
            hybrid = bm25_weight * norm_bm25 + dense_weight * dense_s + weight_boost
            if hybrid >= min_score:
                hits.append(SearchHit(
                    item_id=item_id,
                    item_type=item_type,
                    text=text,
                    token_count=tok_cnt,
                    hybrid_score=round(hybrid, 4),
                    bm25_score=round(norm_bm25, 4),
                    dense_score=round(dense_s, 4),
                    rare_anchors_matched=matched_anchors_per_doc[idx],
                ))

        hits.sort(key=lambda h: h.hybrid_score, reverse=True)
        return hits[:top_k]


class LatencyTracker:
    """Tracks per-turn latency in milliseconds and computes p50/p95 metrics."""

    def __init__(self) -> None:
        self.turn_latencies_ms: List[float] = []
        self.drift_skips: int = 0
        self.total_turns: int = 0

    def record(self, elapsed_ms: float, skipped_warehouse: bool) -> None:
        self.turn_latencies_ms.append(elapsed_ms)
        self.total_turns += 1
        if skipped_warehouse:
            self.drift_skips += 1

    @staticmethod
    def _percentile(sorted_vals: List[float], pct: float) -> float:
        if not sorted_vals:
            return 0.0
        if len(sorted_vals) == 1:
            return sorted_vals[0]
        k = (len(sorted_vals) - 1) * pct
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_vals[int(k)]
        return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)

    def summary(self) -> Dict[str, Any]:
        ordered = sorted(self.turn_latencies_ms)
        return {
            "total_turns": self.total_turns,
            "drift_skip_rate": round(self.drift_skips / max(1, self.total_turns), 4),
            "p50_ms": round(self._percentile(ordered, 0.50), 3),
            "p95_ms": round(self._percentile(ordered, 0.95), 3),
            "mean_ms": round(statistics.mean(ordered), 3) if ordered else 0.0,
        }


@dataclass
class FrontRoomAssembly:
    active_turns: List[str]
    retrieved_memories: List[SearchHit]
    total_tokens: int
    token_budget: int
    drifted: bool
    drift_similarity: float
    latency_ms: float


class TokenBudgetedFrontRoom:
    """
    Drop-in v2.1 upgrade for `FrontRoomController` (`src/layers/front_room_controller.py`).
    Enforces a hard token ceiling (default 1,764 tokens) instead of turn-count-only
    slicing, while preserving the O(1) active window and conditional drift gate.
    """

    def __init__(
        self,
        gateway: SemanticIngestionGateway,
        retriever: Optional[AdaptiveHybridRetriever] = None,
        active_window_size: int = 2,
        max_context_tokens: int = 1764,
        max_retrieval_bundles: int = 3,
    ) -> None:
        self.gateway = gateway
        self.retriever = retriever or AdaptiveHybridRetriever(embed_fn=gateway.embed_fn)
        self.active_window_size = active_window_size
        self.max_context_tokens = max_context_tokens
        self.max_retrieval_bundles = max_retrieval_bundles
        self.active_turns: List[str] = []
        self.tracker = LatencyTracker()

    def process_turn(self, user_query: str) -> FrontRoomAssembly:
        t0 = time.perf_counter()

        recent = self.active_turns[-self.active_window_size:] if self.active_window_size > 0 else []
        packed_active: List[str] = []
        used_tokens = count_tokens(user_query)

        for turn in reversed(recent):
            t_tok = count_tokens(turn)
            if used_tokens + t_tok <= self.max_context_tokens:
                packed_active.insert(0, turn)
                used_tokens += t_tok

        active_context_str = " ".join(packed_active)
        drifted, sim = self.retriever.detect_semantic_drift(user_query, active_context_str)

        retrieved: List[SearchHit] = []
        if drifted:
            candidates = self.retriever.search(
                query=user_query,
                gateway=self.gateway,
                top_k=self.max_retrieval_bundles * 2,
            )
            for hit in candidates:
                if len(retrieved) >= self.max_retrieval_bundles:
                    break
                if used_tokens + hit.token_count <= self.max_context_tokens:
                    retrieved.append(hit)
                    used_tokens += hit.token_count

        self.active_turns.append(user_query)
        if len(self.active_turns) > self.active_window_size:
            self.active_turns = self.active_turns[-self.active_window_size:]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.tracker.record(elapsed_ms, skipped_warehouse=(not drifted))

        return FrontRoomAssembly(
            active_turns=packed_active,
            retrieved_memories=retrieved,
            total_tokens=used_tokens,
            token_budget=self.max_context_tokens,
            drifted=drifted,
            drift_similarity=round(sim, 4),
            latency_ms=round(elapsed_ms, 3),
        )


if __name__ == "__main__":
    gw = SemanticIngestionGateway()
    gw.ingest("Hey, so basically QdrantWarehouse uses cosine similarity threshold 0.5.", "t1", topic_label="warehouse")
    gw.ingest("Um QdrantWarehouse uses cosine similarity threshold 0.5, thanks!", "t2", topic_label="warehouse")
    gw.ingest("Actually QdrantWarehouse does not use cosine similarity threshold 0.5.", "t3", topic_label="warehouse")
    compactor = HierarchicalCompactor()
    report = compactor.compact(gw)
    fr = TokenBudgetedFrontRoom(gw)
    res = fr.process_turn("What threshold does QdrantWarehouse use?")
    print("Smoke test:", {"report": report, "assembly_tokens": res.total_tokens, "latency_summary": fr.tracker.summary()})
