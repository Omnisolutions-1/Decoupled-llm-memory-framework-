# Decoupled LLM Memory Framework v2.1
### O(1) Context Management, Semantic Ingestion Gateway & Adaptive Hybrid Retrieval

An event-driven, decoupled memory framework designed to enforce a permanently flat LLM context window footprint, eliminate token accumulation bloat, and preserve rare technical details across long-running agentic sessions.

## What's New in v2.1

- **O(1) Token-Budgeted Front Room (`FrontRoomController` + `TokenBudgetedFrontRoom`)**: Enforces both a fixed active turn window and a hard token ceiling (`max_context_tokens = 1764` by default) using BPE-aligned tokenization (`tiktoken` when installed, subword-aware fallback otherwise), packing in strict priority order: `user_query` -> `active_turns` -> `retrieved_memories`.
- **Preserved Semantic Drift Gate**: Evaluates cosine similarity between incoming queries and the active window (`< 0.65` triggers warehouse and hybrid memory lookup; on-topic turns short-circuit without external retrieval).
- **Semantic Ingestion Gateway (`SemanticIngestionGateway`)**:
  - **Surface Fluff Stripping**: Removes conversational filler and greetings while preserving core assertions, entities, and rare terms.
  - **Consistency-Weight Folding**: Folds repeated consistent claims into a single weighted `RuleNode` rather than appending duplicates.
  - **Open Puzzle Tables**: Routes contradictions and unplaced signals to `PuzzleTable` partitions as `PuzzlePiece` records with non-destructive cross-table copy-linking and `resolve_puzzle_piece()` as the caller/LLM-judge hook for `distortion_confidence`.
- **Distortion-Only Hierarchical Compaction (`HierarchicalCompactor`)**:
  - **Zero Blind TTL**: Valid `RuleNode` and `ConsolidatedElephant` records are never evicted for sitting idle.
  - **Audit Archive & `CondemnedPattern` Stubs**: Proven-distorted rules (`is_distorted=True` and `distortion_confidence >= threshold`) are archived to `audit_archive` and replaced with frozen-weight `CondemnedPattern` ("wanted poster") stubs so matching-shape arrivals are flagged on `ingest()`.
  - **Upward Consolidation**: Merges semantically related `RuleNode` clusters (`sim >= cluster_similarity_threshold` with `shared_ent`, or `>= 2` shared non-numeric entities) into denser `ConsolidatedElephant` macro-nodes.
- **Adaptive Rare-Detail Hybrid Search (`AdaptiveHybridRetriever`)**:
  - Dynamically switches fusion weights (`0.65` BM25 / `0.35` Dense when high-IDF rare anchors are present; `0.35` / `0.65` otherwise) with a bounded `0.15` log-cap on `consistency_weight` (`min(0.15, 0.03 * log1p(weight))`).
- **Latency Observability (`LatencyTracker`)**: Tracks per-turn `p50_ms`, `p95_ms`, `mean_ms`, and `drift_skip_rate`.

## Repository Structure

```text
Decoupled-llm-memory-framework-/
├── src/
│   ├── __init__.py
│   └── layers/
│       ├── __init__.py
│       └── front_room_controller.py
├── semantic_ingestion_gateway.py
├── warehouse.py
├── requirements.txt
├── test_semantic_ingestion_gateway.py
├── test_front_room_integration.py
├── daemon.py
├── front_room.py
└── README.md
```

## Quickstart

Works out of the box with Python 3.9+ and `numpy` (heavy ML dependencies like `sentence-transformers`, `scikit-learn`, `tiktoken`, and `qdrant-client` are optional and degrade gracefully).

```bash
# 1. Install core (or full optional) dependencies
pip install -r requirements.txt

# 2. Run the full test suite
python3 -m unittest discover -v

# 3. Run the standalone smoke tests
python3 semantic_ingestion_gateway.py
python3 src/layers/front_room_controller.py
```

## Status

- [x] v1.0 — Functional prototype. Keyword-filter ingestion.
- [x] v2.0 — Semantic drift detection, modular warehouse adapters, strict O(1) context window.
- [x] v2.1 — Semantic Ingestion Gateway, PuzzleTables, `CondemnedPattern` audit stubs, distortion-only Hierarchical Compactor, Adaptive Hybrid Search, and hard 1,764-token Front Room ceiling.
