import time
import unittest

from semantic_ingestion_gateway import (
    AdaptiveHybridRetriever,
    HierarchicalCompactor,
    SemanticIngestionGateway,
    TokenBudgetedFrontRoom,
    count_tokens,
)


class TestDecoupledMemoryUpgradeV21(unittest.TestCase):
    def test_1_fluff_stripping_and_consistency_folding(self):
        gw = SemanticIngestionGateway()
        r1 = gw.ingest(
            "Hey, um, basically FrontRoomController uses active_window_size 2 turns, thanks!",
            turn_id="t1",
            topic_label="front_room",
        )
        self.assertEqual(r1.action, "created_rule")
        self.assertNotIn("Hey", r1.stripped_text)
        self.assertNotIn("thanks", r1.stripped_text.lower())

        r2 = gw.ingest(
            "Hello, FrontRoomController uses active_window_size 2 turns.",
            turn_id="t2",
            topic_label="front_room",
        )
        self.assertEqual(r2.action, "folded_consistency")
        self.assertEqual(r2.target_id, r1.target_id)
        self.assertEqual(len(gw.rules), 1)
        self.assertEqual(gw.rules[r1.target_id].evidence_count, 2)
        self.assertGreater(gw.rules[r1.target_id].consistency_weight, 1.0)

    def test_2_anomaly_puzzle_table_and_cross_table_copy(self):
        gw = SemanticIngestionGateway()
        r1 = gw.ingest(
            "QdrantWarehouse uses cosine similarity threshold 0.5.",
            turn_id="t1",
            topic_label="warehouse",
        )
        a1 = gw.ingest(
            "QdrantWarehouse does not use cosine similarity threshold 0.5.",
            turn_id="t2",
            topic_label="warehouse",
        )
        self.assertEqual(a1.action, "flagged_anomaly")
        self.assertIsNotNone(a1.target_id)
        self.assertEqual(a1.conflicting_rule_id, r1.target_id)

        copied = gw.copy_piece_to_table(a1.target_id, a1.table_name, "retrieval_audit")
        self.assertTrue(copied)
        self.assertIn(a1.target_id, gw.puzzle_tables[a1.table_name])
        self.assertIn(a1.target_id, gw.puzzle_tables["retrieval_audit"])

        new_rule_id = gw.resolve_puzzle_piece(
            a1.target_id,
            a1.table_name,
            is_valid_truth=True,
            supersedes_conflicting_rule=True,
        )
        self.assertIsNotNone(new_rule_id)
        self.assertTrue(gw.rules[r1.target_id].is_distorted)
        self.assertNotIn(a1.target_id, gw.puzzle_tables["retrieval_audit"])

    def test_3_hierarchical_compaction_no_blind_ttl(self):
        gw = SemanticIngestionGateway()
        old_ts = time.time() - (365 * 24 * 3600)
        gw.ingest(
            "SQLite3 LocalWarehouse stores SemanticPointer metadata.",
            turn_id="t_old",
            topic_label="storage",
            now=old_ts,
        )
        gw.ingest(
            "SQLite3 LocalWarehouse persists ArchivedBundle historical turns.",
            turn_id="t_old2",
            topic_label="storage",
            now=old_ts,
        )
        r_bad = gw.ingest(
            "Deprecated temporary cache uses invalid port 99999.",
            turn_id="t_bad",
            topic_label="misc",
            now=time.time(),
        )
        gw.rules[r_bad.target_id].is_distorted = True
        gw.rules[r_bad.target_id].distortion_confidence = 0.95

        compactor = HierarchicalCompactor()
        report = compactor.compact(gw)

        self.assertEqual(report.purged_distorted_rules, 1)
        self.assertNotIn(r_bad.target_id, gw.rules)
        self.assertEqual(report.created_elephants, 1)
        self.assertEqual(report.merged_rule_count, 2)
        self.assertEqual(len(gw.elephants), 1)
        elephant = next(iter(gw.elephants.values()))
        self.assertIn("LocalWarehouse", elephant.summary_text)

    def test_4_adaptive_rare_detail_hybrid_search(self):
        gw = SemanticIngestionGateway()
        for i in range(5):
            gw.ingest(
                f"Standard memory pipeline step {i} processes active context turns.",
                turn_id=f"c_{i}",
                topic_label="pipeline",
            )
        rare_res = gw.ingest(
            "ConnectionShield uses xylophone_mutex to prevent Qdrant fallback races.",
            turn_id="rare_1",
            topic_label="resilience",
        )

        retriever = AdaptiveHybridRetriever()
        hits = retriever.search("Why does ConnectionShield need xylophone_mutex?", gw, top_k=3)
        self.assertGreaterEqual(len(hits), 1)
        self.assertEqual(hits[0].item_id, rare_res.target_id)
        self.assertIn("xylophone_mutex", hits[0].rare_anchors_matched)

    def test_5_hard_token_ceiling_and_latency_telemetry(self):
        gw = SemanticIngestionGateway()
        gw.ingest(
            "HandoverDaemon ding_threshold 1 triggers snapshot persistence.",
            turn_id="t1",
            topic_label="daemon",
        )
        fr = TokenBudgetedFrontRoom(gw, max_context_tokens=120)
        long_turn = " ".join(["architecture"] * 150)
        fr.active_turns.append(long_turn)

        assembly = fr.process_turn("Explain HandoverDaemon ding_threshold.")
        self.assertLessEqual(assembly.total_tokens, 120)
        self.assertEqual(count_tokens("hello world"), 2)

        fr.process_turn("Explain HandoverDaemon ding_threshold.")
        stats = fr.tracker.summary()
        self.assertEqual(stats["total_turns"], 2)
        self.assertGreaterEqual(stats["p95_ms"], stats["p50_ms"])
        self.assertGreater(stats["p50_ms"], 0.0)

    def test_6_condemned_pattern_stub_and_audit_archive(self):
        gw = SemanticIngestionGateway()
        bad = gw.ingest(
            "LegacyCache uses insecure default port 99999.",
            turn_id="t_bad",
            topic_label="security",
        )
        gw.rules[bad.target_id].is_distorted = True
        gw.rules[bad.target_id].distortion_confidence = 0.95
        gw.rules[bad.target_id].distortion_reason = "Verified invalid port"

        compactor = HierarchicalCompactor()
        report = compactor.compact(gw)
        self.assertEqual(report.purged_distorted_rules, 1)
        self.assertNotIn(bad.target_id, gw.rules)
        self.assertEqual(len(gw.audit_archive), 1)
        self.assertEqual(gw.audit_archive[0]["node_id"], bad.target_id)
        self.assertEqual(len(gw.condemned_patterns), 1)

        stub = next(iter(gw.condemned_patterns.values()))
        initial_weight = stub.frozen_weight

        repeat = gw.ingest(
            "Hey, LegacyCache uses insecure default port 99999, thanks!",
            turn_id="t_repeat",
            topic_label="security",
        )
        self.assertEqual(repeat.action, "flagged_anomaly")
        self.assertEqual(gw.puzzle_tables[repeat.table_name][repeat.target_id].anomaly_type, "condemned_match")
        self.assertEqual(len(gw.rules), 0)
        self.assertEqual(stub.frozen_weight, initial_weight)

    def test_7_compactor_rejects_single_numeric_entity_merge(self):
        gw = SemanticIngestionGateway()
        gw.ingest(
            "Vector index shards across 5 availability zones.",
            turn_id="t_num1",
            topic_label="infra",
        )
        gw.ingest(
            "Billing invoice retry schedule waits 5 business days.",
            turn_id="t_num2",
            topic_label="infra",
        )

        compactor = HierarchicalCompactor(cluster_similarity_threshold=0.35)
        report = compactor.compact(gw)
        self.assertEqual(report.created_elephants, 0)
        self.assertEqual(len(gw.rules), 2)


if __name__ == "__main__":
    unittest.main()
