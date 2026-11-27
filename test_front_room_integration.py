import os
import tempfile
import unittest

from semantic_ingestion_gateway import count_tokens
from src.layers.front_room_controller import FrontRoomController
from warehouse import LocalWarehouse


class TestFrontRoomControllerV21Integration(unittest.TestCase):
    def test_end_to_end_o1_window_gateway_and_telemetry(self):
        captured_contexts = []

        def mock_llm(ctx: str) -> str:
            captured_contexts.append(ctx)
            return f"ack:{len(ctx)}"

        controller = FrontRoomController(
            active_window_size=2,
            drift_threshold=0.65,
            max_retrieval_bundles=3,
            max_context_tokens=200,
            use_local_embeddings=True,
        )

        turns = [
            "Hey, basically ConnectionShield uses xylophone_mutex for fallback safety, thanks!",
            "Hello, ConnectionShield uses xylophone_mutex for fallback safety.",
            "What is the pricing for the enterprise plan?",
        ]
        for t in turns:
            resp = controller.process_turn(t, mock_llm)
            self.assertTrue(resp.startswith("ack:"))

        self.assertEqual(len(controller.active_window), 2)
        self.assertEqual(len(controller.gateway.rules), 2)

        for ctx in captured_contexts:
            self.assertLessEqual(count_tokens(ctx), 200)

        stats = controller.get_telemetry()
        self.assertEqual(stats["total_turns"], 3)
        self.assertGreater(stats["p50_ms"], 0.0)

        report = controller.compact_memory()
        self.assertGreaterEqual(report.remaining_active_nodes, 1)

    def test_local_warehouse_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                wh = LocalWarehouse(user_id="test_user")
                wh.store_bundle("b1", {"turn": "hello"}, ["anchor_a"])
                hits = wh.query_warehouse("anchor_a")
                self.assertEqual(len(hits), 1)
                self.assertEqual(hits[0]["turn"], "hello")
            finally:
                os.chdir(orig_cwd)


if __name__ == "__main__":
    unittest.main()
