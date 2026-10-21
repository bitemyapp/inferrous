import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("make_batch_matrix", Path(__file__).parents[1] / "make_batch_matrix.py")
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)


class BatchMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = dict(max_context=32768, prefill_chunk=512, kv_cache_bytes=64 << 30)
        cls.workloads, cls.scenarios = matrix.build_matrix(cls.config)

    def test_complete_cartesian_matrix_and_ragged_boundaries(self):
        self.assertEqual(len(self.workloads), 99)
        self.assertEqual(list(self.workloads), sorted(self.workloads))
        homogeneous = [s for s in self.scenarios if s["kind"] == "homogeneous"]
        self.assertEqual(len(homogeneous), 96)
        expected = {(n if n != "near32k" else 32768 - out, out, b)
                    for n in [5, 129, 512, 2048, 4096, 8192, 16384, "near32k"]
                    for out in [2, 32] for b in [1, 2, 4, 8, 16, 32]}
        self.assertEqual({(s["prompt_lengths"][0], s["output_lengths"][0], s["batch"])
                          for s in homogeneous}, expected)
        for scenario in self.scenarios:
            for n, out in zip(scenario["prompt_lengths"], scenario["output_lengths"]):
                self.assertLessEqual(n + out, 32768)
        self.assertIn("ragged-native-sustained", self.workloads)
        self.assertEqual([len(r["prompt_ids"]) for r in self.workloads["page-growth-boundaries"]],
                         [63, 64, 65, 127, 128, 129])

    def test_near_native_capacity_is_retained_as_three_waves(self):
        scenario = next(s for s in self.scenarios if s["name"] == "homogeneous-p32766-o02-b32")
        self.assertFalse(scenario["fits_simultaneously"])
        self.assertEqual([len(w) for w in scenario["cold_admission_waves"]], [14, 14, 4])
        self.assertEqual(scenario["reserved_pages_per_request"], [512] * 32)

    def test_reservations_round_pages_and_reject_individual_overflow(self):
        page_bytes = 64 * 147456
        requests = [matrix.request(65, 1), matrix.request(65, 1, 1)]
        result = matrix.reservations(requests, 3 * page_bytes, 32)
        self.assertEqual(result["cold_admission_waves"], [[0], [1]])
        self.assertEqual(result["reserved_pages_per_request"], [2, 2])
        with self.assertRaisesRegex(ValueError, "cannot be admitted"):
            matrix.reservations(requests, page_bytes, 32)

    def test_generator_never_silently_shrinks_native_scope(self):
        with self.assertRaisesRegex(ValueError, "native 32768"):
            matrix.build_matrix(dict(self.config, max_context=4096))
        self.assertEqual(matrix.request(3, 2, 3)["prompt_ids"], [45, 62, 79])


if __name__ == "__main__":
    unittest.main()
