import json
import unittest

from zkml_quality.inference import ROOT, infer


class InferenceTest(unittest.TestCase):
    def test_bundled_examples_and_scores(self):
        for name, expected in (("normal", [0.75, 0.25]), ("inspect", [-1.8125, 2.8125])):
            result = infer(json.loads((ROOT / "samples" / f"{name}.json").read_text()))
            self.assertEqual(result["label"], name)
            for actual, score in zip(result["scores"], expected):
                self.assertAlmostEqual(actual, score)
            self.assertEqual(len(result["model_sha256"]), 64)

    def test_invalid_features(self):
        for features in ([], [0] * 5, [0] * 7, [True] * 6, [float("nan")] * 6, [1.1] * 6, [-0.1] * 6):
            with self.assertRaises(ValueError):
                infer({"features": features})


if __name__ == "__main__":
    unittest.main()
