import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from iros.authority_features import recording_features
from tools.annotation.merge_alpha_annotations import aligned_scores


class PaperPreprocessingTests(unittest.TestCase):
    def test_split_history_and_distance(self):
        force = np.repeat(np.arange(8, dtype=np.float32)[:, None], 2, axis=1)
        distance = np.arange(8, dtype=np.float32)
        iou = np.ones(8, dtype=np.float32)
        features, safety = recording_features(force, iou, distance, (3, 6))
        np.testing.assert_array_equal(features[0, :, 1:], 0)
        np.testing.assert_array_equal(safety[:, 1], [3, 4, 5])
        force[:3] = 1000
        distance[6:] = 1000
        after = recording_features(force, iou, distance, (3, 6))
        np.testing.assert_array_equal(features, after[0])
        np.testing.assert_array_equal(safety, after[1])
        old_features, old_safety = recording_features(force, iou, distance, (3, 6), legacy=True)
        self.assertEqual(old_features[0, 0, 1], -997)
        np.testing.assert_allclose(old_safety[:, 1], np.array([3, 4, 5]) / 1000)

    def test_annotator_frame_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b, c = (Path(directory) / f"{name}.csv" for name in "abc")
            header = "frame,alpha_l,alpha_r\n"
            a.write_text(header + "1,0.1,0.2\n2,0.2,0.3\n")
            b.write_text(header + "2,0.4,0.5\n1,0.3,0.4\n")
            c.write_text(header + "1,0.5,0.6\n2,0.6,0.7\n")
            np.testing.assert_allclose(aligned_scores([a, b, c]).mean(axis=0),
                                       [[0.3, 0.4], [0.4, 0.5]])
            c.write_text(header + "1,0.5,0.6\n3,0.6,0.7\n")
            with self.assertRaisesRegex(ValueError, "IDs differ"):
                aligned_scores([a, b, c])
            c.write_text(header + "1,0.5,0.6\n1,0.6,0.7\n")
            with self.assertRaisesRegex(ValueError, "duplicate frame IDs"):
                aligned_scores([a, b, c])
            with self.assertRaisesRegex(ValueError, "distinct file"):
                aligned_scores([a, b, a])
            positional = Path(directory) / "labels.pkl"
            positional.write_bytes(pickle.dumps({"alpha_L": [0.1], "alpha_R": [0.2]}))
            with self.assertRaisesRegex(ValueError, "counts differ"):
                aligned_scores([a, b, positional])
