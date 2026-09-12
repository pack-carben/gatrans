import unittest

import numpy as np
import torch

from models import TREE
from utils.data import NodeDataset, split_from_masks
from utils.preprocess import build_subgraphs, shortest_path_in_subgraph


class RegressionTests(unittest.TestCase):
    def test_repeated_nodes_have_identical_distances(self):
        actual = shortest_path_in_subgraph(
            np.array([0, 1, 0]), [np.array([1]), np.array([0])]
        )
        np.testing.assert_array_equal(actual, [[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    def test_isolated_repeated_node(self):
        actual = shortest_path_in_subgraph(np.array([0, 0, 0]), [np.array([], dtype=int)])
        np.testing.assert_array_equal(actual, np.zeros((3, 3)))

    def test_column_labels_forward_backward(self):
        labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.float32)[:, None]
        masks = [np.arange(6) // 2 == i for i in range(3)]
        train_id, y_train, *_ = split_from_masks(labels, *masks)
        self.assertEqual(y_train.shape, (2,))
        adjacent = np.ones((6, 6)) - np.eye(6)
        nodes, distances, degrees = build_subgraphs(adjacent, 2, 3)
        model = TREE(np.ones((6, 4)), degrees, nodes, distances, 2, 3,
                     d_model=8, n_layers=1, num_heads=2, dff=16, d_sp_enc=8)
        dataset = NodeDataset(train_id, y_train)
        logits = model(dataset.node_ids)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, dataset.labels)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(model.classifier.weight.grad).all())

    def test_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            split_from_masks(np.array([0, 1]), *[np.ones(2, dtype=bool)] * 3)

    def test_missing_validation_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing val"):
            split_from_masks(np.array([0, 1]), [True, True], np.array(None), [True, True])


if __name__ == "__main__":
    unittest.main()
