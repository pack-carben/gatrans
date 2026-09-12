import unittest

import numpy as np
import torch

from models import TREE
from utils.data import NodeDataset, split_from_masks
from utils.preprocess import build_subgraphs, shortest_path_in_subgraph
from utils.cuda_preprocess import build_subgraphs_cuda, sampled_distances


class RegressionTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_sampling_and_distances(self):
        rng = np.random.default_rng(7)
        adjacency = rng.random((20, 20)) < 0.2
        adjacency[0] = False
        neighbors = [np.flatnonzero(row) for row in adjacency]
        nodes, distances, degrees = build_subgraphs_cuda(adjacency, 3, 8, batch_size=7)
        np.testing.assert_array_equal(degrees, adjacency.sum(axis=1))
        for node in range(20):
            for channel in range(3):
                walk = nodes[node, channel]
                self.assertEqual(walk[0], node)
                for source, target in zip(walk[:-1], walk[1:]):
                    self.assertTrue(adjacency[source, target] or (degrees[source] == 0 and source == target))
                np.testing.assert_array_equal(distances[node, channel], shortest_path_in_subgraph(walk, neighbors))
        repeated, _, _ = build_subgraphs_cuda(adjacency, 3, 8, batch_size=7)
        np.testing.assert_array_equal(nodes, repeated)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_empty_graph(self):
        nodes, distances, _ = build_subgraphs_cuda(np.zeros((3, 3)), 2, 4, batch_size=2)
        np.testing.assert_array_equal(nodes, np.broadcast_to(np.arange(3)[:, None, None], (3, 2, 4)))
        np.testing.assert_array_equal(distances, np.zeros((3, 2, 4, 4)))

    def test_batched_distances_with_disconnected_repeated_nodes(self):
        adjacency = torch.tensor([[False, True, False], [False, False, False], [False, False, False]])
        walks = torch.tensor([[0, 1, 0, 2]])
        np.testing.assert_array_equal(sampled_distances(adjacency, walks).numpy()[0],
                                      [[0, 1, 0, -1], [-1, 0, -1, -1], [0, 1, 0, -1], [-1, -1, -1, 0]])

    def test_distances_match_scipy_on_sampled_directed_graphs(self):
        from scipy.sparse.csgraph import shortest_path

        rng = np.random.default_rng(42)
        for _ in range(20):
            adjacency = rng.random((12, 12)) < 0.25
            np.fill_diagonal(adjacency, False)
            neighbors = [np.flatnonzero(row) for row in adjacency]
            sampled = rng.integers(0, 12, size=8)
            unique, inverse = np.unique(sampled, return_inverse=True)
            expected = shortest_path(adjacency[np.ix_(unique, unique)].astype(float),
                                     directed=True, unweighted=True)
            expected = expected[np.ix_(inverse, inverse)]
            expected[~np.isfinite(expected)] = -1
            np.testing.assert_array_equal(shortest_path_in_subgraph(sampled, neighbors), expected)

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
