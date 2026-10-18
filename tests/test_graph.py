from __future__ import annotations

import unittest

import numpy as np


from vdroad.graph import expand_graph


class TwoPhaseExpansionTests(unittest.TestCase):
    def test_query_permutation_does_not_change_graph(self) -> None:
        nodes = np.asarray([[0.0, 0.0], [2.0, 0.0]])
        edges = np.asarray([[0, 1]])
        scores = np.asarray([[0.9, 0.8], [0.9, 0.8]])
        def first(_ids, _xy, direction, _round):
            return scores, np.stack([direction * 3.0, direction * 5.0], axis=1)
        def permuted(_ids, _xy, direction, _round):
            return scores[:, ::-1], np.stack([direction * 5.0, direction * 3.0], axis=1)
        a = expand_graph(nodes, edges, first, rounds=1, score_threshold=0.5, merge_distance=0.25)
        b = expand_graph(nodes, edges, permuted, rounds=1, score_threshold=0.5, merge_distance=0.25)
        self.assertTrue(np.array_equal(a.nodes, b.nodes))
        self.assertTrue(np.array_equal(a.edges, b.edges))

    def test_snap_excludes_source_and_existing_neighbor(self) -> None:
        nodes = np.asarray([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
        edges = np.asarray([[0, 1]])
        def predictor(ids, xy, direction, _round):
            del ids, xy, direction
            return np.asarray([[1.0], [1.0]]), np.asarray([[[4.0, 0.0]], [[2.0, 0.0]]])
        result = expand_graph(
            nodes, edges, predictor, rounds=1, score_threshold=0.5,
            merge_distance=0.5, mode="learned_snap",
        )
        self.assertIn((1, 2), {tuple(edge) for edge in result.edges.tolist()})

    def test_round_start_prevents_merging_into_same_round_insert(self) -> None:
        nodes = np.asarray([[0.0, 0.0], [2.0, 0.0]])
        edges = np.asarray([[0, 1]])
        def predictor(_ids, _xy, direction, _round):
            return np.ones((2, 1)), direction[:, None, :] * 3.0
        result = expand_graph(nodes, edges, predictor, rounds=1, score_threshold=0.5, merge_distance=0.25)
        self.assertEqual(len(result.nodes), 4)
        self.assertEqual(len(result.edges), 3)


if __name__ == "__main__":
    unittest.main()
