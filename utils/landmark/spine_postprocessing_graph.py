
import numpy as np
import networkx as nx
from utils.landmark.common import Landmark
import pickle


class SpinePostprocessingGraph(object):
    """
    Extract landmark sequence for spine images.
    """
    def __init__(self,
                 num_landmarks,
                 possible_successors,
                 offsets_mean,
                 distances_mean,
                 distances_std,
                 bias=2.0,
                 l=0.2):
        """
        Initializer.
        :param num_landmarks: The number
        :param bias: The bias value.
        :param l: The lambda weighting factor between unary and pairwise terms.
        """
        self.num_landmarks = num_landmarks
        self.possible_successors = possible_successors
        self.offsets_mean = offsets_mean
        self.distances_mean = distances_mean
        self.distances_std = distances_std
        self.bias = bias
        self.l = l

    def distance_value(self, landmark_from, landmark_to, landmark_from_index, landmark_to_index):
        """
        Calculate the pairwise distance value
        :param landmark_from: The landmark from.
        :param landmark_to: The landmark to.
        :param landmark_from_index: The landmark from index.
        :param landmark_to_index: The landmark to index.
        :return: The pairwise distance value.
        """
        mean_dir = self.offsets_mean[landmark_from_index][landmark_to_index]
        mean_dist = self.distances_mean[landmark_from_index][landmark_to_index]
        offset = landmark_from.coords - landmark_to.coords
        diff_single = (mean_dir * mean_dist - offset) / mean_dist
        diff_single[2] = diff_single[2] * 3 if diff_single[2] > 0 else diff_single[2]
        diff = diff_single * 2
        dist = np.sum(np.square(diff))
        return 1 - dist

    def unary_term(self, landmark):
        """
        Unary terms based on heatmap value and bias.
        :param landmark: Landmark.
        :return: Unary term.
        """
        return self.l * landmark.value + self.bias

    def pairwise_term(self, landmark_from, landmark_to, landmark_from_index, landmark_to_index):
        """
        Pairwise terms based on distance and offset.
        :param landmark_from: Landmark from.
        :param landmark_to: Landmark to.
        :param landmark_from_index: Landmark from index.
        :param landmark_to_index: Landmark to index.
        :return: Pairwise term.
        """
        distance_value = self.distance_value(landmark_from, landmark_to, landmark_from_index, landmark_to_index)
        return (1 - self.l) * distance_value

    def create_graph(self, local_heatmap_maxima):
        """
        Create a graph for the given local_heatmap_maxima.
        :param local_heatmap_maxima: The list of heatmap maxima lists.
        """
        # directed graph
        G = nx.DiGraph()
        # form all currents to all nexts.
        for curr in range(self.num_landmarks):
            nexts = self.possible_successors[curr]
            for next in nexts:
                # form all currents local maxima (i) to all nexts local maxima (k).
                for i in range(len(local_heatmap_maxima[curr])):
                    for j in range(len(local_heatmap_maxima[next])):
                        curr_landmark = local_heatmap_maxima[curr][i]
                        next_landmark = local_heatmap_maxima[next][j]
                        if not curr_landmark.is_valid or not next_landmark.is_valid:
                            continue
                        # weight contains pairwise distance value, and unary current landmark value
                        weight = - (self.unary_term(curr_landmark) + self.pairwise_term(curr_landmark, next_landmark, curr, next))
                        G.add_edge(f'{curr}_{i}', f'{next}_{j}', weight=weight)

        # add virtual start and end node for each local maxima.
        for curr in range(self.num_landmarks):
            for i in range(len(local_heatmap_maxima[curr])):
                curr_landmark = local_heatmap_maxima[curr][i]
                if not curr_landmark.is_valid:
                    continue
                # use unary landmark value only for end edge, as otherwise the weight would be counted twice.
                weight = - self.unary_term(curr_landmark)
                G.add_edge('s', f'{curr}_{i}', weight=0)
                G.add_edge(f'{curr}_{i}', 't', weight=weight)
        return G

    def vertex_name_to_indizes(self, name):
        """
        Convert vertex names to indizes.
        :param name: The vertex name.
        :return: Tuple of landmark_index, maxima_index.
        """
        landmark_index = int(name[:name.find('_')])
        maxima_index = int(name[name.find('_') + 1:])
        return landmark_index, maxima_index

    def path_to_landmarks(self, path, local_heatmap_maxima):
        """
        Converts a path to a list of landmarks. The length of the list is the same as the number of landmarks.
        If a landmark is not valid, a Landmark with np.nan coordinates and is_valid = False is inserted at its position.
        :param path: The path.
        :param local_heatmap_maxima: The local heatmap maxima.
        :return: List of landmarks.
        """
        landmarks = [Landmark(coords=[np.nan] * 3, is_valid=False) for _ in range(self.num_landmarks)]
        for node in path:
            if node == 's' or node == 't':
                continue
            landmark_index, maxima_index = self.vertex_name_to_indizes(node)
            landmarks[landmark_index] = local_heatmap_maxima[landmark_index][maxima_index]
        return landmarks

    def solve_local_heatmap_maxima(self, local_heatmap_maxima):
        """
        Calculate and return the best path from the local heatmap maxima.
        :param local_heatmap_maxima: List of lists of landmarks.
        :return: List of landmarks representing the detected landmark sequence.
        """
        G = self.create_graph(local_heatmap_maxima)
        shortest_path = nx.shortest_path(G, 's', 't', 'weight', method='bellman-ford')
        distances = []
        for i in range(1, len(shortest_path) - 2):
            edge = G.edges[shortest_path[i], shortest_path[i+1]]
            weight = edge['weight']
            distances.append((f'{shortest_path[i]}_{shortest_path[i+1]}', f'{weight:0.4f}'))
        return self.path_to_landmarks(shortest_path, local_heatmap_maxima)
