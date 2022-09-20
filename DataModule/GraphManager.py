import networkx as nx
import pandas as pd
import numpy as np
import tensorflow as tf

from DataModule.LookupManager import LookupManager

class GraphManager():
    def __init__(
        self
        ) -> (None):
        
        self.graph = nx.Graph()
        
    def construct_graph(
        self, 
        history : pd.DataFrame,
        lookup : LookupManager
        ) -> (None):
        
        if history is None:
            return
        else:
            keys = history.keys()
            
            for key in keys:
                item_list = history[key]
                item_list_idx = lookup.str_to_idx(item_list).numpy()

                for idx in range(len(item_list_idx) -1):
                    edge_i = item_list_idx[idx]
                    edge_j = item_list_idx[idx+1]
                    
                    if self.graph.get_edge_data(edge_i, edge_j) is None:
                        weight = 1
                    else:
                        weight = self.graph.get_edge_data(edge_i, edge_j)['weight'] + 1
                    self.graph.add_edge(edge_i, edge_j, weight = weight)
        
    def get_adjacency_mat(
        self,
        num_items : int,
        ) -> (tf.Tensor):
        
        adjacency_matrix = np.zeros((num_items + 1, num_items + 1))
        for edge in self.graph.edges.data():
            edge_out = edge[0]
            edge_in = edge[1]
            adjacency_matrix[edge_in][edge_out] = 1
            
        np.fill_diagonal(adjacency_matrix, 1)
        adjacency_matrix[0][0] = 0
        
        adjacency_matrix = tf.convert_to_tensor(adjacency_matrix, dtype = tf.float32)
        
        return adjacency_matrix
        
    def get_edge_tensor(
        self,
        adjacency_mat : tf.Tensor
        ) -> (tf.Tensor):
        
        return tf.where(adjacency_mat != 0)

        
        
        
        
        
        