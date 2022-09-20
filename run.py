
from cgitb import lookup
from DataModule.LookupManager import LookupManager
from DataModule.DataLoader import DataLoader
from DataModule.GraphManager import GraphManager

from TrainModule.TrainManager import TrainManager
from config import config

from Models.Feature.GAT_Model import GAT
from Models.GNNManager import GNNManager

import tensorflow as tf


if __name__ == "__main__":


    data_loader = DataLoader(config)
    item_count = data_loader.get_item_len()
    items = data_loader.get_items()
    history = data_loader.get_history()
    
    lookup_manager = LookupManager()
    lookup_manager.set_items(items)
    
    graph_manager = GraphManager()
    graph_manager.construct_graph(history, lookup_manager)
    adj_mat = graph_manager.get_adjacency_mat(item_count)
    edge_mat = graph_manager.get_edge_tensor(adj_mat)

    # Set
    model_config = {
        "b_dim" : config["batch_size"],
        "i_dim" : item_count,
        "n_dim" : config["sequence_length"],
        "d_dim": config["hidden_dim"]
    }

    model = GNNManager(
        feature_model = "GAT",
        session_model = "SRGNN",
        lookup = lookup_manager,
        edge = edge_mat,
        model_config = model_config
    )
        
    # Start Train
    trainmanger = TrainManager(
        model = model, 
        dataloader = data_loader, 
        lookup = lookup_manager,
        config = config
    )

    best_score = trainmanger.start()
    