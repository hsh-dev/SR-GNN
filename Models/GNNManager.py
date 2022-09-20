from ast import Str
from typing import Dict
import tensorflow as tf
from keras import Model, Sequential
from keras.layers import Dense, LeakyReLU
from DataModule.LookupManager import LookupManager

from Models.Feature.GAT_Model import GAT    
from Models.Session.SRGNN_Model import SRGNN

class GNNManager(Model):
    def __init__(
        self,
        feature_model : str,
        session_model : str,
        lookup : LookupManager,
        edge : tf.Tensor,
        model_config : dict
        ):
        super().__init__()
    
        self.edge = edge

    
        if feature_model == "GAT":
            self.f_model = GAT(
                i_dim = model_config['i_dim'],
                n_dim = model_config['n_dim'],
                d_dim = model_config['d_dim'],
                lookup = lookup
            )
        
        if session_model == "SRGNN":
            self.s_model = SRGNN(
                b_size = model_config['b_dim'],
                d_dim = model_config['d_dim'],
                lookup = lookup
            )
    
    def get_item_emb(self):
        return self.f_model.get_emb_mat()
    
    def call(self, x_input):
        '''
        Args:
            - x_input : (Batch, Seq)
            - output : (Batch, Items)
        '''
        
        # Feature model
        item_emb = self.f_model(self.edge)
        
        # Session Model
        output = self.s_model(x_input, item_emb)
                
        return output
        
    
