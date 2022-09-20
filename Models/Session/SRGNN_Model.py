from threading import local
import tensorflow as tf
from keras import Model, Sequential
from DataModule.LookupManager import LookupManager
from keras.layers import Dense

class SRGNN(Model):
    def __init__(
        self,
        b_size : int,        
        d_dim: int,
        lookup: LookupManager):
        
        super().__init__()
        
        self.lookup = lookup
        
        self.global_layer = Dense(d_dim)
        self.local_layer = Dense(d_dim)
        
        initializer = tf.keras.initializers.GlorotNormal()
        self.bias = tf.Variable(
            initializer(shape=[b_size, 1, 1]),
            dtype=tf.float32,
            trainable=True)
        
        self.q = Dense(1)
        
        self.hybrid_layer = Dense(d_dim)
        
        
    def call(self, x_input, item_emb):
        '''
        Args:
            x_input : tensor with user sequence index (Batch, Seq)
            item_emb : tensor with item embedding matrix (Items, Hidden dims)
            
            output : tensor with item recommendations (Batch, Items)
        '''        
        ''' Get global feature and local feature '''
        seq_feature = tf.gather(params=item_emb, indices=x_input)
        
        global_feature = self.global_layer(seq_feature)
        
        local_feature = self.local_layer(seq_feature[:, -1, :])
        local_feature_tile = tf.tile(local_feature[:, tf.newaxis, :],
                                tf.constant([1, global_feature.shape[1], 1], tf.int32))
                
        ''' Calculate soft attention '''
        sum_feature = global_feature + local_feature_tile + self.bias
        sum_feature = tf.nn.sigmoid(sum_feature)
        attention_scores = self.q(sum_feature)
                
        ''' Calculate Global session preference
        (Batch, Seq, Hid) * (Batch, Seq, 1) -> (Batch, Hid) '''
        session_global = seq_feature * attention_scores
        session_global = tf.math.reduce_sum(session_global, axis = 1)
                
        ''' Calculate Hybrid session preference
        (Batch, Hid) + (Batch, Hid) -> (Batch, Hid) '''
        session_hybrid = tf.concat([local_feature, session_global], axis = 1)
        session_hybrid = self.hybrid_layer(session_hybrid)
        
        ''' Recommend Items '''
        output = tf.matmul(session_hybrid, item_emb, transpose_b = True)
        # output = tf.nn.softmax(rec_items, axis = 1)
                
        return output