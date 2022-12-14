import tensorflow as tf
from keras import Model, Sequential
from keras.layers import Dense, LeakyReLU


class AttentionHead(Model):
    def __init__(
            self,
            i_dim: int,
            n_dim: int,
            d_dim: int):
        '''
        i_dim : item dimension
        n_dim : maximum sequence length
        d_dim : hidden state dimension
        '''
        super().__init__()

        self.i_dim = i_dim
        self.n_dim = n_dim
        self.d_dim = d_dim

        self.scale_layer = Dense(self.d_dim)

        self.attention_layer = Dense(1)
        self.leaky_relu = LeakyReLU(alpha=0.2)

    def call(self, emb_mat, edge):
        # Make edge matrix
        # (Edges, H) -> (Edges, 2, H) -> (Edges, 2, H') -> (Edges, 2H')
        edge_embedding = tf.gather(
            params=emb_mat,
            indices=edge
        )
        edge_embedding = self.scale_layer(edge_embedding)
        edge_embedding = tf.reshape(
            edge_embedding, [edge_embedding.shape[0], -1])

        # Calculate attention coefficient
        # (Edges, 2H') -> (Edges, )
        attention = self.attention_layer(edge_embedding)
        attention = self.leaky_relu(attention)

        # Calculate attention score
        attention = tf.squeeze(attention)
        attention_scores = tf.math.exp(
            attention - tf.ones(shape=(attention.shape)))

        attention_score_sum = tf.math.unsorted_segment_sum(
            attention_scores,
            segment_ids=edge[:, 0],
            num_segments=self.i_dim + 1
        )

        attention_score_sum = tf.repeat(attention_score_sum,
                                        tf.math.bincount(
                                            tf.cast(edge[:, 0],
                                                    dtype=tf.int32)))
        attention_score_norm = attention_scores / attention_score_sum

        # Get Item scaled embedding
        # (Items, H) -> (Items, H')
        item_scaled_embedding = self.scale_layer(emb_mat)
        item_embeddings = tf.gather(
            params=item_scaled_embedding,
            indices=edge[:, 1]
        )

        # Get aggregated features
        aggregated_features = tf.math.unsorted_segment_sum(
            data=attention_score_norm[:, tf.newaxis] * item_embeddings,
            segment_ids=edge[:, 0],
            num_segments=self.i_dim + 1
        )

        new_item_emb = tf.nn.sigmoid(aggregated_features)

        return new_item_emb


class GAT(Model):
    def __init__(
        self, 
        i_dim : int, 
        n_dim : int,
        d_dim : int):
        '''
        i_dim : item dimension
        n_dim : maximum sequence length
        d_dim : hidden state dimension
        '''
        super().__init__()
            
        self.i_dim = i_dim
        self.n_dim = n_dim
        self.d_dim = d_dim

        self.h_dim = 4
        # Item Embedding Matrix
        initializer = tf.keras.initializers.GlorotNormal()
        self.item_emb_mat = tf.Variable(
            initializer(shape=[i_dim + 1, d_dim], dtype=tf.float32),
            trainable=True
        )
        
        self.multi_head = [
            AttentionHead(i_dim = i_dim, n_dim = n_dim, d_dim = d_dim) for i in range(self.h_dim)
        ]
    
    def get_emb_mat(self):
        emb_mat = self.item_emb_mat
        return emb_mat
    
    def call(self, edge):
        emb_mat = self.get_emb_mat()
        
        emb_mat_list = list(map(lambda head : head(emb_mat, edge), self.multi_head))
        
        new_emb_mat = sum(emb_mat_list) / self.h_dim

        return new_emb_mat