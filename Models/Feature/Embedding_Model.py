import tensorflow as tf
from tensorflow.keras import Model


class EmbeddingModel(Model):
    def __init__(self, i_dim, d_dim, n_dim):
        super().__init__()

        self.i_dim = i_dim  # total item number

        self.d_dim = d_dim  # latent vector dimension
        self.n_dim = n_dim  # maximum sequence length

        initializer = tf.keras.initializers.GlorotNormal()

        self.item_emb_mat = tf.Variable(
            initializer(shape=[self.i_dim + 1, self.d_dim], dtype=tf.float32),
            trainable=True)

        self.pos_emb_mat = tf.Variable(
            initializer(shape=[self.n_dim, self.d_dim], dtype=tf.float32),
            trainable=True)

    def call(self, x):
        # Input : B x N

        output = tf.nn.embedding_lookup(self.item_emb_mat, x)

        return output
