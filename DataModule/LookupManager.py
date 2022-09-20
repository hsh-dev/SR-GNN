import tensorflow as tf
import keras

class LookupManager():
    def __init__(self) -> (None):        
        
        self.encoding_layer = keras.layers.StringLookup()
        self.decoding_layer = keras.layers.StringLookup(invert=True)

    def set_items(self, items):
        self.encoding_layer.set_vocabulary(vocabulary = items)
        self.decoding_layer.set_vocabulary(vocabulary = items)
    
    def str_to_idx(self, x):
        return self.encoding_layer(x)

    def idx_to_str(self, x):
        decoded_x = self.decoding_layer(x)
        list_x = decoded_x.numpy().tolist()
        str_x = list(map(lambda x: x.decode('UTF-8'), list_x))

        return str_x
