import tensorflow as tf


'''
Caculating Accuracy
'''
class ScoreManager():
    def __init__(self) -> None:
        pass
    
    def hit_rate(self, 
                 y_true : tf.Tensor, 
                 y_pred : tf.Tensor, 
                 metrics : list,
                 sequence_dims : bool = False):
        '''
        Recording hit numbers in metrics
        Args:
            - sequence_dims :
                - True :
                    y_true (Batch, Seq)
                    y_pred (Batch, Seq, Items)
                - False :
                    y_true (Batch,)
                    y_pred (Batch, Items)
        '''
        hit_number_dict = {}
        
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)
        
        if sequence_dims:
            y_pred = y_pred[:,-1,:]
            y_pred = tf.squeeze(y_pred)
            y_true = y_true[:,-1]
        y_true = tf.reshape(y_true, [-1, 1])
            
        y_pred_sort = tf.argsort(y_pred, axis=1, direction='DESCENDING')
        
        for k in metrics:
            top_k = int(k)
            y_pred_top = y_pred_sort[:, :top_k]
        
            y_exist = tf.equal(y_pred_top, y_true)
            y_exist = tf.cast(y_exist, tf.int32)
            hit_number = tf.math.count_nonzero(y_exist)
            hit_number_dict[k] = hit_number
        
        return hit_number_dict

        
        