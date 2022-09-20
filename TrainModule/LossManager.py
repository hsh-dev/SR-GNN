import tensorflow as tf
import keras.backend as K


class LossManager():
    '''
    Loss Functions
    '''
    def __init__(self) -> None:
        pass

    def cross_entropy_loss(self, y_true, y_pred):
        cce = tf.keras.losses.CategoricalCrossentropy()
        err = cce(y_true, y_pred)

        return err

    def negative_log_with_mask(self, y_true, y_pred, mask):
        '''
        y_true : B x N
        y_pred : B x N x I
        -> true logits : B x N
        -> masking : B x N
        '''
        logits = tf.gather(y_pred,
                           indices = y_true,
                           axis = 2,
                           batch_dims = 2)
        logits = -K.log(logits + 1e-10)
        
        mask_count = tf.cast(tf.math.count_nonzero(mask), tf.float32)
        mask = tf.cast(mask, tf.float32)

        masked_logits = logits * mask
        logit_sum = K.sum(masked_logits)
        loss = logit_sum / mask_count

        return loss
    
    def bpr_loss_with_ns(self, y_true_idx, negative_idx, y_pred):  
        '''
        BPR Loss with Negative Sampling
        '''      
        # Negative Sample
        negative_idx = tf.expand_dims(negative_idx, axis = 1)
        seq_dim = y_pred.shape[1]
        multiples = tf.constant([1, seq_dim, 1], dtype = tf.int32)
        negative_idx_repeat = tf.tile(negative_idx, multiples)
        negative_list = tf.gather(y_pred, indices = negative_idx_repeat, axis = 2, batch_dims=2)

        # Positive Sample
        positive_list = tf.gather(y_pred, 
                                  indices=tf.expand_dims(y_true_idx, axis=2), 
                                  axis=2, batch_dims=2)

        # Calculate
        substract = positive_list - negative_list
        sig = K.sigmoid(substract)
        log = K.log(sig)
        sum = K.sum(log, axis=2)
        loss = sum / (-100)
        
        # Loss Mask
        mask = tf.greater(y_true_idx, 0)
        non_zero_loss = tf.boolean_mask(loss, mask)
        
        loss = K.mean(non_zero_loss)
    
        return loss
    
    def bpr_loss(self, 
                 y_true_idx : tf.Tensor,
                 y_pred : tf.Tensor, 
                 sequence_dims : bool = False,
                 mask : bool = False,):
        '''
        BPR Loss without Negative Sampling
        Args :
            - sequence_dims : 
                - True : 
                    y_true_idx (Batch, Seq)
                    y_pred (Batch, Seq, Items)
                - False :
                    y_true_idx (Batch, )
                    y_pred (Batch, Items)
            - mask :
                True if you use masking label.
                In order to neglect mask in calculation.
        '''

        new_dims = 1
        if sequence_dims:
            new_dims = 2
        
        positive_list = tf.gather(
            y_pred, 
            indices = tf.expand_dims(y_true_idx, axis = new_dims), 
            axis = new_dims, 
            batch_dims = new_dims
        )
        
        substract = positive_list - y_pred
        item_len = substract.shape[new_dims]
        
        sig = K.sigmoid(substract)        
        log = K.log(sig)
        sum = K.sum(log, axis = new_dims)
        loss = sum / (-item_len)
            
        if mask:
            # Loss Mask
            mask = tf.greater(y_true_idx, 0)
            non_zero_loss = tf.boolean_mask(loss, mask)
            loss = K.mean(non_zero_loss)

        return loss
            
    
    def top_1_ranking_loss(self, y_true_idx, y_pred):
        negative_list = tf.gather(y_pred, indices=y_true_idx, axis=1)

        y_true_idx = tf.expand_dims(y_true_idx, axis=1)
        positive_list = tf.gather(
            y_pred, indices=y_true_idx, axis=1, batch_dims=1)

        cal = K.sigmoid(negative_list - positive_list)

        loss = K.mean(cal)

        return loss
