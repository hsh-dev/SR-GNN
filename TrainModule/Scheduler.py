import math
import keras.backend as K

class CosineDecayWrapper:
    # constructor
    def __init__(self, optimizer, max_lr, min_lr, max_epochs, decay_cycles, decay_epochs=None):
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.decay_cycles = decay_cycles
        self.decay_epochs = decay_epochs
        
        self.max_lr = max_lr
        self.min_lr = min_lr
        
        self.mul_t = 0.8
        self.cycle = 0
        
        self.decay_T = int(self.decay_epochs / self.decay_cycles)

    # caculate learning rate for an epoch
    def cosine_annealing(self, epoch):
        cycle_idx = int(epoch / self.decay_T)
        
        if self.cycle < cycle_idx:
            self.cycle = cycle_idx
            self.max_lr = self.max_lr * self.mul_t
            
        cos_inner = (math.pi * (epoch % self.decay_T) / (self.decay_T - 1))
        return self.min_lr + (self.max_lr/2 - self.min_lr/2) * (math.cos(cos_inner)+1)

    # calculate and set learning rate at the start of the epoch
    def get_lr(self, epoch):
        if(epoch < self.decay_epochs):
            lr = self.cosine_annealing(epoch)
        else:
            lr = self.min_lr
        return lr

    def update_lr(self, epoch):
        new_lr = self.get_lr(epoch)
        K.set_value(self.optimizer.lr, new_lr)
        return new_lr