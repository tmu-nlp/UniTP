from torch import optim
from math import exp

def warm_adam(model, base_lr = 0.001, wander_threshold = 0.15):

    adam = optim.Adam(model.parameters(), betas = (0.9, 0.98), weight_decay = 0.01, eps = 1e-6)
        
    def schedule_lr(epoch, wander_ratio):
        if wander_ratio < wander_threshold:
            learning_rate = base_lr * (1 - exp(- epoch))
        else:
            linear_dec = (1 - (wander_ratio - wander_threshold) / (1 - wander_threshold + 1e-20))
            learning_rate = base_lr * linear_dec

        for opg in adam.param_groups:
            opg['lr'] = learning_rate
        # self._lr_discount_rate = 0.0001
        # for params in self._model.parameters():
        #     if len(params.shape) > 1:
        #         nn.init.xavier_uniform_(params)
        return learning_rate
            
    return adam, schedule_lr