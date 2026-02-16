class GlobalKLLoss(AddedLossTerm):
    def __init__(self, kl_val):
        self.kl_val = kl_val
    def loss(self):
        return self.kl_val
    def register_loop_loss(self, current_loss):
        return current_loss + self.kl_val.sum()