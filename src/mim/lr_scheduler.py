import math
from torch import optim

def cosine_scheduler(optimizer, training_steps, warmup_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = current_step - warmup_steps
        progress /= max(1, training_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def build_scheduler(args, optimizer):
    num_steps = int(args.epochs * args.n_iter_per_epoch)
    warmup_steps = int(args.warmup_epochs * args.n_iter_per_epoch)
    
    lr_scheduler = cosine_scheduler(optimizer, num_steps, warmup_steps)
    
    return lr_scheduler