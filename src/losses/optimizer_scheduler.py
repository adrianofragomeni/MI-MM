from torch.optim.lr_scheduler import LambdaLR

def get_constant_schedule(optimizer, last_epoch = -1):
    """
    Create a schedule with a constant learning rate, using the learning rate set in optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    return LambdaLR(optimizer, lambda _: 1, last_epoch=last_epoch)