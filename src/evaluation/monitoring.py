from torch.utils.tensorboard import SummaryWriter
import os
import time


class Summary:
    """ Monitoring metrics """

    def __init__(self,filepath):
    
        self.writer = SummaryWriter(filepath / "logs/runs")
                                    
                                    
    def write_test(self,metrics,epoch):
        
        """ Monitoring metrics """
        self.writer.add_scalars("Loss",{"Test": metrics["testing_loss"]},epoch)
        self.writer.add_scalars("Loss_tv",{"Test": metrics["testing_loss_tv"]},epoch)
        self.writer.add_scalars("Loss_vt",{"Test": metrics["testing_loss_vt"]},epoch)

    def write_train(self,metrics,epoch):
        
        """ Monitoring metrics """
        self.writer.add_scalars("Loss",{"Train": metrics["training_loss"]},epoch)
        self.writer.add_scalars("Loss_tv",{"Train": metrics["training_loss_tv"]},epoch)
        self.writer.add_scalars("Loss_vt",{"Train": metrics["training_loss_vt"]},epoch)
        

def create_log_file(log_path, log_name=""):
    # Make log folder if necessary
    log_folder = os.path.dirname(log_path)
    os.makedirs(log_folder, exist_ok=True)

    # Initialize log files
    with open(log_path, "a") as log_file:
        now = time.strftime("%c")
        log_file.write("==== log {} at {} ====\n".format(log_name, now))


def log_metrics(epoch, metrics, log_path=None):
    """log_path overrides the destination path of the log"""
    
    now = time.strftime("%c")
    message = "(epoch: {}, time: {})".format(epoch, now)
    for k, v in metrics.items():
        message = message + ",{}:{}".format(k, v)

    # Write log message to correct file
    with open(log_path, "a") as log_file:
        log_file.write(message + "\n")
    return message


class Monitor:
    """" Monitor all the variables during training and evaluation"""
    
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.train_path = os.path.join(self.checkpoint, "train.txt")
        self.val_path = os.path.join(self.checkpoint, "val.txt")
        create_log_file(self.train_path)
        create_log_file(self.val_path)


    def log_train(self, epoch, metrics):
        log_metrics(epoch, metrics, self.train_path)

    def log_val(self, epoch, metrics):
        log_metrics(epoch, metrics, self.val_path)