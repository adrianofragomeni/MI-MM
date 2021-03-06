import torch as th
import shutil

class Model_utils:
    
    def __init__(self,filepath,best_metric=10000):
        self.best_metric=best_metric
        self.filepath=filepath
    
    def save_checkpoint(self,state,filename="checkpoint.pth.tar"):
        """ save checkpoint of the model
        
            Args:
            state: a dictionary with all the info which will be saved in the checkpoint
            filename: name of the checkpoint file
        """
        filepath=self.filepath / filename
        th.save(state, filepath)
        if state["best_score"]<self.best_metric:
            self.best_metric=state["best_score"]
            shutil.copyfile(filepath, self.filepath / "model_best.pth.tar")










