import torch as th
import shutil

class Model_utils:
    
    def __init__(self,filepath,best_metric=0):
        self.Best_RSum=best_metric
        self.filepath=filepath
    
    def save_checkpoint(self,state,filename="checkpoint.pth.tar"):
        """ save checkpoint of the model
        
            Args:
            state: a dictionary with all the info which will be saved in the checkpoint
            filename: name of the checkpoint file
        """
        filepath=self.filepath / filename
        th.save(state, filepath)
        if state["best_score"]>self.Best_RSum:
            self.Best_RSum=state["best_score"]
            shutil.copyfile(filepath, self.filepath / "model_best.pth.tar")










