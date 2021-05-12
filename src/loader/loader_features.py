from torch.utils.data import Dataset
import torch as th
import numpy as np
import pandas as pd  
import pickle5 as pickle
import random

class FeatureLoader(Dataset):
    """ EPIC loader"""
    
    def __init__(self,path_csv,path_root_embed,path_dataframes,path_relevancy_mat=None,sentences_csv=None,transform=None,relevancy=None):
        
        self.relevancy=relevancy
        self.path_relevancy_mat=path_relevancy_mat
        
        if self.path_relevancy_mat:
            with open(path_relevancy_mat, 'rb') as fp:
                self.relevancy_mat = pickle.load(fp)
        
            self.captions_sentences=pd.read_csv(path_dataframes/sentences_csv).values[:,1]
            self.dict_narration ={v:k for k,v in enumerate(self.captions_sentences)}    
        
        self.video_feat= np.load(path_root_embed)
        
        self.narrations=pd.read_csv(path_csv).values[:,8]

        self.transform=transform
    
            
    def __len__(self):
        return self.video_feat.shape[0]
    
    def __getitem__(self,idx):
        
        if th.is_tensor(idx):
            idx=idx.tolist()
            
        if self.path_relevancy_mat:
            positive_list=np.where(self.relevancy_mat[idx]>=self.relevancy)[0].tolist()                    
            pos=random.sample(positive_list,min(len(positive_list),1))
            sample= {"video_embed": self.video_feat[idx,:],"text_embed":self.captions_sentences[pos][0]}
        else:
            sample= {"video_embed": self.video_feat[idx,:], "text_embed": self.narrations[idx]}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):    
        sample["video_embed"]=th.from_numpy(sample["video_embed"]).float()
        
        return sample
