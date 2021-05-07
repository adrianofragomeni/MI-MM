#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch as th
import torch.nn as nn
import models.layers as layers
from models.text_model import Sentence_Embedding
    
class Embedding_Projection(nn.Module):
    def __init__(self,embd_dim, path_resources, gating=True, normalize=True):
        super(Embedding_Projection, self).__init__()
        
        self.path_resources=path_resources
        
        self.text_module=Sentence_Embedding(path_resources)
        
        self.embd_text = layers.Gated_Embedding_Unit(embd_dim*2, embd_dim,gating=gating, normalize=normalize) 
        self.embd_video = layers.Gated_Embedding_Unit(embd_dim, embd_dim, gating=gating,normalize=normalize)  
    
        
    def forward(self, narration, video_emb):
        
        text_emb=self.text_module(narration,"cuda")
        text = self.embd_text(text_emb["text_features"])         
        video = self.embd_video(video_emb)
        return text,video
    
    
    def pretrained_text_model(self,model):
        
        pretrained_dict = th.load(self.path_resources / "s3d_howto100m.pth")
        
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        model.load_state_dict(model_dict)    
        
        model.text_module.update_embeddings()    
        
        return model




