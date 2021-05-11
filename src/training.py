#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch as th
from pathlib import Path
from utils.utils import parsing, get_default_device
from utils.model_utils import Model_utils
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from evaluation.monitoring import Summary,Monitor
from loader.loader_features import FeatureLoader, ToTensor
from models.embedding_projection import Embedding_Projection
from losses import  MAXMARGIN,optimizer_scheduler
import numpy as np
import random

# Random Initialization
np.random.seed(0)
random.seed(0)
th.manual_seed(0)

def train_epoch(training_loader,device, optimizer,scheduler,net, type_loss):
    
    running_loss = 0.0
    running_loss_tv=0.0
    running_loss_vt=0.0
    
    for i_batch, sample_batched in enumerate(training_loader):
    
        video_features=sample_batched["video_embed"].to(device)
        text_features=sample_batched["text_embed"]

        optimizer.zero_grad()
        text_embed,video_embed = net(text_features,video_features)
        similarity_matrix=th.matmul(text_embed, video_embed.t())

        loss_tv = type_loss(similarity_matrix)
        loss_vt = type_loss(similarity_matrix.t())
        loss=loss_vt+loss_tv
        
        running_loss_tv += loss_tv.data
        running_loss_vt += loss_vt.data
        
        running_loss += loss.data
        loss.backward()

        optimizer.step()
    
    scheduler.step()
        
    return {"training_loss" :running_loss / (i_batch + 1), "training_loss_tv" :running_loss_tv / (i_batch + 1), "training_loss_vt" :running_loss_vt / (i_batch + 1)}


def validation_epoch(testing_loader,device,net, type_loss):
    
    all_text_embed=[]
    all_video_embed=[]
    running_loss = 0.0
    running_loss_tv=0.0
    running_loss_vt=0.0

    with th.no_grad():
        for i_batch, sample_batched in enumerate(testing_loader):

            video_features=sample_batched["video_embed"].to(device)
            text_features=sample_batched["text_embed"]
 
            text_embed,video_embed=net(text_features,video_features)
            batch_similarity_matrix=th.matmul(text_embed, video_embed.t())

            loss_tv = type_loss(batch_similarity_matrix)
            loss_vt = type_loss(batch_similarity_matrix.t())
            loss=loss_vt+loss_tv
            
            running_loss_tv += loss_tv.data
            running_loss_vt += loss_vt.data

            all_video_embed.append(video_embed.cpu().numpy())
            all_text_embed.append(text_embed.cpu().numpy())
            running_loss += loss.data    

    return {"testing_loss" :running_loss / (i_batch + 1),"testing_loss_tv" :running_loss_tv / (i_batch + 1), "testing_loss_vt" :running_loss_vt / (i_batch + 1)}



def train(args_):
    
    summary= Summary(Path(args_.path_model))
    device= get_default_device()
    monitor = Monitor(Path("../data/models/logs"))
    model_utils=Model_utils(Path(args_.path_model))
    
    print("load model...")
    net= Embedding_Projection(args_.embed_dim, Path(args_.path_resources))
    net=net.pretrained_text_model(net)
    

    print("load features...")
    
    training_set=FeatureLoader(path_csv=Path(args_.path_csv) / "EPIC_100_retrieval_train.csv",path_root_embed=Path(args_.path_features)/"training_video_features.npy",
                               path_dataframes=Path(args_.path_csv),path_relevancy_mat=Path(args_.path_relevancy)/'caption_relevancy_EPIC_100_retrieval_train.pkl',
                               sentences_csv="caption_sentences.csv",transform=ToTensor(),relevancy=args_.rel)
    
    training_loader = DataLoader(training_set, batch_size=args_.batch_size,pin_memory=True,shuffle=True,num_workers=args_.cpu_count)

    testing_set=FeatureLoader(path_csv=Path(args_.path_csv) / "EPIC_100_retrieval_test.csv",path_root_embed=Path(args_.path_features)/"validation_video_features.npy",
                              path_dataframes=Path(args_.path_csv),transform=ToTensor())
    
    testing_loader = DataLoader(testing_set, batch_size=args_.batch_size,pin_memory=True,shuffle=False,num_workers=args_.cpu_count)    


    # Optimizers + Loss
    type_loss=MAXMARGIN.MaxMarginRankingLoss(margin=args_.margin)
    type_loss.to(device)
    optimizer = optim.Adam(net.parameters(), lr=args_.lr)
    scheduler = optimizer_scheduler.get_constant_schedule(optimizer)

    net.train()
    net.to(device)
    
    print('\nStarting training loop ...')

    for epoch in tqdm(range(args_.epochs)):
        
        training_metrics=train_epoch(training_loader,device, optimizer,scheduler,net, type_loss)

        monitor.log_train(epoch +1,training_metrics)
        summary.write_train(training_metrics,epoch)

        net.eval()
        validation_metrics=validation_epoch(testing_loader,device,net,type_loss)
        monitor.log_val(epoch +1,validation_metrics)
        
        model_utils.save_checkpoint({
                    "epoch": epoch+1,
                    "state_dict": net.state_dict(),
                    "best_score": validation_metrics["testing_loss"],
                    "optimizer": optimizer.state_dict(),
                    "scheduler":scheduler.state_dict()
                })
    
        
        summary.write_test(validation_metrics,epoch)
        net.train()
        
if __name__== "__main__":

    # create parser
    args_=parsing()
    train(args_)
