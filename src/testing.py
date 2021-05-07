import torch as th
from pathlib import Path
from utils.utils import get_default_device, parsing
from torch.utils.data import DataLoader
from loader.loader_features import FeatureLoader, ToTensor
from evaluation.create_submission import submision_pickle
from models.embedding_projection import Embedding_Projection
import numpy as np

def eval_model(args_, name_model):
    
    device= get_default_device()
    print("load features...")
    
    dataset=FeatureLoader(path_csv=Path(args_.path_csv) / "EPIC_100_retrieval_test.csv",path_root_embed=Path(args_.path_features)/"validation_video_features.npy",
                              path_dataframes=Path(args_.path_csv),transform=ToTensor())
    
    loader = DataLoader(dataset, batch_size=args_.batch_size,pin_memory=True,shuffle=False,num_workers=args_.cpu_count)    


    print("load model...")
    net=Embedding_Projection(args_.embed_dim,  Path(args_.path_resources))
    net=net.pretrained_text_model(net)
    net.eval()
    
    # load best model
    best_model = th.load(Path(args_.path_model)/name_model)
    best_epoch=best_model["epoch"]
    print("Best Epoch: {}".format(best_epoch))
    net.load_state_dict(best_model['state_dict'])
    
    net.to(device)
    
    all_text_embed=[]
    all_video_embed=[]
    
    with th.no_grad():
        for i_batch, sample_batched in enumerate(loader):
            
            video_features=sample_batched["video_embed"].to(device)

            text_features=sample_batched["text_embed"]

            text_embed,video_embed=net(text_features,video_features)
            
            all_video_embed.append(video_embed.cpu().numpy())
            all_text_embed.append(text_embed.cpu().numpy())
        
        all_video_embed=np.vstack(all_video_embed)
        all_text_embed=np.vstack(all_text_embed)

        similarity_matrix= np.matmul(all_text_embed,all_video_embed.T)     

    submision_pickle(similarity_matrix,Path(args_.path_csv))

    
    
if __name__== "__main__":


    name_model="model_best.pth.tar"
    # create parser
    args_=parsing()
    eval_model(args_, name_model)
    
