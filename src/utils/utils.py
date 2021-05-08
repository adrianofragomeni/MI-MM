import argparse 
import torch as th


def parsing():
    
    parser=argparse.ArgumentParser()
    
    parser.add_argument("--epochs", type=int, help="Number of epochs",
                        default=400)
    
    parser.add_argument("--batch-size", type=int, help="Batch size",
                        default=512)
    
    parser.add_argument("--cpu-count", type=int, help="Num workers",
                        default=0)
    
    parser.add_argument("--embed-dim", type=int, help="Embedding size",
                        default=1024)
    
    parser.add_argument("--margin", type=float, help="Margin costant of the Multi-instance MaxMargin loss",
                        default=0.2)
    
    parser.add_argument("--rel", type=float, help="Relevancy value to create the positive set",
                        default=0.5)
    
    parser.add_argument('--lr', type=float, help='Learning rate',
                        default=1e-4)

    parser.add_argument("--path-features", type=str, help="Path of the features",
                        default="../data/features")
    
    parser.add_argument("--path-csv",type=str, help="Path of the csv of the dataset ",
                        default="../data/dataframes")

    parser.add_argument("--path-relevancy",type=str, help="Path of the training relevancy matrix ",
                        default="../data/relevancy")

    parser.add_argument("--path-model", type=str, help="Path of the saved model",
                        default="../data/models")
    
    parser.add_argument("--path-resources", type=str, help="Path of resources to train the model",
                    default="../data/resources")

    parser.add_argument("--best-model", type=str, help="Name of the trained model",
                    default="model_best.pth.tar")
    
    args_= parser.parse_args()
    
    return args_



def get_default_device():
    """ Pick GPU or CPU as device"""
    
    if th.cuda.is_available():
        device=th.device("cuda")
    else:
        device=th.device("cpu")
    return device


