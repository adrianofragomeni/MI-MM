import pandas as pd
import pickle5 as pickle
import os
from pathlib import Path

def submision_pickle(similarity_matrix,path_dataframes):
    
    log_folder = Path("../output")
    os.makedirs(log_folder, exist_ok=True)
    
    dictionary_info={"version": "0.1","challenge": "multi_instance_retrieval","sls_pt": 2,
                      "sls_tl": 3,"sls_td": 3}
    
    # considered unique narrations for evaluation of EPIC
    video_id=pd.read_csv(path_dataframes / "EPIC_100_retrieval_test.csv").values[:,0]
    text_id=pd.read_csv(path_dataframes / "EPIC_100_retrieval_test_sentence.csv").values[:,0]

    indexes=[]
    for elem in text_id:
        indexes.append(video_id.tolist().index(elem))

    similarity_matrix=similarity_matrix.T[:,indexes]
    
    dictionary_info["sim_mat"]=similarity_matrix    
    dictionary_info["txt_ids"]=text_id        
    dictionary_info["vis_ids"]=video_id    

    with open(Path(log_folder) / 'test.pkl', 'wb') as handle:
        pickle.dump(dictionary_info, handle, protocol=pickle.HIGHEST_PROTOCOL)