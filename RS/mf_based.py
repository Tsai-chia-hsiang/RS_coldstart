import os
import sys
from tqdm import tqdm
import numpy as np
import torch
sys.path.append(os.path.dirname(__file__))
from utils.dictutils import *
from utils.mf import WeightedALS_MF, ALS_MF
from utils.plotutils import plotLoss
from utils.dataset import build_cross_domain_matrix, Dataset
from utils.evaluation import Evaluate

def mf(
    matrix:torch.Tensor,
    model_save_path:os.PathLike,d:torch.device=torch.device('cpu'),
    factorizer="WeightedALS_MF", mf_args:dict={'l2_reg':0.1, 'latency':40},
    vis_showinline:bool=False, return_result:list=['prediction']
)->torch.Tensor:

    def construct_matrixfactorizer(R, factorizer, mf_args:dict)->ALS_MF:
        mfer=None
    
        if factorizer == "WeightedALS_MF":
            mfer = WeightedALS_MF(
                R=R,
                fill_empty=mf_args['fill_empty'],
                w_m = mf_args['w_m'],w_obs = mf_args['w_obs'],
                latency = mf_args['latency'],
                l2_reg =  mf_args['l2_reg']   
            )
        elif factorizer == "ALS_MF":
            mfer = ALS_MF(
                R=R,latency = mf_args['latency'],
                l2_reg =  mf_args['l2_reg'],
            )
        return mfer

    print(factorizer)
    matrixfactorizer = construct_matrixfactorizer(R=matrix,factorizer=factorizer,mf_args=mf_args)
    h = matrixfactorizer.train(device=d,tmp_savepath=model_save_path, max_iteration=mf_args['epochs'])
    plotLoss(h['loss'], savename=os.path.join(model_save_path, "wmse.jpg"), showinline=vis_showinline)
    ret = {}
    for k in return_result:
        ret[k]=torch.load(
            os.path.join(model_save_path,f"{k}.pt")
        ).cpu()
    return ret



def Cross_MF(
    cross_matrix:torch.Tensor,testing_range:int,testing_user:list,
    model_args:dict,model:str="ALS_MF",on_device:torch.device=torch.device('cpu'),
    model_save_path=os.path.join(".","MF") , 
    result_saving_path = os.path.join(".","result"),
    show_loss=False
)->os.PathLike:
        
    print("MF .. ")
    prediction = mf(
        matrix=cross_matrix,
        model_save_path=model_save_path,d=on_device,
        factorizer=model,mf_args=model_args,
        vis_showinline=show_loss
    )['prediction']
    print("MF done ..")

    testing_user_book_prediction = prediction[testing_range:, 0:1000].numpy()

    
    print("generate recommend list ..")
    rlist = {}
    for i, testu in tqdm(enumerate(testing_user), total=len(testing_user)):
        rank = np.argsort(-testing_user_book_prediction[i]).tolist()
        rlist[str(testu)] = list(str(cate) for cate in rank)
    
    rlist_save = os.path.join(result_saving_path, "recommendlist.json")
    writejson(rlist,rlist_save)
    print(f"done .., recommend list is at {rlist_save}")
    return rlist_save

def cbmf_prediction(
    origin, clustering_predictions,
    alpha,
    testing_range:int,testing_user:list,
    book_clusters_path:os.PathLike, 
    user_cluster_path:os.PathLike,
    result_saving_path:os.PathLike
):
    origin_prediction = origin[testing_range:, 0:1000]
    book_cluster = loadjson(book_clusters_path) 
    user_course_cluster = loadjson(user_cluster_path)

    rlist = {}
    for i, testu in tqdm(enumerate(testing_user), total=len(testing_user)):
        cates = origin_prediction[i]
        itcoursecluster = np.Inf
        for k,v in user_course_cluster.items():
            if int(testu) in v:
                itcoursecluster = int(k)
                break
    
        cate_cluster = [0]*1000
        for catei in range(1000):
            for k,v in book_cluster.items():
                if str(catei) in v:
                    cate_cluster[catei] = int(k)
                    break
        
        cluster_suggest = clustering_predictions[
            itcoursecluster, cate_cluster
        ]

        finalcates = (1-alpha)*cates + alpha*cluster_suggest
        rank = np.argsort(-finalcates).tolist()
        rlist[testu] = list(str(cate) for cate in rank)

    writejson(
        rlist,
        os.path.join(result_saving_path,"recommendlist.json") 
    )
    return os.path.join(result_saving_path,"recommendlist.json") 
    



def CBMF(
    cluster_level_matrix:torch.Tensor, 
    model_args:dict,model:str="ALS_MF",
    on_device:torch.device=torch.device('cpu'),
    model_save_path=os.path.join(".","CBMF"),
    show_loss=False
):
    print("CBMF .. ")
    clustering_predictions = mf(
        matrix=cluster_level_matrix,
        model_save_path=model_save_path,d=on_device,
        factorizer=model,mf_args=model_args,
        vis_showinline=show_loss
    )['prediction']
    print("CBMF done ..")
    return clustering_predictions
    
    

if __name__ == "__main__":
    
    dataroot = os.path.join("..","data")
    datafolder = {
        "training_user_course":os.path.join(
            dataroot,"course","train.csv"
        ),
        "training_user_book":os.path.join(
            dataroot,"book","user_cate3_train.csv"
        ),
        "testing_user_course":os.path.join(
            dataroot, "course", "test.csv"
        ),
        "testing_user_book":os.path.join(
            dataroot, "book", "user_cate3_test.csv"
        )
    }
    dataset = Dataset(datafolder=datafolder)

    info, cross_matrix = build_cross_domain_matrix(
        dataset=dataset,
        domains=[
            [   
                ("training_user_book",False,True),
                ("testing_user_book",True,True)
            ],
            [
                ("training_user_course", False, False),
                ("testing_user_course",False,False)
            ]
        ],
        savedir=os.path.join(dataroot, "crossdomain","normalize"),
        return_data="torch"  
    )

    #info = loadjson(os.path.join(dataroot, "crossdomain","normalize","info.json"))
    mf_args = {
        'latency':40,'l2_reg':0.7,
        'fill_empty':torch.mean(cross_matrix).item()/2,
        'w_obs':1, 'w_m':0.001,
        'epochs':3
    }
    recommend_list_path = Cross_MF(
        cross_matrix=cross_matrix,
        testing_range=info['testing_range'],testing_user=info['testing_user'],
        model="WeightedALS_MF", model_args=mf_args,
        on_device=torch.device('cuda:3'),
        model_save_path=os.path.join("..","result","CBMF","normalize","model"),
        result_saving_path=os.path.join("..","result","CBMF","normalize")
    ) 
    #recommend_list_path = os.path.join("..","result","CBMF","normalize","recommendlist.json")
    print(recommend_list_path)
    Evaluate(
        result_root=os.path.join("..","result","CBMF","normalize"),
        recommendlist=recommend_list_path,
        gth=os.path.join("..","result","testing_user_groundtruth.json"),
        item_list=list(str(i) for i in range(1000))
    )
