import numpy as np
import torch
import os 
import sys
from tqdm import tqdm
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
sys.path.append(os.path.dirname(__file__))
from plotutils import plot_tsne_2d, plotLoss
from dictutils import *
from mf import ALS_MF

"""
Some functions to process our dataset.
"""

def transpose_df(df:pd.DataFrame, col_to:str)->pd.DataFrame:
    col = df.columns.tolist()
    coldf = pd.DataFrame(data = col, columns=[col_to]) 
    tdf = pd.DataFrame(
        data = df.values.T,
        columns=list(i for i in range(df.shape[0]))
    )
    return pd.concat((coldf, tdf), axis = 1)

def user_item_clustering(User_Item_df:pd.DataFrame, num_clusters:list, savingpath:os.PathLike, d:torch.device, plot_show=False):
    
    def clusterting(items:list, x:np.ndarray, k:int, savingpath:os.PathLike, visshow=False):
        cluster = KMeans(n_clusters=k)
        print("clutsering ..")
        cluster.fit(x)
        y = cluster.predict(x)
        print("OK ..")

        cluster_result = {}
        for i in range(k):
            cluster_result[i] = []
        y_ = y.tolist()
        for idx, i in enumerate(items):
            cluster_result[y_[idx]].append(i)
    
        if not os.path.exists(savingpath):
            os.mkdir(savingpath)
        writejson(cluster_result, os.path.join(savingpath, "cluster.json"))
        print("visualization ..")

        plot_tsne_2d(
            X=x, name=f"clustering_{k}", labels=y,
            savepath=os.path.join(savingpath,f"clustering_{k}.jpg"),
            showinline=visshow
        )    

    mf=ALS_MF(
        R = torch.tensor(
            User_Item_df.drop(columns=['uid']).values
        )
    )
    
    mf_model_path = os.path.join(savingpath, "model")
    if not os.path.exists(mf_model_path):
        os.mkdir(mf_model_path)
    
    h = mf.train(
        device=d,
        tmp_savepath=os.path.join(savingpath, "model")
    )
    
    plotLoss(
        loss=h['loss'], savename=os.path.join(mf_model_path, "mse.jpg"),
        showinline=plot_show
    )

    for i,k in zip(['user','item'],num_clusters):
        
        print(i)
        latency =torch.load( os.path.join(mf_model_path, f"{i}.pt")).cpu().numpy()
        print(latency.shape)
        
        c_path = os.path.join(savingpath, f"{i}clustering")
        if not os.path.exists(c_path):
            os.mkdir(c_path)
        
        items = None
        if i == "user":
            items = User_Item_df.uid.tolist()
        elif i == "item":
            items = User_Item_df.drop(columns=['uid']).columns.tolist()
        
        clusterting(
            items=items,
            x = latency,k=k,
            savingpath=c_path,
            visshow=plot_show  
        )


class Dataset():

    def __init__(self, datafolder:dict) -> None:
        print("build dataset")
        self._data = self._read_data(datafolder)
        self._dname =list(self._data.keys())
        
    def _read_data(self, datafolder:dict)->dict:
        ret = {}
        for dataname, path in datafolder.items():
            print(f"read {dataname}:{path}")
            ret[dataname] = pd.read_csv(path)
            print("..OK")
        return ret
    
    def dataname(self) -> list:
        return self._dname

    def getdata(self, dataname:str, normalize_value=False)-> pd.DataFrame:

        if dataname not in self._data:
            print(f"No {dataname} such a data")
            raise KeyError
        df = self._data[dataname].copy()
        if normalize_value:
            uid = df.uid.tolist()
            df= df.drop(columns=['uid'])
            v = normalize(df.values, axis=1, norm="l1")
            df_nor = pd.DataFrame(data=v, columns=df.columns)
            df_nor['uid'] = uid
            col_order = df_nor.columns.tolist()[-1:]\
                +df_nor.columns.tolist()[:-1]
            return df_nor[col_order]

        return df
    
    def extraction_value(self,dname:str, dropout_col:list, to_torch=False):
        
        if dname not in self._dname:
            print("no such a data")
            raise KeyError
       
        value =(
            self._data[dname].drop(columns=dropout_col)
        ).values
    
        if to_torch:
            return torch.tensor(value, dtype=torch.double)
        return value

    def mask_dataset(self, dataname:str)->pd.DataFrame:
        """
        To make the target df 's value to 0
        (e.g.: mask up the ground truth of testing data)
        """
        if dataname not in self._dname:
            print(f"No such {dataname} data")
            raise KeyError
        df = self._data[dataname].copy()
        uid = df.uid.tolist()
        df = df.drop(columns = ['uid'])
        df_mask = pd.DataFrame(
            data=df.values*0.0,
            columns=df.columns.tolist()
        )
        df_mask['uid'] = uid
        col_order = df_mask.columns.tolist()[-1:]+df_mask.columns.tolist()[:-1]
        return df_mask[col_order]



def combine_multi_domain(Dataset:Dataset=None, datafolder:dict=None, domains:list=[])->pd.DataFrame:


    if (Dataset is not None) and (datafolder is not None):
        print("Warning : if give a Dataset, it will ignore the datafolder")
    dataset = Dataset
    if dataset is None:
        dataset = Dataset(datafolder=datafolder)
    
    def concate_along_row_dimension(m:pd.DataFrame, mi:pd.DataFrame):
        """
        concate m and mi along thire row dimesion by using
        - ```pd.concate(m, mi, axis=0)```
        """
        if m is None:
            return mi
        else:
            return pd.concat([m,mi],axis=0)

    crossdomain_user_item_matrix = None
    for di in domains:
        single_domain = None
        for dii in di:
            dname, mask, nor = dii
            print(f"name : {dname}, mask: {mask}, normalization: {nor}")
            dii_df = None
            if mask:
                print("     mask up")
                dii_df = dataset.mask_dataset(dname)
            else:
                print(f"    normalization along row : {nor}")
                dii_df = dataset.getdata(dataname=dname,normalize_value=nor)
            
            single_domain = concate_along_row_dimension(single_domain, dii_df)
        
        if crossdomain_user_item_matrix is None:
            crossdomain_user_item_matrix = single_domain
        else:
            crossdomain_user_item_matrix = pd.concat(
                [crossdomain_user_item_matrix, single_domain.drop(columns=['uid'])],
                axis=1
            )
        
    return crossdomain_user_item_matrix

def build_cross_domain_matrix(dataset:Dataset, domains:list, savedir:os.PathLike, return_data="pd")->tuple:
    
    """
    - ```domains```: 
        
        a 2d list, each subentry means how to construct a complete domain matrix.
        it will combine one domain along thire row dimension,
        and combine all domains by column dimension.Note that it will contain only 1 ```uid``` column
        inside the crossdomain user-item matrix

        And, each entry is a tuple, include : 
        
        (
            Dataset_key : str, 
            wether it needed to be masked up to all 0(usually for testing part): bool ,
            normalize the matrix for each row row or not: bool
        )
        
            Example:
            [
                [
                ('user_course_train',False,False),
                ('user_course_test',False,False)
                ], 
                [
                ('user_book_train',False,'True'),
                ('user_book_test',True, False)
                ]
            ]
        
    - ```cross_domain_matrix_save``` :
        save cross_domain.csv and cross_domain.np file in cross_domain_matrix_save.

        default : working directory
    
    """
    if not os.path.exists(savedir):
        print(f"make {savedir} dir")
        os.mkdir(savedir)
    
    info = {
        'testing_range':dataset.getdata(
            dataname="training_user_book"
        ).shape[0],
        'testing_user': dataset.getdata(
            dataname="testing_user_book"
        ).uid.tolist(),
        'all_book':dataset.getdata(
            dataname="testing_user_book").drop(columns=['uid']
        ).columns.tolist(),
        'all_course':dataset.getdata(
            dataname="testing_user_course").drop(columns=['uid'] 
        ).columns.tolist()
    }
    writejson(info, os.path.join(savedir,"info.json"))

    crossdomain_user_item_df = combine_multi_domain(Dataset=dataset, domains=domains)
    
    cross_domain_df_saving_path = os.path.join(savedir, "cross_domain.csv")
    print(f"save cross domain df to : {cross_domain_df_saving_path}")
    crossdomain_user_item_df.to_csv(cross_domain_df_saving_path,index =False)
    
    cross_domain_matrix_saving_path = os.path.join(savedir, "cross_domain")
    print(f"save cross domain matrix to : {cross_domain_matrix_saving_path}")
    cross_domain_matrix = (crossdomain_user_item_df.drop(columns=['uid'])).values
    np.save(cross_domain_matrix_saving_path,cross_domain_matrix)
    
    if return_data == "pd":
        return info, crossdomain_user_item_df
    elif return_data == "np":
        return info, cross_domain_matrix
    elif return_data == "torch":
        return info, torch.tensor(cross_domain_matrix,dtype=torch.double)


def cluster_level_matrix(
    R:pd.DataFrame,user_cluster:dict, item_cluster:dict,
    thr=5
)->np.ndarray:

    UI = np.zeros(
        (len(user_cluster.keys()),len(item_cluster.keys()))
    )
    print(UI.shape)

    z = 0
    t = 0
    for uu, users in tqdm(user_cluster.items(), total= len(user_cluster.keys())):
        cb_sub=R[R['uid'].isin(users)]
        for ii, items in item_cluster.items():
            t+=1
            #cb_sub=R[R['uid'].isin(users)]
            if not cb_sub.shape[0]:
                UI[int(uu)][int(ii)] = 0
                continue
            
            cb_sub_i = cb_sub[items]
            if not cb_sub_i.shape[0]:
                print("Item Error!!!")
                return None
            
            v = cb_sub_i.values
            obs_num = np.count_nonzero(v)
            if obs_num < thr:
                z +=1 
                UI[int(uu)][int(ii)] = 0
            else:
                UI[int(uu)][int(ii)] = np.sum(v)/obs_num
    print(f"{z}/{t}")
    return UI