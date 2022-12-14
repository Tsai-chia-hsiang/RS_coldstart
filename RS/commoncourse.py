import os
import sys
from tqdm import tqdm
import numpy as np
import torch
from torch.nn.functional import normalize
sys.path.append(os.path.dirname(__file__))
from utils.evaluation import Evaluate, item_order
from utils.dictutils import *
from utils.dataset import Dataset as Mydataset

def build_2domain_feature_extraction_matrix(
    d1_matrix:torch.Tensor,d2_matrix:torch.Tensor,
    normalize_domain:list = [],
    device:torch.device = torch.device('cpu'),
    savingpath:os.PathLike=None,
    saving_type:list=['torch','numpy'],
    return_result:bool=True
)->torch.Tensor:
    """
    build d1 x d2 frequency matrix.
    result = d1_matrix.T @ d2_matrix.
    - ```d1_matrix```: 
    
        frequency matrix of each user rates items in d1 
        (i.e. user x item_d1)
    - ```d2_matrix```: 
        frequency matrix of each user rates items in d2
        (i.e. user x item_d2)
    
    - ```normalize_domain``` :
        a list of strings. 
        To set which domain needed to apply l1-normalization
        
        (i.e. mean )
        along the ```row```
        dimension. Note that it can be set to both domain.
        
        (e.g. ('d1', 'd2'))
    
    - ```savingpath```:
        the result saving path.
        default is ```None```, which means no write to disk.
        if set a path, then will save it as the 
        
        ```saving_type```
        (The torch format will automatically 
        fill ```.pt``` to postfix)
    
    - ```device```: 
    
        torch.device
    - ```return_result```: 
        Wether the result needed to be return.
        defualt : Ture
    """
    if d1_matrix.size()[0] != d2_matrix.size()[0]:
        print("Error! User numbers are not the same!")
        return None
    
    d1_ = d1_matrix.to(device=device, dtype=torch.double)
    d2_ = d2_matrix.to(device=device, dtype=torch.double)
    
    if 'd1' in normalize_domain:
        d1_ = normalize(d1_, p=1.0, dim = 0)
    if 'd2' in normalize_domain:
        d2_ = normalize(d2_, p = 1.0, dim = 0)

    r = (d1_.T @ d2_).cpu()
    
    if savingpath is not None:

        for ftype in saving_type:
            
            if ftype == "torch":
                torch.save(r,f"{savingpath}.pt")
            elif ftype =="numpy":
                np.save(savingpath, r.numpy())
    
    if return_result:
        return r


def topK_with_cf(
    train_user_book_course_matrix: np.ndarray,
    test_users:list,test_user_course_matrix:np.ndarray,
    savingpath:os.PathLike
)->dict:

    recommend_list = {}
    for i, uid in enumerate(tqdm(test_users)):
        testi = test_user_course_matrix[i][:]
        itscourses = np.where(testi>0)[0]
        cf = np.sum(
            train_user_book_course_matrix[:, itscourses],
            axis=1
        )/itscourses.shape[0]
        recommend_list[uid] = item_order(cf)
    
    writejson(recommend_list, savingpath)


def recommend_according_course_selection_records(dataset:Mydataset, resultroot:os.PathLike, d:torch.device):
    
    """
    resultroot 
    
        the path where the result
        - cates x courses matrix
        - recommendlist

        will be saved.

    """
    
    training_user_course_tensor = dataset.extraction_value(
        "training_user_course",
        dropout_col=['uid'], to_torch=True
    )
    training_user_book_tensor = dataset.extraction_value(
        "training_user_book", 
        dropout_col=['uid'], to_torch=True
    )
    print("build book x user ..", end=" ")
    book_user_matrix= build_2domain_feature_extraction_matrix(
        d1_matrix=training_user_book_tensor, 
        d2_matrix=training_user_course_tensor,
        normalize_domain=['d2'],
        device=d,
        savingpath=os.path.join(resultroot, "bookcate3_course")
    )
    print("OK ..")

    #print(book_user_matrix.size())
    test_user = list(
        map(lambda x:str(x), 
        dataset.getdata(dataname="testing_user_course").uid.tolist()
    ))
    test_user_course_matrix = dataset.extraction_value(
        dname="testing_user_course", dropout_col=['uid']
    )
    #print(test_user_course_matrix.shape)

    print("prediction ..")
    rslist_save_to=os.path.join(resultroot, "recommendlist.json")
    prediction = topK_with_cf(
        train_user_book_course_matrix=book_user_matrix.numpy(),
        test_users=test_user,
        test_user_course_matrix=test_user_course_matrix,
        savingpath=rslist_save_to
    )
    print(f"OK .. save at {rslist_save_to}")
    return rslist_save_to


def main():
    """
    for script execution (if need background exection)

    (e.g. nohup python commoncourse.py)
    
    """
    dataroot = os.path.join("..","data")
    resultroot = os.path.join("..", "result", "commoncourse")
    """
    dataset = Mydataset(datafolder = {
        'training_user_course':os.path.join(
            dataroot,"course","train.csv"
        ),
        'training_user_book':os.path.join(
            dataroot,"book","user_cate3_train.csv"
        ),
        'testing_user_course':os.path.join(
            dataroot, "course", "test.csv"
        )
    }
    )

    
    if not os.path.exists(resultroot):
        os.mkdir(resultroot)
    
    rslist_saving_path = generate_rslist(
        dataset=dataset, 
        resultroot=resultroot, 
        d=torch.device('cuda:3')
    )
    """
    rslist_saving_path=os.path.join(
        resultroot, "recommendlist.json"
    )

    Evaluate(
        result_root=resultroot, 
        recommendlist = rslist_saving_path,
        gth = os.path.join("..", "result", "testing_user_groundtruth.json"),
        item_list=list(str(i) for i in range(1000))
    )


if __name__ == "__main__":
    main()