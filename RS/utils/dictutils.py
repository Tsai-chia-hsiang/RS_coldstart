import os
import json
import sys
sys.path.append(os.path.dirname(__file__))

__all__ = ['loadjson','writejson']

def loadjson(jsfilepath:os.PathLike, encoding='utf-8'):
    
    """
    read a json format file

    **using ```json.load()```, so don't support binary format file

    - ```jsfilepath``` : the file path
    - ```encoding```:
        default : 'utf-8'
    """
    
    ret = None
    with open(jsfilepath, "r", encoding=encoding) as jf: 
        ret = json.load(jf)
    return ret

def writejson(dictionary:dict, jsfilepath:os.PathLike, encoding='utf-8')->None:

    """
    write data to a json file.
    
    ** using ```json.dump()```, so don't send a binary format data

    - ```dictionary``` : the data that want to write
    - ```jsifilepath``` : write to where
    - ```encoding``` : decfault: 'utf-8'

    Note that it uses the following setting:
    - indent = 4
    - ensure_ascii = False
    
    the reset of json setting remains as defualt.

    """
    with open(jsfilepath, "w+", encoding=encoding) as jf:
        json.dump(
            dictionary, jf, 
            indent=4, ensure_ascii=False
        )

