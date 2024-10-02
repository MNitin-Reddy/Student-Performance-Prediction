import os
import sys
import dill

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from exception import CustomException


import numpy as np
import pandas as pd



def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj,file_obj)


    except Exception as e:
        raise CustomException(e,sys)