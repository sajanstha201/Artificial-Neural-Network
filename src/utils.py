import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import customException
from src.logger import logging
def save_obj(obj_path,obj):
    try:
        os.makedirs(os.path.dirname(obj_path),exist_ok=True)
        with open(obj_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        logging.info(e)
        raise customException(e,sys)