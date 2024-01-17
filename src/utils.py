import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import customException
from src.logger import logging
from sklearn.metrics import accuracy_score
def save_obj(obj_path,obj):
    try:
        os.makedirs(os.path.dirname(obj_path),exist_ok=True)
        with open(obj_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        logging.info(e)
        raise customException(e,sys)
def model_evaluate(x_train,y_train,x_test,y_test,models):
    accuracy=models.copy()
    for model_name,model in models.items():
        model.fit(x_train,y_train)
        predict=model.predict(x_test)
        accuracy.update({model_name:accuracy_score(y_test,predict)})
    return accuracy
        