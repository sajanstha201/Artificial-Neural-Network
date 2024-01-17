import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import customException
from src.logger import logging
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
def save_obj(obj_path,obj):
    try:
        os.makedirs(os.path.dirname(obj_path),exist_ok=True)
        with open(obj_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        logging.info(e)
        raise customException(e,sys)
def model_evaluate(x_train,y_train,x_test,y_test,models,params):
    try:
        report={}
        for i in range(len(list(models.keys()))):
            model=list(models.values())[i]
            param=list(params.values())[i]
            gs=GridSearchCV(estimator=model,param_grid=param,cv=5)
            gs.fit(x_train,y_train)
            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)
            predict=model.predict(x_test)
            report[list(models.keys())[i]]=accuracy_score(y_test,predict)
        return report
    except Exception as e:
        logging.info(e)
        raise customException(e,sys)
        