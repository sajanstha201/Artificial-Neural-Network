import os
import sys
import pandas as pd
import numpy as np
from src.exception import customException
from src.logger import logging
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,recall_score,f1_score,confusion_matrix


from src.components.data_ingestion import data_ingestion
from src.components.data_transformation import data_transformation
from src.utils import model_evaluate,save_obj
class model_trainerConfig:
    def __init__(self):
        self.model_trainer_path=os.path.join('artifacts','model.pkl')
class model_trainer:
    def __init__(self):
        self.model_trainerConfig=model_trainerConfig()
    def model_trainer_initiate(self,train_arr,test_arr):
        try:
            logging.info("starting the modle trainer procedure")
            models={
                "Logistic Regression":LogisticRegression(),
                "SVM":SVC(),
                "Naive Bayes":GaussianNB(),
                "KNN":KNeighborsClassifier(),
                "K Mean":KMeans(),
                "Decision Tree":DecisionTreeClassifier(),
                "Random Forest":RandomForestClassifier()
            }
            logging.info("loaded the required model")
            params={
                    "Logistic Regression":{
                    #'estimator':['C','class_weight','dual','n_jobs'],
                    'C':[0.1,1,10],
                    #'solver':['lbfgs','liblinear','sag','saga'],
                   'max_iter':[100,500,1000,50000]
                    },
                    "SVM":{
                    #'estimator':['C','break_ties','cache_size','gamma','kernel']
                    #'C':[0.01,0.1,1,10,10,100],
                    'kernel':['linear','rbf','poly','sigmoid'],
                    'gamma':[0.01,0.1,1,10,100],
                    #'degree':[2,3,4,5]
                    },
                "Naive Bayes":{
                   # 'estimator':['priors','var_smoothing'],
                    'alpha':[0.001,0.01,0.1,1,10,100]
                    },
                "KNN":{
                    #'n_neighbors':[5,6,7,8],
                    #'metric':['euclidean','manhattan','minkowski'],
                    #'algorithm':['auto','ball_tree','kd_tree'],
                    #'leaf_size':[10,20]
                    },
                "K Mean":{
                   # 'n_init':['auto']
                    #'n_clusters':[2,3,4,5,6,7],
                    #'max_iter':[100,200,300],
                    },
                "Decision Tree":{
                    'criterion':['gini','entropy','squared_error','absolute_error','poisson'],
                    #'splitter':['random','best'],
                    #'max_depth':[100,200,300],
                    #'min_samples_split':[1,2,3,4,5],
                   #'min_samples_leaf':[1,2,3,4,5],
                    #'max_features':[]
                    },
                "Random Forest":{
                    #'criterion':['gini','entropy','squared_error','absolute_error','poisson'],
                    #'splitter':['random','best'],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }
            logging.info("declared the hyperparameter for each model")
            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            logging.info("splitted the data for training and testing purpose")
            model_report=model_evaluate(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models,
                params=params
            )
            logging.info("got the model report of each model")
            best_model_score=np.max(list(model_report.values()))
            best_mode_name=[key for key,value in model_report.items() if(value==best_model_score)]
            best_model=models[best_mode_name[0]]
            if(best_model_score<0.6):
                raise customException("none of the model is selected")
            save_obj(
                obj_path=self.model_trainerConfig.model_trainer_path,
                obj=best_model
            )
            print(best_mode_name)
            return(
                accuracy_score(y_test,best_model.predict(x_test))
            )
        except Exception as e:
            logging.info(e)
            raise customException(e,sys)
if __name__=="__main__":
    train_path,test_path,test_result_path=data_ingestion().data_ingestion_initiate()
    train_arr,test_arr,_=data_transformation().data_transformation_initiate(train_path,test_path,test_result_path)
    accuracy=model_trainer().model_trainer_initiate(train_arr,test_arr)
    print(accuracy)
        