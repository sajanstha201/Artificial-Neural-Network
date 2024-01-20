import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn import pipeline
from sklearn.impute import SimpleImputer
from src.exception import customException
from src.logger import logging
from src.components.data_ingestion import data_ingestion
from src.utils import save_obj
class data_transformationConfig:
    def __init__(self):
        self.preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
class data_transformation:
    def __init__(self):
        self.data_transformationConfig=data_transformationConfig()
    def get_data_transformer(self):
        try:
            num_fcol=['Pclass','Age','SibSp','Parch']
            cat_fcol=['Sex','Embarked']
            num_transformer=pipeline.Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            logging.info(f"Numerical column: {num_fcol}")
            cat_transformer=pipeline.Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot",OneHotEncoder())
                ]
            )
            logging.info(f"Categorical column: {cat_fcol}")
            Preprocessor=ColumnTransformer(
                [
                    ('num_col_tranformer',num_transformer,num_fcol),
                    ('cat_col_tranformer',cat_transformer,cat_fcol)
                ]
            )
            logging.info("preprocessor making is completed")
            return Preprocessor
        except Exception as e:
            logging.info(e)
            raise customException(e,sys)
        

    def data_transformation_initiate(self,train_path,test_path,test_result_path):
        try:
            logging.info("data transformation initiate")
            preprocessor=self.get_data_transformer()
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)
            test_result_data=pd.read_csv(test_result_path)['Survived']
            logging.info("successfully loaded the traning and testing dataset for preprocessing")
            
            target_feature="Survived"
            target_feature_train_data=train_data[target_feature]
            input_feature_train_data=train_data.drop(columns=[target_feature],axis=1)
            logging.info("target result is seperated form the training dataset")

            preprocessor.fit(input_feature_train_data)
            logging.info("successfully fit the merged input feature")
            
            input_feature_train_arr=preprocessor.transform(input_feature_train_data)
            logging.info("successfully transformed the training dataset")
            input_feature_test_arr=preprocessor.transform(test_data)
            logging.info("successfully transformed the testing dataset")

            train_arr=np.concatenate((input_feature_train_arr,np.array(target_feature_train_data).reshape(-1,1)),axis=1)
            test_arr=np.concatenate((input_feature_test_arr,np.array(test_result_data).reshape(-1,1)),axis=1)
            logging.info("concatenated the input feature and target feature for both training and testing dataset")

            save_obj(
                obj_path=self.data_transformationConfig.preprocessor_path,
                obj=preprocessor
                )
            logging.info("successfully save the object in preprocessor.pkl file")
            return(
                train_arr,
                test_arr,
                self.data_transformationConfig.preprocessor_path
            )
        except Exception as e:
            logging.info(e)
            raise customException(e,sys)  
    


if __name__=="__main__":
    train_path,test_path,test_result_path=data_ingestion().data_ingestion_initiate()
    train_arr,test_arr,preprocessor_path=data_transformation().data_transformation_initiate(
        train_path=train_path,
        test_path=test_path,
        test_result_path=test_result_path
        )
    
        