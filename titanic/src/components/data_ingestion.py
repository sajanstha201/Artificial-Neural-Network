import os
import sys
from src.exception import customException
from src.logger import logging
import pandas as pd
import numpy as np

class data_ingestionConfig:
    train_data_path=os.path.join('artifacts','train.csv')
    test_data_path=os.path.join('artifacts','test.csv')
    test_result_data_path=os.path.join('artifacts','test_result.csv')
class data_ingestion:
    def __init__(self):
        self.data_ingestionConfig=data_ingestionConfig()
    def data_ingestion_initiate(self):
        logging.info("Data ingestion process initiated.")
        try:
            train_data=pd.read_csv('/Users/sajanshrestha/Data Science/Project/Titanic/NoteBook/train.csv')
            train_data=train_data.drop(columns=['Name','Ticket','Cabin','PassengerId','Fare'])
            logging.info("completed reading the training dataset from source")
            
            test_data=pd.read_csv('/Users/sajanshrestha/Data Science/Project/Titanic/NoteBook/test.csv')
            test_data=test_data.drop(columns=['Name','Ticket','Cabin','PassengerId','Fare'])
            logging.info("completed reading the testing datset from source")
            
            test_result_data=pd.read_csv('/Users/sajanshrestha/Data Science/Project/Titanic/NoteBook/test_result.csv')
            logging.info("completed reading the test result dataset from source")
            
            os.makedirs(os.path.dirname(self.data_ingestionConfig.train_data_path),exist_ok=True)
            logging.info("created a file for storing the training and testing dataset")
            
            train_data.to_csv(self.data_ingestionConfig.train_data_path,index=False,header=True)
            logging.info("stored the training data set")
            
            test_data.to_csv(self.data_ingestionConfig.test_data_path,index=False,header=True)
            logging.info("stored the testing dataset")
            
            test_result_data.to_csv(self.data_ingestionConfig.test_result_data_path,index=False,header=True)
            logging.info("stored the test result dataset")
        except Exception as e:
            logging.info(e)
            raise customException(e,sys)
        return(
            self.data_ingestionConfig.train_data_path,
            self.data_ingestionConfig.test_data_path,
            self.data_ingestionConfig.test_result_data_path
        )
        



if __name__=="__main__":
    data_ingestion=data_ingestion()
    data_ingestion.data_ingestion_initiate()