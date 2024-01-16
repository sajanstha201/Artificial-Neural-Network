import os
import sys
from src.exception import customException
from src.logger import logging
import pandas as pd
import numpy as np
class data_ingestionConfig:
    train_data_path=os.path.join('artifacts','train.csv')
    test_data_path=os.path.join('artifacts','test.csv')
class data_ingestion:
    def __init__(self):
        self.data_ingestionConfig=data_ingestionConfig()
    def data_ingestion_initiate(self):
        logging.info("Data ingestion process initiated.")
        try:
            train_data=pd.read_csv('/Users/sajanshrestha/Data Science/Neural_Network/NoteBook/train.csv')
            logging.info("completed reading the training dataset from source")
            test_data=pd.read_csv('/Users/sajanshrestha/Data Science/Neural_Network/NoteBook/test.csv')
            logging.info("completed reading the testing datset from source")
            os.makedirs(os.path.dirname(self.data_ingestionConfig.train_data_path),exist_ok=True)
            logging.info("created a file for storing the training and testing dataset")
            train_data.to_csv(self.data_ingestionConfig.train_data_path)
            logging.info("stored the training data set")
            test_data.to_csv(self.data_ingestionConfig.test_data_path)
            logging.info("stored the testing dataset")
        except Exception as e:
            logging.info(e)
            raise customException(e,sys)



if __name__=="__main__":
    data_ingestion=data_ingestion()
    data_ingestion.data_ingestion_initiate()