import os
import sys
import pandas as pd
import numpy as np
from src.exception import customException
from src.logger import logging
from src.utils import load_file

def get_data_frame(dict):
    data={
            'Pclass':[int(dict['Pclass'])],
            'Age':[int(dict['Age'])],
            'SibSp':[int(dict['spouses'])+int(dict['Sibling'])],
            'Parch':[int(dict['parent'])],
            'Sex':[dict['Sex']],
            'Embarked':[dict['Embarked']]
    }
    dataframe=pd.DataFrame(data)
    return dataframe
    
def prediction(dict):
    try:
        data=get_data_frame(dict)
        model_path="artifacts/model.pkl"
        transformer_path="artifacts/preprocessor.pkl"
        model=load_file(model_path)
        transformer=load_file(transformer_path)
        transformed_data=transformer.transform(data)
        prediction=model.predict(transformed_data)
        return int(prediction[0])
    except Exception as e:
        logging.info(e)
        raise customException(e,sys)
if __name__=="__main__":
        model_path="artifacts/model.pkl"
        transformer_path="artifacts/preprocessor.pkl"
        model=load_file(model_path)
        transformer=load_file(transformer_path)
        data={
            'Pclass':[1],
            'Age':[25],
            'SibSp':[0],
            'Parch':[0],
            'Sex':['male'],
            'Embarked':['S']
    }
        dataframe=pd.DataFrame(data)
        transformed_data=transformer.transform(dataframe)
        prediction=model.predict(transformed_data)
        print(int(prediction[0]))