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
            'Fare':[int(dict['Fare'])],
            'Sex':[dict['Sex']],
            'Embarked':[dict['Embarked']]
    }
    dataframe=pd.DataFrame(data)
    return dataframe
    
def prediction(dict):
    try:
        data=get_data_frame(dict)
        model_path="/Users/sajanshrestha/Data Science/Project/Titanic/artifacts/model.pkl"
        transformer_path="/Users/sajanshrestha/Data Science/Project/Titanic/artifacts/preprocessor.pkl"
        model=load_file(model_path)
        transformer=load_file(transformer_path)
        transformed_data=transformer.transform(data)
        prediction=model.predict(transformed_data)
        return prediction[0]
    except Exception as e:
        logging.info(e)
        raise customException(e,sys)
