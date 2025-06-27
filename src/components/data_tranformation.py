import sys,os
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass 
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer 
from src.utils import save_obj

@dataclass 
class DataTranformation_Config:
    preprocessor_path : str = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.DataTranformation_Config = DataTranformation_Config()
    
    def get_preprocessor(self):

        num_features = ['reading score','writing score']
        cat_features = [ 'gender', 'race/ethnicity', 'parental level of education', 'lunch','test preparation course']

        num_pipeline = Pipeline (
            steps= [
               ('Imputer',SimpleImputer(strategy="median")),
               ('Scalar',StandardScaler()) 
            ]
        )

        cat_pipeline = Pipeline (
            steps= [
               ('Imputer',SimpleImputer(strategy="most_frequent")),
               ('Encoder',OneHotEncoder(drop='first')) 
            ]
        )

        preprocessor = ColumnTransformer([
            ('num_tranformer',num_pipeline,num_features),
            ('cat_transfomer',cat_pipeline,cat_features)
        ]
        )

        logging.info("Preprocessor as been created")

        return (preprocessor)


    def data_transformation(self,train_data_path,test_data_path):
        try :
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)
            preprocessor = self.get_preprocessor()
            target = 'math score'
            train_data_target = train_data[[target]].to_numpy()
            test_data_target = test_data[[target]].to_numpy()
    
            train_data.drop(target,axis=1,inplace = True)
            test_data.drop(target,axis=1, inplace = True)

            train_data_arr = preprocessor.fit_transform(train_data)
            test_data_arr = preprocessor.transform(test_data)
            save_obj(preprocessor,self.DataTranformation_Config.preprocessor_path)
            logging.info("Preprocessing has been completed")

            train_data_arr = np.concatenate([train_data_arr,train_data_target],axis=1)
            test_data_arr = np.concatenate([test_data_arr,test_data_target],axis=1)

            logging.info('Concatenation has been done')

            return (
                train_data_arr,
                test_data_arr
            )

        except Exception as e:
            raise CustomException(e,sys)
       