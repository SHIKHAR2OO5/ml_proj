import os,sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass 
from sklearn.model_selection import train_test_split
import pandas as pd
from src.components.data_tranformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass 
class DataIngestionConfig:
    train_data_path : str = os.path.join('artifacts','train_data.csv')
    test_data_path : str = os.path.join('artifacts','test_data.csv')
    raw_data_path : str = os.path.join('artifacts','raw_data.csv')

class DataIngestion:
    def __init__ (self):
        self.data_ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Initiated")
        try:
            df = pd.read_csv(r"D:\Shikhar\Python\ml_proj\src\notebook\Data\StudentsPerformance.csv")
            logging.info("DataFrame has been created from the fetched data")
            os.makedirs('artifacts',exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path,index=False)

            train_data,test_data = train_test_split(df,test_size=0.25,random_state=42)
            train_data.to_csv(self.data_ingestion_config.train_data_path,index=False)
            test_data.to_csv(self.data_ingestion_config.test_data_path,index=False)
            logging.info("Data Ingestion has been successfully done")

            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )

        except Exception as e :
            raise CustomException(e,sys)
        

if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    transformer_obj = DataTransformation()
    train_data_arr , test_data_arr = transformer_obj.data_transformation(train_data_path,test_data_path)
    ModelTrainer_obj = ModelTrainer()
    ModelTrainer_obj.trainModels(train_data_arr,test_data_arr)
    