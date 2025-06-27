from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from src.utils import save_obj, find_best_model
from src.logger import logging 
from src.exception import CustomException 
from dataclasses import dataclass 
import sys,os

@dataclass
class ModelTrainer_config: 
    Model_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.ModelTrainer_config = ModelTrainer_config()

    def trainModels(self,train_data,test_data):
        X_train, X_test, y_train, y_test = (train_data[:,:-1],test_data[:,:-1],train_data[:,-1],test_data[:,-1])

        models = {
            "LinearRegression" : LinearRegression(),
            "Lasso" : Lasso(),
            "Ridge" : Ridge(),
            "KNeighborsRegressor" : KNeighborsRegressor(),
            "SVR" : SVR(),
            "DecisionTreeRegressor" : DecisionTreeRegressor(),
            "RandomForestRegressor" : RandomForestRegressor(),
            "AdaBoostRegressor" : AdaBoostRegressor(),
            "GradientBoostingRegressor" : GradientBoostingRegressor(),
            "XGBRegressor" : XGBRegressor()
        }

        best_model,r2 = find_best_model(models,X_train, X_test, y_train, y_test)
        print(r2)
        logging.info('Found out the best model based upon R2_score')

        save_obj(best_model,self.ModelTrainer_config.Model_path)





