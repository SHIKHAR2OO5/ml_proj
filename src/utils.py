import pickle,sys
from src.logger import logging 
from src.exception import CustomException
from sklearn.metrics import r2_score

def save_obj (obj, p):
    try:          
        with open (p,'wb') as file :  
            pickle.dump(obj, file)
            logging.info("Object has been saved!")
    
    except Exception as e :
        raise CustomException(e,sys)
    
def find_best_model(models:dict,X_train,X_test,y_train,y_test):
    try:
        r2 = 0
        best_model = None
        for keys in models.keys():
            model = models[keys]
            model.fit(X_train,y_train)
            pred = model.predict(X_test)
            score = r2_score(y_test,pred)
            if (score>r2):
                r2 = score
                best_model = model
        return (best_model,r2)
    
    except Exception as e:
        raise CustomException(e,sys)

