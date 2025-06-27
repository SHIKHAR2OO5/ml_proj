import pickle
from src.logger import logging 

def save_obj (obj, p):
    with open (p,'wb') as file :  
        pickle.dump(obj, file)
        logging.info("Object has been saved!")
