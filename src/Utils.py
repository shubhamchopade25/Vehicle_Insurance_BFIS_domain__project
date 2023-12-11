import os
import sys
from src.logger import logging
import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.model_selection import GridSearchCV
import joblib

from src.exception import CustomExceptions
from sklearn.metrics import accuracy_score,f1_score,precision_score

def save_object(file_path, obj, **kwargs):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            data_to_save = {'obj': obj}
            data_to_save.update(kwargs)
            dill.dump(data_to_save, file_obj)

    except Exception as e:
        raise CustomExceptions(e, sys)
    
def joblib_data_open(file_path, obj,**kwargs):
    try:
        joblib.dump(file_path,'my_best_model.joblib')
    except Exception as e:
            raise CustomExceptions(e,sys)
    
    #This should resolve the "too many values to unpack" error. The key change is using pipe.named_steps.items() 
    #instead of pipe.steps to iterate over the named steps of the pipeline.(error: for name, model in pipe.steps: this used for pipeline)
def model_evaluation(X_train, y_train, X_test, y_test, model, param):
    try:
        model_data = {}
        for name, modell in model.named_steps.items():
            for i, j in param.items():
                if name == i:
                    gs = GridSearchCV(modell, j, cv=3)
                    gs.fit(X_train, y_train)

                    modell.set_params(**gs.best_params_)
                    train_modell = modell.fit(X_train, y_train)
                    logging.info(f'HyperParameter Tuning are completed for {i} algorithm')
        
                    y_train_pred = train_modell.predict(X_train)
                    y_test_pred = train_modell.predict(X_test)

                    train_model_score = accuracy_score(y_train, y_train_pred)
                    test_model_score = accuracy_score(y_test, y_test_pred)

                    #model_data[name] = train_model_score
                    model_data[name] = {
                        'model_instance': train_modell,
                        'train_model_score': train_model_score
                    }

        return model_data

    except Exception as e:
        raise CustomExceptions(e, sys)
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomExceptions(e, sys)
    
    