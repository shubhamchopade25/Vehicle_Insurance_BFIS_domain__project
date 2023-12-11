import os
import sys
from src.exception import CustomExceptions
from src.logger import logging
from src.components.data_transform import DataTransformation
from src.components.model_training import modelTraining
from src.Utils import load_object
import pandas as pd
import joblib



class PredictPipeline:
    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        try:
            # Load the model here
            return joblib.load('my_best_model.joblib')
        except Exception as e:
            raise CustomExceptions(e, sys)
    def predict(self, features):
        try:
            process_data_path = os.path.join('artifacts', 'proprocessor_data_a.pkl')
            
            # Load model data
            process_data = load_object(file_path=process_data_path)
            
            # Check if process_data is an instance of DataTransformation
            if isinstance(process_data, DataTransformation):
                processed_features = process_data.transform(features)
            else:
                logging.info("Loaded object is not an instance of DataTransformation.")
            if isinstance(self.model,modelTraining): # replace with actual class
                logging.info("Best model instance is not the expected type.")
                 # Perform prediction
                pred_analysis = self.model.predict(processed_features)
                return pred_analysis
            
           
        except Exception as e:
            raise CustomExceptions(e, sys)
           
class CustomInputData:
    def __init__(self,
                 ID:int, AGE:str, GENDER:str, DRIVING_EXPERIENCE:str, EDUCATION:str, INCOME:str,
       CREDIT_SCORE:float, VEHICLE_OWNERSHIP:int, VEHICLE_YEAR:str, MARRIED:int,
       CHILDREN:int, POSTAL_CODE:int, ANNUAL_MILEAGE:int, SPEEDING_VIOLATIONS:int,
       DUIS:int, PAST_ACCIDENTS:int, TYPE_OF_VEHICLE:str):
        try:
            self.ID=ID
            self.AGE=AGE
            self.GENDER=GENDER
            self.DRIVING_EXPERIENCE=DRIVING_EXPERIENCE
            self.EDUCATION=EDUCATION
            self.INCOME=INCOME
            self.CREDIT_SCORE=CREDIT_SCORE
            self.VEHICLE_OWNERSHIP=VEHICLE_OWNERSHIP
            self.VEHICLE_YEAR=VEHICLE_YEAR
            self.MARRIED=MARRIED
            self.CHILDREN=CHILDREN
            self.POSTAL_CODE=POSTAL_CODE
            self.ANNUAL_MILEAGE=ANNUAL_MILEAGE
            self.SPEEDING_VIOLATIONS=SPEEDING_VIOLATIONS
            self.DUIS=DUIS
            self.PAST_ACCIDENTS=PAST_ACCIDENTS
            self.TYPE_OF_VEHICLE=TYPE_OF_VEHICLE
        except Exception as e:
            raise CustomExceptions(e,sys)
        
    def collect_data_user_to_dataframe(self):
        try:
            user_data_dict={
            "ID":[self.ID],
            "AGE":[self.AGE],
            "GENDER":[self.GENDER],
            "DRIVING_EXPERIENCE":[self.DRIVING_EXPERIENCE],
            "EDUCATION":[self.EDUCATION],
            "INCOME":[self.INCOME],
            "CREDIT_SCORE":[self.CREDIT_SCORE],
            "VEHICLE_OWNERSHIP":[self.VEHICLE_OWNERSHIP],
            "VEHICLE_YEAR":[self.VEHICLE_YEAR],
            "MARRIED":[self.MARRIED],
            "CHILDREN":[self.CHILDREN],
            "POSTAL_CODE":[self.POSTAL_CODE],
            "ANNUAL_MILEAGE":[self.ANNUAL_MILEAGE],
            "SPEEDING_VIOLATIONS":[self.SPEEDING_VIOLATIONS],

            "DUIS":[self.DUIS],
            "PAST_ACCIDENTS":[self.PAST_ACCIDENTS],
            "TYPE_OF_VEHICLE":[self.TYPE_OF_VEHICLE]
            }
            return pd.DataFrame(user_data_dict)
        except Exception as e:
            raise CustomExceptions(e,sys)
            
        