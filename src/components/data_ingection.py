import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomExceptions
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transform import DataTransformation,DataTransformationConfig


@dataclass
class DataIngectionconfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path1: str=os.path.join('artifacts',"raw_train_data1.csv")
    raw_data_path2: str=os.path.join('artifacts',"raw_test_data2.csv")
class DataIngection:
    def __init__(self):
        self.data_ingection=DataIngectionconfig()
    def initiated_data_ingection(self):
        logging.info("Entered the data ingestion method or component in this method")
        try:
            df_train=pd.read_csv(r'D:\MY ALL  practice end to end projects\V_Insurance_Financial_project\Notebook\Dataset\train_insurance.csv')
            df_test=pd.read_csv(r'D:\MY ALL  practice end to end projects\V_Insurance_Financial_project\Notebook\Dataset\test_insurance.csv')
            logging.info("Raw data read for using pandas ")
            os.makedirs(os.path.dirname(self.data_ingection.train_data_path),exist_ok=True)# this for give location where you are stored data.
            df_train.to_csv(self.data_ingection.raw_data_path1,index=False,header=True)
            df_test.to_csv(self.data_ingection.raw_data_path2,index=False,header=True)
            logging.info("Train test split initiated already distrubuted two different csv train and test")
            
            # if we have combine data then you can split otherwise take without spliting.
            # train_set,test_set=train_test_split(df_train,train_size=0.90,random_state=42)
            # train_set.to_csv(self.data_ingection.train_data_path,index=False,header=True)
            df_train.to_csv(self.data_ingection.train_data_path,index=False,header=True)
            df_test.to_csv(self.data_ingection.test_data_path,index=False,header=True)
            
            return(
                self.data_ingection.train_data_path,
                self.data_ingection.test_data_path
            )
            
            logging.info("The data ingection is completed")
            
            
            
        except Exception as e:
            raise CustomExceptions(e,sys)
        
if __name__=="__main__":
    obj=DataIngection()
    train_data,test_data=obj.initiated_data_ingection()
    
    data_transformation=DataTransformation()
    train_arr,test_arr, _, _=data_transformation.read_data_transform(train_data,test_data)
