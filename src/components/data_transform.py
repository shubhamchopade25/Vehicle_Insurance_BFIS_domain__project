import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomExceptions
from src.logger import logging
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, FunctionTransformer,OrdinalEncoder
from sklearn.impute import SimpleImputer
from src.Utils import save_object
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel


    

@dataclass
class DataTransformationConfig:
    processor_obj_file_path: str = os.path.join('artifacts', 'proprocessor_data_a.pkl')
    train_preprocess_obj_file_path: str = os.path.join('artifacts', 'train_preprocess.csv')
    #target_preprocess_obj_file_path: str = os.path.join('artifacts', 'target_preprocess.csv')
    test_preprocess_obj_file_path: str = os.path.join('artifacts', 'test_preprocess.csv')

class DataTransformation:
        def __init__(self):
            self.data_transformation_config = DataTransformationConfig()
            self.ll = LabelEncoder()  # Initialize LabelEncoder here

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            # In this method, apply the transformations to the input data X

            # Start data transformations
            X = self.start_data_transform_obj(X, 'AGE')
            X = self.start_data_transform_obj(X, 'DRIVING_EXPERIENCE')
            X = self.cat_num(X)

            # Add any additional transformations here

            return X

        def start_data_transform_obj(self, df, column_name):
            if column_name=='AGE':
                df[column_name] = df[column_name].apply(lambda x: self.map_age_to_group(x))
                return df
            
            elif column_name=='DRIVING_EXPERIENCE':
                df[column_name] = df[column_name].apply(lambda x: self.map_driving_experience_to_group(x))
                return df
            else:
                pass
            

        def map_age_to_group(self,column_name):
            # Function to convert age to age group

            if column_name=='40-64':
                return 2
            elif column_name=='65+':
                return 1
            elif column_name=='26-39':
                return 3
            elif column_name=='16-25':
                return 0
            else:
                return 'Other'
            return df
    


        def map_driving_experience_to_group(self, column_name):
                if column_name=='20-29y':
                    return 2
                elif column_name=='0-9y':
                    return 0
                elif column_name=='10-19y':
                    return 3
                elif column_name=='30y+':
                    return 1
                else:
                    return 'Other'
                return  df

        def cat_num(self, df):
            ll = LabelEncoder()
           # ul=OrdinalEncoder()
            df['VEHICLE_YEAR'] = ll.fit_transform(df['VEHICLE_YEAR'])
            logging.info('Those columns we converted num. using ordinalencoder we faced some error like reshape and length of value not match  so we fixed it this error')
            income_transformed= ll.fit_transform(df['INCOME'])#.values.reshape(-1, 1)
            education_transformed= ll.fit_transform(df['EDUCATION'])
            df['INCOME'] = income_transformed.flatten()  # Flatten to match the length
            df['EDUCATION'] = education_transformed.flatten()  # Flatten to match the length
            df['GENDER'] = ll.fit_transform(df['GENDER'])
            df['TYPE_OF_VEHICLE'] = ll.fit_transform(df['TYPE_OF_VEHICLE'])
            return df
# class Read_data_from_source:
#     def __init__(self):
#         transform_obj=DataTransformation()
#         pass
        def read_data_transform(self, train_path,test_path):
            
            try:
                df_train_data = pd.read_csv(train_path)
                input_feature_test_df = pd.read_csv(test_path)

                logging.info("Read train and test data completed")
                logging.info("Obtaining preprocessing object")
                 # Separate features and target
                logging.info("Applying preprocessing object on training dataframe and testing dataframe.")
                
                input_feature_train = df_train_data.drop(columns=['OUTCOME'], axis=1)
                target_feature_train_df = df_train_data['OUTCOME']
                
                logging.info('We check the dataset have convert cat to num ')
                
                data_transformer = DataTransformation()
                transformed_train = data_transformer.transform(input_feature_train)
               
                
                
                
                #transformed_test = data_transformer.transform(df_test)
                # # Perform data transformations
                # x=self.start_data_transform_obj(df_train_data, 'AGE')
                # #df_test_data=self.start_data_transform_obj(df_test_data, 'AGE')
                # x=self.start_data_transform_obj(df_train_data, 'DRIVING_EXPERIENCE')
                # #df_test_data=self.start_data_transform_obj(df_test_data, 'DRIVING_EXPERIENCE')
                # x = self.cat_num(df_train_data)
                # #df_test_data = self.cat_num(df_test_data)
                # return x
                # Your other transformation logic here
                # os.makedirs(os.path.dirname(self.data_transformation_config.train_preprocess_obj_file_path),exist_ok=True)
                # input_feature_train_df.to_csv(self.data_transformation_config.train_preprocess_obj_file_path,index=False,header=True)
                # input_feature_test_df.to_csv(self.data_transformation_config.test_preprocess_obj_file_path,index=False,header=True)
                #input_feature_train_df.to_csv(self.data_transformation_config.train_preprocess_obj_file_path,index=False,header=True)
                
                
                
                logging.info(f"Saved preprocessing object.")
                save_object(
                    file_path=self.data_transformation_config.processor_obj_file_path,
                    
                    obj=transformed_train
                    
                            
                                            )

                return (
                    transformed_train,
                    target_feature_train_df,
                    input_feature_test_df,
                    self.data_transformation_config.processor_obj_file_path,
                )
            except Exception as e:
                raise CustomExceptions(e, sys)
           
