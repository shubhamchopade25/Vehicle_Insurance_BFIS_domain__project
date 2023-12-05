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
    processor_obj_file_path: str = os.path.join('artifacts', 'processor.pkl')

class DataTransformation:
        def __init__(self):
            self.data_transformation_config = DataTransformationConfig()

        def start_data_transform_obj(self):
            pass
            

        def age_to_group(self, df, column_name):
            # Function to convert age to age group

            if column_name== '40-64':
                return 2
            elif column_name== '65+':
                return 1
            elif column_name== '26-39':
                return 3
            elif column_name== '16-25':
                return 0
            else:
                return 'Other'
            return df
    


        def driving_experience_to_group(self, df, column_name):
                if column_name== '20-29y':
                    return 2
                elif column_name== '0-9y':
                    return 0
                elif column_name== '10-19y':
                    return 3
                elif column_name== '30y+':
                    return 1
                else:
                    return 'Other'
                return  df

        def cat_num(self, df):
            ll = LabelEncoder()
            df['VEHICLE_YEAR'] = ll.fit_transform(df['VEHICLE_YEAR'])
            df['INCOME'] = df['INCOME'].map({'upper class': 1, 'working class': 3, 'middle class': 2, 'poverty': 0})
            df['EDUCATION'] = df['EDUCATION'].map({'high school': 2, 'university': 1, 'none': 0})
            df['GENDER'] = df['GENDER'].map({'female': 0, 'male': 1})
            df['TYPE_OF_VEHICLE'] = ll.fit_transform(df['TYPE_OF_VEHICLE'])
            return df
        def read_data_transform(self, train_path, test_path):
            
            try:
                df_train_data = pd.read_csv(train_path)
                df_test_data = pd.read_csv(test_path)

                logging.info("Read train and test data completed")
                logging.info("Obtaining preprocessing object")

                # Perform data transformations
                self.age_to_group(df_train_data, 'AGE')
                self.age_to_group(df_test_data, 'AGE')

                self.driving_experience_to_group(df_train_data, 'DRIVING_EXPERIENCE')
                self.driving_experience_to_group(df_test_data, 'DRIVING_EXPERIENCE')

                df_train_data = self.cat_num(df_train_data)
                df_test_data = self.cat_num(df_test_data)

                # Your other transformation logic here

                logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

                # Separate features and target
                input_feature_train_df = df_train_data.drop(columns=['OUTCOME'], axis=1)
                target_feature_train_df = df_train_data['OUTCOME']

                input_feature_test_df = df_test_data

                # Apply categorical to numerical transformation
                input_feature_train_df = self.cat_num(input_feature_train_df)
                input_feature_test_df = self.cat_num(input_feature_test_df)

                # Your other transformation logic here

                logging.info(f"Saved preprocessing object.")
                save_object(
                    file_path=self.data_transformation_config.processor_obj_file_path,
                    obj=self  # Replace with your actual preprocessing object
                )

                return (
                    input_feature_train_df,
                    target_feature_train_df,
                    input_feature_test_df,
                    self.data_transformation_config.processor_obj_file_path,
                )
            except Exception as e:
                raise CustomExceptions(e, sys)
           
