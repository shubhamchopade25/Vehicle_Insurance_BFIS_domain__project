import os
import sys
from src.exception import CustomExceptions
from src.logger import logging
from src.Utils import save_object,model_evaluation
from src.Utils import joblib_data_open
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
#from dataclasses import dataclass
#from src.components.data_transform import DataTransformation
#from src.components.data_ingection import DataIngection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# @dataclass
# class ModelTrainingConfig:
#     train_model_file_pickle_path=os.path.join('artifacts','best_model.pkl')
class modelTraining:
    def __init__(self):
        #self.model_train_pickle_config=ModelTrainingConfig()
        self.model=Pipeline({
                'LogisticRegression':LogisticRegression(),
                'KNeighborsClassifier':KNeighborsClassifier(),
                'GaussianNB':GaussianNB(),
                'DecisionTreeClassifier':DecisionTreeClassifier(),
                'RandomForestClassifier':RandomForestClassifier(),
                'AdaBoostClassifier':AdaBoostClassifier(),
                'GradientBoostingClassifier':GradientBoostingClassifier(),
                'XGBClassifier':XGBClassifier()
            }
            )
    def model_training_step(self,input_feature_train_df,target_feature_train_df):
        try:
            logging.info("Split training and test input data")
            X_train,X_test,y_train,y_test=train_test_split(
                input_feature_train_df,
                target_feature_train_df,
                train_size=0.90,random_state=42
                
            )
            # pipe=Pipeline({
            #     'LogisticRegression':LogisticRegression(),
            #     'KNeighborsClassifier':KNeighborsClassifier(),
            #     'GaussianNB':GaussianNB(),
            #     'DecisionTreeClassifier':DecisionTreeClassifier(),
            #     'RandomForestClassifier':RandomForestClassifier(),
            #     'AdaBoostClassifier':AdaBoostClassifier(),
            #     'GradientBoostingClassifier':GradientBoostingClassifier(),
            #     'XGBClassifier':XGBClassifier()
            # }
            # )
            # Hyperparameter tuning perform on this algorithm
            logging.info('Hyperparameter tuning perform on this algorithm')
            params = {
            "LogisticRegression": [{'penalty': ['l2'], 'max_iter': [10]}],
            "KNeighborsClassifier": [{'n_neighbors': [5], 'leaf_size': [30], 'metric': ['minkowski']}],
            "GaussianNB": [{'priors': [None], 'var_smoothing': [1e-9]}],
           
           
            "DecisionTreeClassifier": [{'criterion': ['gini', 'entropy'],
                            'max_depth': [1, 2, 5, 4, 8],
                            'min_samples_split': [2, 3],
                            'min_samples_leaf': [1, 2, 3],
                            'max_features': ['sqrt']}],
            "RandomForestClassifier": [{'criterion': ['gini'],
                            'max_depth': [10, 8, 4, 6, 9],
                            'max_features': ['sqrt'],
                            'n_estimators': [8, 16, 32, 64, 128, 256]}],
            "AdaBoostClassifier": [{'n_estimators': [5, 8, 10, 11],
                        'learning_rate': [0.01, 0.001, 0.02, 0.05]
                                }],
            "GradientBoostingClassifier": [{'learning_rate': [0.1, 0.01, 0.05, 0.001],
                                'max_features': [ 'log2'],
                                'n_estimators': [8, 6, 3, 6, 12, 25]}],
            "XGBClassifier": [{'learning_rate': [0.1, 0.01, 0.05, 0.001],
                        'n_estimators': [8, 16, 32, 64, 128, 256]}]
}
           # models=pipe.fit(X_train,y_train)
            model_data:dict=model_evaluation(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,model=self.model,param=params)
            
            # best_model_score = max(sorted(model_report.values()))
            # if best_model_score < 0.6:
            #     raise CustomExceptions("No best model found")
            # logging.info('Best found model training data set')
            
            # best_model_name = list(model_report.keys())[
            #     list(model_report.values()).index(best_model_score)
            # ]
            
            # best_model =pipe[best_model_name]
            best_model_name = max(model_data, key=lambda x: model_data[x]['train_model_score'])
            best_model = model_data[best_model_name]['model_instance']

            logging.info('Stored the best model in the pickle file.')

            joblib_data_open(
            file_path=self.model,
            obj=best_model  # Save the entire dictionary
                 )
            
            
            
            

            predicted=best_model.predict(X_test)
            model_accuracy_score = accuracy_score(y_test, predicted)
            print(model_accuracy_score)
            return best_model
        except Exception as e:
            raise CustomExceptions(e,sys)
            
        
        
        
        
   