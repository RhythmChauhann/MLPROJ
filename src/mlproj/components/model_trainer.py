import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.mlproj.exception import CustomException
from src.mlproj.logger import logging
from src.mlproj.utils import save_object, evaluate_models

import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from urllib.parse import urlparse
import dagshub
dagshub.init(repo_owner='rhythmchauhann',
             repo_name='MLPROJ',
             mlflow=True)


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifact','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def eval_metrics(self,actual,pred):
         rmse= np.sqrt(mean_squared_error(actual,pred))
         mae = mean_absolute_error(actual,pred)
         r2 = r2_score(actual,pred)
         return rmse, mae,r2

    def initiate_model_trainer(self,train_array,test_array):
            try:
                logging.info("Split training and test input data")
                X_train,y_train,X_test,y_test = (
                    train_array[:,:-1],
                    train_array[:,-1],
                    test_array[:,:-1],
                    test_array[:,-1]
                )
                models={
                    "Random Forest":RandomForestRegressor(),
                    "Decission Tree":DecisionTreeRegressor(),
                    "Gradient Boost":GradientBoostingRegressor(),
                    "Linear Regression":LinearRegression(),
                    "XGBRegressor":XGBRegressor(),
                    "Catboosting Regressor":CatBoostRegressor(verbose=False),
                    "AdaBoost Regressor":AdaBoostRegressor(),
                    }
                params={
                "Decission Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boost":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Catboosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
                }
                
                model_report:dict = evaluate_models(X_train,y_train,X_test,y_test,models,params)
                #to get the best model form the dictionary
                best_model_score = max(sorted(model_report.values())) 
                #to get the best model namee from dictionary
                best_model_name=list(model_report.keys())[
                    list(model_report.values()).index(best_model_score)
                ]
                best_model = models[best_model_name]
                print(f"This is the best model :{best_model_name}")
                model_names = list(params.keys())
                actual_model = ""
                for model in model_names:
                     if model == best_model_name:
                          actual_model = actual_model+model
                best_params = params[actual_model]

                mlflow.set_registry_uri("https://dagshub.com/RhythmChauhann/MLPROJ.mlflow")
                tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
                #mlflow pipeline
                with mlflow.start_run():

                    predicted_qualities = best_model.predict(X_test)

                    (rmse, mae, r2) = self.eval_metrics(y_test, predicted_qualities)

                    mlflow.log_params(best_params)

                    mlflow.log_metric("rmse", rmse)
                    mlflow.log_metric("r2", r2)
                    mlflow.log_metric("mae", mae)


                    # Model registry does not work with file store
                    if tracking_url_type_store != "file":

                        # Register the model
                        # There are other ways to use the Model Registry, which depends on the use case,
                        # please refer to the doc for more information:
                        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                        try:
                            mlflow.sklearn.log_model(best_model, "model")
                        except Exception as e:
                            print("⚠️ Model logging to DagsHub skipped due to unsupported endpoint:", e)
                            mlflow.sklearn.save_model(best_model, "artifacts/model")
                            mlflow.log_artifact("artifacts/model")

                    else:
                        mlflow.sklearn.save_model(best_model, "artifacts/model")
                        mlflow.log_artifact("artifacts/model")



                if best_model_score<0.6:
                    raise CustomException("No Best Model Found")
                logging.info("Best model found on both training and test data")
                save_object(
                    file_path = self.model_trainer_config.trained_model_file_path,
                    obj = best_model
                )
                predicted = best_model.predict(X_test)
                r2_square = r2_score(y_test,predicted)
                return r2_square
            except Exception as e:
                raise CustomException(e,sys)