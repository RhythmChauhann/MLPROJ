from src.mlproj.logger import logging
from src.mlproj.exception import CustomException
from src.mlproj.components.data_ingestion import DataIngestion
from src.mlproj.components.data_ingestion import DataIngestionConfig
from src.mlproj.components.data_transformation import DataTransformationConfig,DataTransformation
from src.mlproj.components.model_trainer import ModelTrainerConfig,ModelTrainer

import sys

if __name__ == "__main__":
    logging.info("Starting the ML Project Application")
    try:
        data_ingestion=DataIngestion()
        # # data_ingestion_config = DataIngestionConfig()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()

        data_transformation_config = DataTransformationConfig()
        data_transformation = DataTransformation()
        train_arr,test_arr,_ =data_transformation.initiate_data_transformation(train_data_path,test_data_path)
        ##model training
        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr,test_arr))
        

    except Exception as e:
        logging.info("An exception occurred")
        raise CustomException(e, sys)
