from src.mlproj.logger import logging
from src.mlproj.exception import CustomException
from src.mlproj.components.data_ingestion import DataIngestion
from src.mlproj.components.data_ingestion import DataIngestionConfig
import sys

if __name__ == "__main__":
    logging.info("Starting the ML Project Application")
    try:
        data_ingestion=DataIngestion()
        # data_ingestion_config = DataIngestionConfig()
        data_ingestion.initiate_data_ingestion()
    except Exception as e:
        logging.info("An exception occurred")
        raise CustomException(e, sys)
