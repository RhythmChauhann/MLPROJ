import os
import sys
from src.mlproj.exception import CustomException
from src.mlproj.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql


load_dotenv()

host = os.getenv('host')
user = os.getenv('user')
passwrd = os.getenv('password')
db = os.getenv('db')



def read_sql_data():
    logging.info("Reading mysql databases started")
    try:
        mydb = pymysql.connect(
            host=host,
            user = user,
            password = passwrd,
            db=db
        )
        logging.info(f"Connection established{mydb}")
        df=pd.read_sql_query("SELECT * FROM student",mydb)
        print(df.head())
        return df

    except Exception as e:
        raise CustomException(e,sys)
        
     