import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  #sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from logger import logging
from exception import CustomException


from components.data_transformation import DataTransformation
from components.data_transformation import DataTransformationConfig
from components.model_trainer import ModelTrainer
from components.model_trainer import ModelTrainerConfig

import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path: str= os.path.join('artifacts', 'data.csv')
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv') 
# can be given in config entity file

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def intiate_data_ingestion(self):
        logging.info('Inside the Data Ingestion Component')
        try:
            df = pd.read_csv(r'notebook\data\stud.csv')
            logging.info('Read the data file as a DataFrame')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False,header=True)
            
            logging.info('Initiating train test split')
            train_set, test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False, header=True)

            logging.info('Data Ingestion has been completed')

            return (

                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e :
            raise CustomException(e,sys)


if __name__ =='__main__':
    obj = DataIngestion()
    train_path, test_path = obj.intiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr, _ = data_transformation.initiate_data_transformation(train_path,test_path)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))

    