import os
import sys
from sec.exception import CustomException
from sec.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', 'train.csv')
    test_data_path: str=os.path.join('artifacts', 'test.csv')
    raw_data_path: str=os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_congfig = DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('/Users/sashwotkoirala/Desktop/MDS/Prep/ML/mlProjects/notebook/dataVII.csv')
            df = df.drop(index=df.index[0]).reset_index(drop=True)
            logging.info('Read the dataset as dataframe')
            os.makedirs(self.ingestion_congfig.train_data_path)
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_congfig.raw_data_path, index=False, header=True)
            logging.info('Train Test Split initiated')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_congfig.train_data_path, index=False, header=True )
            test_set.to_csv(self.ingestion_congfig.test_data_path, index=False, header=True )
            logging.info('Initation of data is completed')
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_congfig.test_data_path
            )

        except Exception as e:
            pass

if __name__ == '__main__':
    obj=DataIngestion()
    obj.initiate_data_ingestion()


    

