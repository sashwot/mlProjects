import os
import sys
from exception import CustomException
from logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.artifacts_dir = "artifacts"
        self.train_data_path = os.path.join(self.artifacts_dir, "train.csv")
        self.test_data_path = os.path.join(self.artifacts_dir, "test.csv")
        self.raw_data_path = os.path.join(self.artifacts_dir, "raw.csv")

class DataIngestion:
    def __init__(self):
         self.ingestion_config = DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('/Users/sashwotkoirala/Desktop/MDS/Prep/ML/mlProjects/notebook/dataVII.csv')
            df = df.drop(index=df.index[0]).reset_index(drop=True)  # Dropping the first row (assuming it's unwanted)
            logging.info("Dataset read successfully.")

            # Create the "artifacts" directory if it doesn’t exist
            os.makedirs(self.ingestion_config.artifacts_dir, exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw dataset saved at {self.ingestion_config.raw_data_path}")

            # Split dataset
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Train-test split completed.")

            # Save train and test data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info(f"Train dataset saved at {self.ingestion_config.train_data_path}")
            logging.info(f"Test dataset saved at {self.ingestion_config.test_data_path}")

            logging.info("Data ingestion completed successfully.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            pass 

if __name__ == '__main__':
    obj=DataIngestion()
    obj.initiate_data_ingestion()
    


    

