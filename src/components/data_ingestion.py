# Importing necessary libraries
import os  # For file path operations
import sys  # For system-specific parameters and functions
from src.exception import CustomException  # Custom exception handling
from src.logger import logging  # Custom logging for tracking events
import pandas as pd  # For working with data (read CSV, manipulate, etc.)

# For splitting data into training and testing sets
from sklearn.model_selection import train_test_split
from dataclasses import dataclass  # For creating simple data-holding classes

# Importing data transformation components from your project
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

# Configuration class using @dataclass decorator
# Holds paths where the ingested data will be saved
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")  # Path to save training data
    test_data_path: str = os.path.join('artifacts', "test.csv")    # Path to save test data
    raw_data_path: str = os.path.join('artifacts', "data.csv")     # Path to save original full data

# Class to handle the data ingestion process
class DataIngestion:
    def __init__(self):
        # Create an instance of the configuration class
        self.ingestion_config = DataIngestionConfig()

    # Main method to initiate the data ingestion process
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")  # Log start of ingestion
        try:
            # Read the CSV dataset from the specified path
            df = pd.read_csv(r'notebook\data\stud.csv')  # Use raw string to avoid backslash issue
            logging.info('Read the dataset as dataframe')  # Log success

            # Create directory if it doesn't exist to store processed files
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the full dataset as a raw backup
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")  # Log split step
            # Split the data into training and testing sets (80% train, 20% test)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save training set to the specified file
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            # Save test set to the specified file
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")  # Log completion

            # Return the file paths of the train and test sets
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        # If any error occurs, raise a custom exception with full traceback
        except Exception as e:
            raise CustomException(e, sys)

# Entry point of the script
if __name__ == "__main__":
    # Create an object of the DataIngestion class
    obj = DataIngestion()
    
    # Call the ingestion method to process and split the data
    train_data, test_data = obj.initiate_data_ingestion()

    # Create an object of the DataTransformation class
    data_transformation = DataTransformation()

    # Perform data transformation on the train and test data
    # It returns transformed training array, testing array, and preprocessor object (which we ignore here)
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
