# Standard library imports
import sys
from dataclasses import dataclass

# Data and preprocessing libraries
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer  # For combining pipelines for different column types
from sklearn.impute import SimpleImputer  # For handling missing values
from sklearn.pipeline import Pipeline  # To chain multiple preprocessing steps
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # Encoding + Scaling

# Project-specific imports
from src.exception import CustomException  # Custom exception class
from src.logger import logging  # Custom logging module
import os
from src.utils import save_object  # Utility to save Python objects (like pickled models)

# Configuration class for Data Transformation
@dataclass
class DataTransformationConfig:
    # File path where the preprocessor object will be saved
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

# Main class for handling data transformation
class DataTransformation:
    def __init__(self):
        # Initialize the config for this component
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function creates and returns a preprocessing pipeline
        for both numerical and categorical features.
        '''
        try:
            # Define the list of numerical columns
            numerical_columns = ["writing_score", "reading_score"]

            # Define the list of categorical columns
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Pipeline for numerical features:
            # 1. Impute missing values with median
            # 2. Standardize using StandardScaler
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Pipeline for categorical features:
            # 1. Impute missing values with the most frequent category
            # 2. One-hot encode the categories
            # 3. Scale the resulting encoded features
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))  # Avoid centering sparse matrices
                ]
            )

            # Log the column groups being transformed
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combine both pipelines using ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            # Raise custom exception if anything goes wrong
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        '''
        This method performs:
        - Reading data
        - Splitting into input/output
        - Applying preprocessing
        - Saving preprocessing object
        - Returning transformed arrays
        '''
        try:
            # Load train and test datasets from file paths
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            # Get the preprocessor pipeline
            preprocessing_obj = self.get_data_transformer_object()

            # Define target column
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Separate input features and target for training data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Separate input features and target for testing data
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            # Apply transformations: fit on training data, transform both train and test
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine transformed features with their target values
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            # Save the preprocessor pipeline for future use (e.g., in prediction)
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Return the transformed arrays and path to the saved preprocessor
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            # If any error occurs during transformation, raise custom exception
            raise CustomException(e, sys)
