
import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(
        "artifacts", "preprocessor.pkl"
    )


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = [
                "writing_score",
                "reading_score"
            ]

            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            # Numerical preprocessing
            num_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            )

            # Categorical preprocessing
            cat_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),
                    (
                        "one_hot_encoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False
                        )
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            )

            logging.info(
                "Numerical columns: %s",
                numerical_columns
            )

            logging.info(
                "Categorical columns: %s",
                categorical_columns
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "num_pipeline",
                        num_pipeline,
                        numerical_columns
                    ),
                    (
                        "cat_pipeline",
                        cat_pipeline,
                        categorical_columns
                    )
                ]
            )

            logging.info("Preprocessing object created")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Data transformation initiated")

            # Read training and testing data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test datasets")

            target_column_name = "math_score"

            # Validate required columns
            required_columns = [
                "writing_score",
                "reading_score",
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
                target_column_name
            ]

            for dataset_name, dataset in [
                ("train", train_df),
                ("test", test_df)
            ]:
                missing_columns = [
                    col for col in required_columns
                    if col not in dataset.columns
                ]

                if missing_columns:
                    raise ValueError(
                        f"Missing columns in {dataset_name}: "
                        f"{missing_columns}"
                    )

            # Create preprocessing object
            preprocessing_obj = (
                self.get_data_transformer_object()
            )

            # Separate input and target features
            input_feature_train_df = train_df.drop(
                columns=[target_column_name]
            )

            target_feature_train_df = train_df[
                target_column_name
            ]

            input_feature_test_df = test_df.drop(
                columns=[target_column_name]
            )

            target_feature_test_df = test_df[
                target_column_name
            ]

            logging.info(
                "Applying preprocessing on training data"
            )

            # Fit only on training data
            input_feature_train_arr = (
                preprocessing_obj.fit_transform(
                    input_feature_train_df
                )
            )

            # Transform testing data
            input_feature_test_arr = (
                preprocessing_obj.transform(
                    input_feature_test_df
                )
            )

            # Convert sparse output if necessary
            if hasattr(input_feature_train_arr, "toarray"):
                input_feature_train_arr = (
                    input_feature_train_arr.toarray()
                )

            if hasattr(input_feature_test_arr, "toarray"):
                input_feature_test_arr = (
                    input_feature_test_arr.toarray()
                )

            # Combine transformed features with target
            train_arr = np.c_[
                input_feature_train_arr,
                target_feature_train_df.to_numpy()
            ]

            test_arr = np.c_[
                input_feature_test_arr,
                target_feature_test_df.to_numpy()
            ]

            # Save fitted preprocessing object
            logging.info("Saving preprocessing object")

            save_object(
                file_path=(
                    self.data_transformation_config
                    .preprocessor_obj_file_path
                ),
                obj=preprocessing_obj
            )

            logging.info(
                "Data transformation completed successfully"
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error("Data transformation failed: %s", e)
            raise CustomException(e, sys)
