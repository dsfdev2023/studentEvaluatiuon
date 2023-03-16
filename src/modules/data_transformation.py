import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('data', "proprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def dataTransformation_obj(self):
        '''
        This function will create data transformation object

        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())

                ]
            )

            cat_pipeline = Pipeline(

                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]

            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns)

                ]


            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, validation_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            validation_df = pd.read_csv(validation_path)

            logging.info("Completed...Reading train and test data from source")

            preprocessing_obj = self.dataTransformation_obj()

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            train_input_features = train_df.drop(
                columns=[target_column_name], axis=1)
            train_target_input_data = train_df[target_column_name]

            validation_input_features = train_df.drop(
                columns=[target_column_name], axis=1)
            validation_target_input_data = train_df[target_column_name]

            test_input_features = test_df.drop(
                columns=[target_column_name], axis=1)
            test_target_input_data = test_df[target_column_name]

            logging.info(
                f"Invoking Data Transformation object on the train and test  dataframes"
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(
                train_input_features)
            input_feature_validation_arr = preprocessing_obj.transform(
                validation_input_features)
            input_feature_test_arr = preprocessing_obj.transform(
                test_input_features)

            train_arr = np.c_[
                input_feature_train_arr, np.array(train_target_input_data)]
            
            validation_arr = np.c_[input_feature_validation_arr, np.array(
                validation_target_input_data)]
            test_arr = np.c_[input_feature_test_arr,
                             np.array(test_target_input_data)]

            logging.info(
                f"Completed...Data Transformation & Saved Preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                validation_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
