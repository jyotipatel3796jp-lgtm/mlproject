
import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join(
        "artifacts", "model.pkl"
    )


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Starting model training")

            # Split features and target
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            # Define regression models
            models = {
                "Random Forest": RandomForestRegressor(
                    random_state=42
                ),
                "Decision Tree": DecisionTreeRegressor(
                    random_state=42
                ),
                "Gradient Boosting": GradientBoostingRegressor(
                    random_state=42
                ),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(
                    random_state=42
                ),
                "CatBoosting Regressor": CatBoostRegressor(
                    verbose=False,
                    random_seed=42
                ),
                "AdaBoost Regressor": AdaBoostRegressor(
                    random_state=42
                ),
            }

            # Train and evaluate all models
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models
            )

            if not model_report:
                raise ValueError(
                    "Model evaluation returned no results"
                )

            # Select model with highest test R2 score
            best_model_name = max(
                model_report,
                key=model_report.get
            )

            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            logging.info(
                f"Best model: {best_model_name}"
            )

            logging.info(
                f"Best model R2 score: {best_model_score:.4f}"
            )

            # Validate model performance
            if best_model_score < 0.6:
                raise ValueError(
                    "No model achieved the minimum R2 score of 0.6"
                )

            # Save the best trained model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info(
                "Best trained model saved successfully"
            )

            # Generate predictions
            predicted = best_model.predict(X_test)

            # Calculate final R2 score
            r2_square = r2_score(y_test, predicted)

            logging.info(
                f"Final R2 score: {r2_square:.4f}"
            )

            return r2_square

        except Exception as e:
            logging.error(
                f"Error during model training: {str(e)}"
            )
            raise CustomException(e, sys) from e
