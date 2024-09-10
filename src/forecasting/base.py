from typing import TYPE_CHECKING, Tuple
from src.train_test_data import split_datasets
import numpy as np
import pandas as pd


class BaseForecaster:
    def __init__(
        self,
        df: pd.DataFrame,
        train_threshold: pd.DatetimeIndex,
        test_threshold: pd.DatetimeIndex,
        target_col: str,
        idx_col: str,
        regressor_initializer,
        error_calculator,
        hyperparams=None,
    ) -> None:
        self._df = df
        self._train_threshold = train_threshold
        self._test_threshold = test_threshold
        self.train_df, self.test_df = split_datasets(
            df, train_threshold, test_threshold
        )
        self._target = target_col
        self._idx = idx_col
        self._error_calculator = error_calculator
        self._regressor_initializer = regressor_initializer
        self.hyperparams = hyperparams
        self._reset_model()

    def _reset_model(self):
        self._regressor = self._regressor_initializer(**(self.hyperparams or {}))

    def _fit_model_train(self, df: pd.DataFrame) -> None: ...
    def _fit_model(self, df: pd.DataFrame) -> None: ...
    def _parse_forecast(self, forecast) -> pd.DataFrame:
        """
        should take the model forecast return an return a pandas dataframe with columns
        ['datetime', 'prediction', target], indexed by 'datetime'
        """
        ...

    def _calculate_score(self, prediction):
        return self._error_calculator(prediction[self._target], prediction.prediction)

    def make_prediction(self, df: pd.DataFrame) -> pd.DataFrame: ...

    def evaluate_model(self) -> np.float64:
        self._fit_model_train(self._df)
        forecast = self.make_prediction(self.test_df)
        self.prediction = self._parse_forecast(forecast)
        return self._calculate_score(self.prediction)
