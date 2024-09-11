from typing import List
import pandas as pd
from src.forecasting.base import BaseForecaster
from src.forecasting.prophet import ProphetForecaster
from src.forecasting.xgb import XGBForecaster
from src.train_test_data import split_datasets
from src.historical_mean import group_data_by_day_of_year


class ForecastPicker:

    def __init__(
        self,
        df: pd.DataFrame,
        train_threshold: pd.DatetimeIndex,
        test_threshold: pd.DatetimeIndex,
        target_col: str,
        idx_col: str,
        error_calculator,
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
        df: pd.DataFrame

        df_params = [
            df,
            train_threshold,
            test_threshold,
            target_col,
            idx_col,
        ]

        self.optimum_forecast: BaseForecaster = self._rate_forecasts(
            error_calculator, df_params, [XGBForecaster, ProphetForecaster]
        )

        self.historica_data = group_data_by_day_of_year(self.train_df, idx_col)

    def _rate_forecasts(
        self, error_calculator, df_params, forecasts: List[BaseForecaster]
    ):
        self._forecast_scores = {
            i: i.evaluate_model()
            for i in [
                f(*df_params, error_calculator=error_calculator) for f in forecasts
            ]
        }

        return min(self._forecast_scores, key=self._forecast_scores.get)

    def _make_prediction(self):
        fc = self.optimum_forecast
        fc._fit_model()
        future = fc.make_future_dataframe()
        return fc.make_prediction(future)
