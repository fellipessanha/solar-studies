from typing import Dict, List
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
        self.historical_data = group_data_by_day_of_year(self.train_df, idx_col)
        self.historical_score = self._rate_historical_mean(error_calculator)

    def _rate_forecasts(
        self, error_calculator, df_params, forecasts: List[BaseForecaster]
    ):
        self._forecast_scores: Dict[BaseForecaster, float] = {
            i: i.evaluate_model()
            for i in [
                f(*df_params, error_calculator=error_calculator) for f in forecasts
            ]
        }

        return min(self._forecast_scores, key=self._forecast_scores.get)

    def _rate_historical_mean(self, error_calculator):
        df = (
            self.test_df[[self._target]].rename(columns={self._target: "target"}).copy()
        )
        df["doy"] = df.index.day_of_year
        df = df.merge(self.historical_data[self._target], on="doy").rename(
            columns={self._target: "historical"}
        )
        return error_calculator(df["target"], df["historical"])

    def compare_predictions(self):
        plot_pred = self.optimum_forecast.prediction.rename(
            columns={
                self._target: "target daily measurements",
                "prediction": f"{type(self.optimum_forecast).__name__} - error = {self._forecast_scores[self.optimum_forecast]:.4f}",
            }
        )
        for fc in self._forecast_scores:
            if fc == self.optimum_forecast:
                continue
            plot_pred = plot_pred.merge(
                fc.prediction["prediction"], on=self._idx
            ).rename(
                columns={
                    "prediction": f"{type(fc).__name__} - error = {self._forecast_scores[fc]:.4f}"
                }
            )

        plot_pred["doy"] = plot_pred.index.day_of_year
        return (
            plot_pred.merge(self.historical_data[self._target], on="doy")
            .rename(
                columns={
                    self._target: f"historical mean - error = {self.historical_score:.4f}"
                }
            )
            .drop(columns="doy")
            .copy()
        )

    def make_prediction(
        self, starting_on: pd.DatetimeIndex = None, duration_in_months: int = None
    ):
        fc = self.optimum_forecast
        fc._fit_model()
        return fc.make_future_prediction(starting_on, duration_in_months)
