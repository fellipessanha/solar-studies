import pandas as pd
from src.forecasting.base import BaseForecaster
from src.train_test_data import testing_treshold
from prophet import Prophet


class ProphetForecaster(BaseForecaster):
    def __init__(
        self,
        df: pd.DataFrame,
        train_threshold: pd.DatetimeIndex,
        test_threshold: pd.DatetimeIndex,
        target_col: str,
        idx_col: str,
        error_calculator,
        *args
    ) -> None:
        super().__init__(
            df,
            train_threshold,
            test_threshold,
            target_col,
            idx_col,
            Prophet,
            error_calculator,
        )

    def _fit_model_train(self, df):
        fit_data = pd.DataFrame(
            dict(ds=self.train_df.index, y=self.train_df[self._target])
        )
        self._regressor.fit(fit_data)

    def _fit_model(self, df: pd.DataFrame) -> None:
        self._reset_model()
        fit_data = pd.DataFrame(dict(ds=df.index, y=df[self._target]))
        self._regressor.fit(fit_data)
        ...

    def _parse_forecast(self, forecast):
        prediction = (
            forecast.loc[
                testing_treshold(
                    forecast.ds, self._train_threshold, self._test_threshold
                ),
                ["ds", "yhat"],
            ]
            .rename(columns={"ds": self._idx, "yhat": "prediction"})
            .set_index("datetime")
        )
        return prediction.merge(self.test_df[self._target], on=self._idx)

    def make_prediction(self, df):
        future = self._regressor.make_future_dataframe(df.size)
        return self._regressor.predict(future)
