from xgboost import XGBRegressor
from src.forecasting.base import BaseForecaster
import pandas as pd
from src.train_test_data import split_datasets


class XGBForecaster(BaseForecaster):
    def __init__(
        self,
        df,
        train_threshold,
        test_threshold,
        target_col: str,
        idx_col: str,
        error_calculator,
        verbose=None,
    ) -> None:
        df = self._make_regressor_columns(df, target_col)

        self.verbose = verbose
        super().__init__(
            df,
            train_threshold,
            test_threshold,
            target_col,
            idx_col,
            XGBRegressor,
            error_calculator,
            hyperparams=dict(
                base_score=1.0,
                booster="gbtree",
                objective="reg:squarederror",
                max_depth=3,
                learning_rate=0.1,
            ),
        )

        self._features = ["doy", "woy", *[c for c in df.columns if "lag" in c]]

    def split_x_y_xgb(self, df):
        return (df.loc[:, self._features], df.loc[:, self._target])

    def _make_regressor_columns(self, df: pd.DataFrame, target: str):
        df = df.copy()
        df["lag1"] = (df.index - pd.DateOffset(years=1)).map(df[target].to_dict())
        df.index - pd.DateOffset(years=1)
        df["lag2"] = (df.index - pd.DateOffset(years=2)).map(df[target].to_dict())
        df["lag3"] = (df.index - pd.DateOffset(years=3)).map(df[target].to_dict())
        df["lag4"] = (df.index - pd.DateOffset(years=4)).map(df[target].to_dict())

        df["date"] = df.index

        df["doy"] = df["date"].apply(lambda d: d.day_of_year)
        df["woy"] = df["date"].apply(lambda d: d.date().isocalendar()[1])
        df["month"] = df["date"].apply(lambda d: d.month)

        return df

    def _fit_model_train(self, df: pd.DataFrame) -> None:
        train, test = split_datasets(df, self._train_threshold, self._test_threshold)
        x_train, y_train = self.split_x_y_xgb(train.loc[df[self._target].notna()])
        x_test, y_test = self.split_x_y_xgb(test.loc[df[self._target].notna()])
        self._regressor.fit(
            x_train,
            y_train,
            eval_set=[(x_train, y_train), (x_test, y_test)],
            verbose=self.verbose,
        )

    def _fit_model(self) -> None:
        df = self._df
        x, y = self.split_x_y_xgb(df.loc[df[self._target].notna()])
        self._regressor.fit(
            x,
            y,
            eval_set=[(x, y), (x, y)],
            verbose=self.verbose,
        )

    def _parse_forecast(self, forecast):
        return pd.DataFrame(data=forecast, index=forecast.index, columns=["prediction"])

    def _merge_score_dfs(self, prediction, test) -> pd.DataFrame:
        prediction = prediction.merge(
            test.loc[:, self._target],
            left_index=True,
            right_index=True,
        )

        return prediction

    def make_prediction(self, df):
        x_axis, _ = self.split_x_y_xgb(df)
        forecast = self._regressor.predict(x_axis)

        return pd.DataFrame(data=forecast, index=x_axis.index, columns=["prediction"])

    def make_future_prediction(
        self,
        df: pd.DataFrame,
        starting_on: pd.DatetimeIndex = None,
        duration_in_months: int = None,
    ) -> pd.DataFrame:
        future = self.make_future_dataframe(starting_on, duration_in_months)
        return self.make_prediction(future)

    def make_future_dataframe(
        self,
        starting_on: pd.DatetimeIndex = None,
        duration_in_months: int = None,
    ):
        df = self._df.copy()
        starting_on = starting_on or df.index.max()
        future = pd.date_range(
            df.index.max(), starting_on + pd.DateOffset(months=duration_in_months or 6)
        )
        future = pd.DataFrame(index=future)
        future["future"] = True
        df["future"] = False
        concat = pd.concat([df, future])

        return self._make_regressor_columns(concat, self._target).loc[
            concat["future"], [self._target, *self._features]
        ]
