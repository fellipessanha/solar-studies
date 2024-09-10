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
    ) -> None:
        df = self._make_regressor_columns(df, target_col)
        super().__init__(
            df,
            train_threshold,
            test_threshold,
            target_col,
            idx_col,
            XGBRegressor(
                base_score=1.0,
                booster="gbtree",
                n_estimators=1000,
                early_stopping_rounds=500,
                objective="reg:squarederror",
                max_depth=3,
                learning_rate=0.1,
            ),
            error_calculator,
        )

        self._features = ["doy", "woy", *[c for c in df.columns if "lag" in c]]

    def split_x_y_xgb(self, df):
        return (
            df.loc[df[self._target].notna(), self._features],
            df.loc[df[self._target].notna(), self._target],
        )

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

    def _fit_model(self, df):
        train, test = split_datasets(df, self._train_threshold, self._test_threshold)
        x_train, y_train = self.split_x_y_xgb(train)
        x_test, y_test = self.split_x_y_xgb(test)
        self._regressor.fit(
            x_train,
            y_train,
            eval_set=[(x_train, y_train), (x_test, y_test)],
            verbose=100,
        )

    def _parse_forecast(self, forecast):
        prediction = pd.DataFrame(
            data=forecast, index=forecast.index, columns=["prediction"]
        )
        prediction = prediction.merge(
            self.test_df.loc[:, self._target],
            left_index=True,
            right_index=True,
        )

        return prediction

    def make_prediction(self, df):
        x_axis, _ = self.split_x_y_xgb(df)
        forecast = self._regressor.predict(x_axis)

        return pd.DataFrame(data=forecast, index=x_axis.index, columns=["prediction"])
