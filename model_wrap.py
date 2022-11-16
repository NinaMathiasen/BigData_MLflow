# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import mlflow
import mlflow.sklearn


# Define transformations

dict_direction_degree = {"N" : 0, 
                        "NNE" : 22.5, 
                        "NE": 45, 
                        "ENE" : 67.5, 
                        "E": 90,
                        "ESE" : 112.5,
                        "SE" : 135,
                        "SSE" : 157.5,
                        "S" : 180,
                        "SSW" : 202.5,
                        "SW" : 225,
                        "WSW" : 247.5,
                        "W" : 270,
                        "WNW" : 292.5,
                        "NW" : 315,
                        "NNW" : 337.5}

class DataframeFunctionTransformer():
    def __init__(self, func):
        self.func = func

    def transform(self, input_df, **transform_params):
        return self.func(input_df)

    def fit(self, X, y=None, **fit_params):
        return self

def remove_cols(df):
    df.drop(['Lead_hours', 'Source_time', 'ANM', 'Non-ANM'], axis=1, inplace = True)
    return df

def remove_rows_with_missing_values(df):
    df.dropna(inplace = True)
    return df

def delete_rows_with_negative_values(df):
    df.drop(df[df.Total < 0].index, inplace = True)
    df.drop(df[df.Speed < 0].index, inplace = True)

    return df

def direction_to_degree(df):
    df['Direction'] = df['Direction'].map(dict_direction_degree)
    return df

def direction_and_speed_to_vector(df):
    speed = df.pop('Speed')
    radians = df.pop('Direction')*np.pi / 180

    # Calculate the wind x and y components.
    df['wind_x'] = speed*np.cos(radians)
    df['wind_y'] = speed*np.sin(radians)

    return df

def timestamp_to_continuous_variables(df):
    # Getting the timestamp (in seconds)
    timestamp = df.index.to_series()

    dayofyear = timestamp.dt.dayofyear
    hour = timestamp.dt.hour
    minute = timestamp.dt.minute

    # Define constants
    day_constant = 24*60*60
    year_constant = (365.2425)

    # Calculate daily and yearly periodicity
    df['day_sin'] = np.sin((minute*60 + hour*60*60)/ day_constant * (2 * np.pi))
    df['day_cos'] = np.cos((minute*60 + hour*60*60)/ day_constant * (2 * np.pi))
    df['year_sin'] = np.sin((dayofyear)/year_constant * (2 * np.pi))
    df['year_cos'] = np.cos((dayofyear)/year_constant * (2 * np.pi))
    return df


# Define evaluation function


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

# function to wrap the pipeline to be logged by mlflow
class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
  def __init__(self, model):
    self.model = model
    
  def predict(self, context, model_input):
    return self.model.predict(model_input)[:,1]
  

# Program to run
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Import data 

    data = pd.read_json("dataset.json", orient = 'split')

    # Trandform data using the transformation pipeline
    data_transformation_pipeline = Pipeline([
        ('delete_cols', DataframeFunctionTransformer(remove_cols)),
        ("delete_nans", DataframeFunctionTransformer(remove_rows_with_missing_values)),
        ("delete_negatives", DataframeFunctionTransformer(delete_rows_with_negative_values)),
        ("direction_to_degree", DataframeFunctionTransformer(direction_to_degree)),
        ("direction_and_speed_to_vector", DataframeFunctionTransformer(direction_and_speed_to_vector)),
        ("timestamp_to_continuous_variables", DataframeFunctionTransformer(timestamp_to_continuous_variables))
    ])

    # apply the pipeline to the input dataframe
    data_transformed = data_transformation_pipeline.fit_transform(data)

    # Split the data into X and y:

    y = data_transformed.copy().Total
    X = data_transformed.copy().drop('Total', axis=1)

    # Split into test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = False, test_size=0.2, random_state=42)

    # Set model parameters
    param_n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    param_max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    param_min_samples_split = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    param_min_samples_leaf = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    with mlflow.start_run():
        RF_model = RandomForestRegressor(n_estimators=param_n_estimators, max_depth=param_max_depth, min_samples_split=param_min_samples_split, min_samples_leaf=param_min_samples_leaf, random_state=42)
        model_pipeline = Pipeline([('scaler', StandardScaler()), ('random_forest', RF_model)])
        model_pipeline.fit(X_train, y_train)
        y_pred = model_pipeline.predict(X_test)


        (rmse, mae, r2) = eval_metrics(y_test, y_pred)

        print("Randomforest model (n_estimators=%f, max_depth=%f, min_samples_split=%f, min_samples_leaf=%f):" % (param_n_estimators, param_max_depth, param_min_samples_split, param_min_samples_leaf))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("n_estimators", param_n_estimators)
        mlflow.log_param("max_depth", param_max_depth)
        mlflow.log_param("min_samples_split", param_min_samples_split)
        mlflow.log_param("min_samples_leaf", param_min_samples_leaf)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        #wrap the model for mlflow log
        wrappedModel = SklearnModelWrapper(model_pipeline)

    
        print("Saving model")

        mlflow.pyfunc.save_model("saved_model", python_model=wrappedModel, conda_env="model_mlflow.yaml")
