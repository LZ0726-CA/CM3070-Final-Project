from datetime import datetime
import re
import joblib
import numpy as np
import pandas as pd
from flask import request
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score)
from io import StringIO
from datadownload import training_data_download

from features import (
    normalize_columns,
    extract_stock_features,
    load_benchmark_data,
    add_relative_features,
    validate_selected_features,
)

from models import (
    SAVED_MODEL_FOLDER,
    get_available_models
)


# convert the user-entered model name into a valide filename
def model_filename(model_name):
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", model_name.strip())
    cleaned = cleaned.strip("_")

    if not cleaned:
        raise ValueError("Model name cannot be empty.")

    return cleaned.lower()

# create the training target 
def add_training_target(df, horizon=100):
    df = df.copy()
    # calculate stock return over the next 100 trading days
    df["stock_future_return"] = (df["Close"].shift(-horizon)/ df["Close"]- 1)
    # calculate benchmark return over the next 100 trading days
    df["benchmark_future_return"] = (df["benchmark_close"].shift(-horizon)/ df["benchmark_close"]- 1)
    # calculate future stock return relative to the benchmark
    df["future_excess_return"] = (df["stock_future_return"]- df["benchmark_future_return"])
    # initialize target row
    df["target_outperform"] = np.nan
    # outperformers
    df.loc[df["future_excess_return"] > 0,"target_outperform"] = 1
    # underperformers
    df.loc[df["future_excess_return"] <= 0,"target_outperform"] = 0

    return df

# prepare one stock's data for model training
def prepare_training_stock(stock_path,benchmark_path,selected_features,horizon=100):
    # validate the user-selected feature list
    selected_features = validate_selected_features(selected_features)
    # load and calculate stock features
    stock_df = pd.read_csv(stock_path)
    stock_df = normalize_columns(stock_df)
    stock_df = extract_stock_features(stock_df)
    # load benchmark 
    benchmark_features = load_benchmark_data(benchmark_path,include_target=False)
    benchmark_prices = pd.read_csv(benchmark_path)
    benchmark_prices = normalize_columns(benchmark_prices)
    benchmark_prices = benchmark_prices[["Date", "Close"]].rename(columns={"Close": "benchmark_close"})

    # add benchmark features
    df = stock_df.merge(benchmark_features,on="Date",how="inner")

    # add benchmark closing price for target 
    df = df.merge(benchmark_prices,on="Date",how="inner")

    df = add_relative_features(df)
    df = add_training_target(df,horizon=horizon)

    # keep selected features, and target by date
    required_columns = (["Date"]+ selected_features+ ["target_outperform"])
    # remove invalide data and sort by date
    prepared_df = df[required_columns].replace([np.inf, -np.inf], np.nan)
    prepared_df = prepared_df.dropna().sort_values("Date").reset_index(drop=True)

    return prepared_df
        
    
# split data chronologically into training, validation and test sets
def train_validate_test_split(df,train_fraction=0.70,validation_fraction=0.15,gap=100):
    # sort rows by date
    df = (df.copy().sort_values("Date").reset_index(drop=True))
    # use unique dates so all stocks from the same date are bundled together
    unique_dates = pd.Series(sorted(df["Date"].dropna().unique()))
    total_dates = len(unique_dates)

    # calculate split positions and gap to ensure data do not bleed into other datasets
    train_end_position = int(total_dates * train_fraction)
    validation_start_position = (train_end_position + gap)
    validation_end_position = int(total_dates* (train_fraction + validation_fraction))
    test_start_position = (validation_end_position + gap)

    # check that enough data exists for all three splits
    if (train_end_position <= 0
        or validation_start_position>= validation_end_position
        or test_start_position >= total_dates):
        raise ValueError("The stock data is too short.")

    # convert split positions into actual dates
    train_end_date = unique_dates.iloc[train_end_position - 1]
    validation_start_date = unique_dates.iloc[validation_start_position]
    validation_end_date = unique_dates.iloc[validation_end_position - 1]
    test_start_date = unique_dates.iloc[test_start_position]

    # create training, validation and test sets
    train_df = df[df["Date"] <= train_end_date].copy()
    validation_df = df[(df["Date"] >= validation_start_date)& (df["Date"] <= validation_end_date)].copy()
    test_df = df[df["Date"] >= test_start_date].copy()

    if (train_df.empty or validation_df.empty or test_df.empty):
        raise ValueError("Error! One or more data splits results empty dataframe.")

    return train_df, validation_df, test_df

# train, evaluate and save the model
def train_random_forest(training_df,selected_features,model_name,benchmark,start_date,end_date,horizon=100):
    # get selected features
    selected_features = validate_selected_features(selected_features)
    # prepare data
    train_df, validation_df, test_df = (train_validate_test_split(training_df,gap=horizon))
    # select target
    target_column = "target_outperform"

    # separate feature inputs and target outputs
    X_train = train_df[selected_features]
    y_train = train_df[target_column]

    X_validation = validation_df[selected_features]
    y_validation = validation_df[target_column]

    X_test = test_df[selected_features]
    y_test = test_df[target_column]

    # check if the training data contains 2 target classes
    if y_train.nunique() < 2:
        raise ValueError("Error! The training set contains only one target class.")

    # create the Random Forest classifier
    # RandomForestClassifier reference:
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # train model
    model.fit(X_train,y_train)
    # set threshold
    evaluation_threshold = 0.55

    # get probability
    validation_probabilities = (model.predict_proba(X_validation)[:, 1])
    test_probabilities = (model.predict_proba(X_test)[:, 1])

    # apply the threshold
    validation_predictions = (validation_probabilities >= evaluation_threshold).astype(int)
    test_predictions = (test_probabilities >= evaluation_threshold).astype(int)

    # calculate validation and test performance metrics
    metrics = {
        "validation_accuracy": float(accuracy_score(y_validation,validation_predictions)),
        "validation_precision": float(precision_score(y_validation,validation_predictions,zero_division=0)),
        "validation_recall": float(recall_score(y_validation,validation_predictions,zero_division=0)),
        "validation_f1": float(f1_score( y_validation,validation_predictions,zero_division=0)),
        "test_accuracy": float(accuracy_score(y_test,test_predictions)),
        "test_precision": float(precision_score(y_test,test_predictions,zero_division=0)),
        "test_recall": float(recall_score(y_test,test_predictions,zero_division=0)),
        "test_f1": float(f1_score(y_test,test_predictions,zero_division=0)),
        "evaluation_threshold": evaluation_threshold
    }

    # create a filename and save the trained model
    filename = (model_filename(model_name)+ ".joblib")
    model_path = (SAVED_MODEL_FOLDER/ filename)
    bundle = {
        "model": model,
        "model_name": model_name.strip(),
        "benchmark": benchmark,
        "feature_names": selected_features,
        "target": target_column,
        "forecast_horizon": horizon,
        "prediction_threshold": evaluation_threshold,
        "training_start_date": str(start_date),
        "training_end_date": str(end_date),
        "metrics": metrics,
        "created_at": (datetime.now().isoformat(timespec="seconds")),
    }
    joblib.dump(bundle,model_path)

    return bundle, model_path

# get the current stock tickers in the selected index
# Code reference:
# https://requests.readthedocs.io/en/latest/user/quickstart/
# https://pandas.pydata.org/docs/reference/api/pandas.read_html
def get_index_tickers(selected_index):
    if selected_index == "SP500":
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0"}

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        tables = pd.read_html(StringIO(response.text))
        sp500_table = tables[0]

        if sp500_table is None:
            raise ValueError("Could not find the SP500 components table")

        tickers = sp500_table["Symbol"].tolist()
        tickers = [str(ticker).strip() for ticker in tickers]
        tickers = [ticker.replace(".", "-") for ticker in tickers] 
                  
    elif selected_index == "NASDAQ100":
        url = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"
        headers = {"User-Agent": "Mozilla/5.0"}

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        tables = pd.read_html(StringIO(response.text))
        nasdaq100_table = tables[0]

        if nasdaq100_table is None:
            raise ValueError("Could not find the Nasdaq components table")

        tickers = nasdaq100_table["Ticker"].tolist()
        tickers = [str(ticker).strip() for ticker in tickers]
        tickers = [ticker.replace(".", "-") for ticker in tickers]
               
    elif selected_index == "DOWJONES":
        url = "https://en.wikipedia.org/wiki/List_of_Dow_Jones_Industrial_Average_companies"
        headers = {"User-Agent": "Mozilla/5.0"}

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        tables = pd.read_html(StringIO(response.text))
        dow_table = None

        for table in tables:
            table.columns = [str(column).strip() for column in table.columns]

            if "Company" in table.columns and "Symbol" in table.columns:
                dow_table = table
                break

        if dow_table is None:
            raise ValueError("Could not find the Dow Jones components table")

        tickers = dow_table["Symbol"].tolist()
        tickers = [str(ticker).strip() for ticker in tickers]
        tickers = [ticker.replace(".", "-") for ticker in tickers]

    else:
        raise ValueError("Invalid index selection!")

    return tickers


# combine all training functions and run model training
def run_training(page_data):
    # get training inputs from the HTML 
    model_name = request.form.get("new_model_name","").strip()
    selected_index = request.form.get("training_index","" ).strip()
    start_date = request.form.get("training_start_date","").strip()
    end_date = request.form.get("training_end_date","").strip()
    selected_features = request.form.getlist("selected_features")

    # keep the submitted values in the page.
    page_data["new_model_name"] = model_name
    page_data["selected_index"] = selected_index
    page_data["training_start_date"] = start_date
    page_data["training_end_date"] = end_date
    page_data["selected_features"] = selected_features

    # validate input and start to download
    try:
        if not model_name:
            raise ValueError("Please enter a model name.")

        if selected_index not in {
            "SP500",
            "NASDAQ100",
            "DOWJONES",
        }:
            raise ValueError("Please select a valid index.")

        if not start_date or not end_date:
            raise ValueError("Please select both training dates.")

        if pd.Timestamp(start_date) >= pd.Timestamp(end_date):
            raise ValueError("The training start date must be earlier than the end date.")

        selected_features = ( validate_selected_features(selected_features))
        page_data["selected_features"] = (selected_features)

        # download training data
        tickers = get_index_tickers(selected_index=selected_index)
        download_result = training_data_download(tickers=tickers,selected_index=selected_index,start_date=start_date,end_date=end_date)

        stock_paths = download_result["stock_paths" ]
        benchmark_path = download_result["benchmark_path"]

        if not stock_paths:
            raise ValueError("No stock data was downloaded.")

        # prepare each downloaded stock for training
        prepared_stock_data = []
        preparation_failures = []

        for stock_item in stock_paths:
            ticker = stock_item["ticker"]
            stock_path = stock_item["path"]

            try:
                stock_training_df = prepare_training_stock(stock_path=stock_path,benchmark_path=benchmark_path,
                                                           selected_features=selected_features,horizon=100)
                prepared_stock_data.append(stock_training_df)

            except Exception as error:
                preparation_failures.append({"ticker": ticker,"error": str(error)})

        if not prepared_stock_data:
            raise ValueError("No downloaded stocks!" )

        # combine all stocks dataset
        combined_training_df = pd.concat(prepared_stock_data,ignore_index=True)
        combined_training_df = combined_training_df.sort_values("Date").reset_index(drop=True)

        # train, evaluate and save the model
        model_bundle, model_path = (
            train_random_forest(
                training_df=combined_training_df,selected_features=(selected_features),
                model_name=model_name,benchmark=selected_index,
                start_date=start_date,end_date=end_date,horizon=100)
        )

        # send results to the HTML page
        page_data["training_metrics"] = (model_bundle["metrics"])
        page_data["saved_model_filename"] = (model_path.name)
        page_data["training_message"] = (f"Model '{model_bundle['model_name']}' was trained and saved as {model_path.name}.")

        # refresh dropdown so the new model appears 
        page_data["available_models"] = (get_available_models())

    except Exception as error:
        page_data["training_error"] = str(error)

