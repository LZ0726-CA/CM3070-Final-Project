from datetime import date
import numpy as np
import pandas as pd
import requests
from flask import request
from features import (
    extract_stock_features,
    normalize_columns,
    load_benchmark_data,
    add_relative_features
)
from datadownload import (
    normalize_ticker,
    prediction_data
)
from models import load_selected_model
from training import get_index_tickers
import pandas_market_calendars as mc
# get market trading days
nyse = mc.get_calendar("NYSE")

# prepare stock and benchmark data for prediction
def prepare_prediction_data(ticker,benchmark,start_date, end_date):
    ticker = normalize_ticker(ticker)
    # get data path
    stock_path, benchmark_path = prediction_data(
        ticker=ticker,
        selected_model=benchmark,
        start_date=start_date,
        end_date=end_date
    )

    # data processing
    stock_df = pd.read_csv(stock_path)
    stock_df = normalize_columns(stock_df)
    stock_df = extract_stock_features(stock_df)
    # load benchmark data and merge data
    benchmark_df = load_benchmark_data(benchmark_path,include_target=False)
    prediction_df = stock_df.merge(benchmark_df,on="Date",how="inner")
    # calculate relative features 
    prediction_df = add_relative_features(prediction_df)
    prediction_df = prediction_df.replace([np.inf, -np.inf],np.nan)

    return (prediction_df.sort_values("Date").reset_index(drop=True))

# use local AI model to explain results
def AI_explain(ticker,model_name,predicted_class,probability,benchmark, threshold):
    # convert class to text
    if predicted_class == 1:
        result_text = "Outperform"
    else: 
        result_text = "Underperform"

    if probability is None:
        probability_text = "Error"
    else:
        probability_text = (f"{probability:.1%}")

    # prompt sent to the local AI model
    prompt = f"""
    you are explaining a stock prediction result to a beginner investor.

    Prediction information:
    - Stock: {ticker}
    - Model: {model_name}
    - Benchmark: {benchmark}
    - Prediction period: 100 trading days
    - Classification: {result_text}
    - Model score for outperforming: {probability_text}
    - Classification threshold: {threshold:.0%}

    Write an explanation of the prediction.

    The numbered instructions below describe what information to include.
    Do not copy the numbers, instruction titles, or wording into your response.
    Do not use headings, bullet points, numbered lists, or markdown formatting.
    Write 2 short paragraphs in normal prose.

    1. In the first paragraph, explain whether the model classified the stock as
    outperforming or not outperforming {benchmark} over the next 100 trading days.
    Explain that the model gave the stock an outperform score of {probability_text}
    You must explicitly explain that this does not mean there is a
    {probability_text} real-world chance that the stock will outperform.
    Explain that the score is produced by the Random Forest model based on
    patterns learned from historical data. It represents how strongly the
    model favors the "outperform" class, but it is not real-world probability.
    The classification is determined by comparing the model score with
    the threshold of {threshold:.0%}.
    If the score is at or above the threshold, the model classifies the
    stock as likely to outperform.
    If the score is below the threshold, it does not.
    Explain the Random Forest model learned relationships from historical stock
    prices, benchmark performance, and technical indicators.

    2. In the second paragraph, explain the main limitations. Historical patterns may
    not repeat, and the model does not consider information such as company news,
    earnings surprises, economic changes, interest-rate decisions, or unexpected
    market events. Make clear that the prediction is uncertain and should not be
    treated as financial advice. Do not give concrete financial advice. 
    State the prediction is not certain finacial gain.
    """
        
    # request response from local AI
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "glm-4.7-flash:latest",
            "prompt": prompt,
            "stream": False,
            "options": {
                # keep temperature low to avoid AI deviate too much from the script
                "temperature": 0.4
            }
        },
        # timeout if AI model over thinks 
        timeout=360,
    )
    # raise an error if request fails
    response.raise_for_status()
    # convert returned JSON response 
    data = response.json()
    # return the generated text
    return data.get("response","").strip()

# load the selected stock and model and use model to make prediction
def predict_stock(ticker,model_filename,start_date,end_date, prediction_threshold = 0.55):
    # select and load model information
    model_info = load_selected_model(model_filename)
    model = model_info["model"]
    model_name = model_info["model_name"]
    benchmark = model_info["benchmark"]
    expected_features = model_info["feature_names"]
    threshold = model_info.get("prediction_threshold",prediction_threshold)
    # prepare prediction data
    prediction_df = prepare_prediction_data(
        ticker=ticker,
        benchmark=benchmark,
        start_date=start_date,
        end_date=end_date,
    )
    # data with only selected feature
    feature_rows = prediction_df.dropna(subset=expected_features)

    if feature_rows.empty:
        raise ValueError(f"Features have unexpected error for {ticker}.")

    # use the latest complete row for prediction
    latest_row = feature_rows.iloc[[-1]]
    # select only the features expected by the saved model
    model_input = latest_row[expected_features]
    # make an initial class prediction
    predicted_class = int(model.predict(model_input)[0])

    probability = None

    # get the probability score 
    # RandomForestClassifier predict_proba reference:
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier
    if hasattr(model, "predict_proba"):
        class_probabilities = (model.predict_proba(model_input)[0])
        classes = list(model.classes_)
        positive_class_index = (classes.index(1))
        probability = float(class_probabilities[positive_class_index])
        predicted_class = int(probability >= threshold)

    return {
        "predicted_class": predicted_class,
        "probability": probability,
        "threshold": threshold,
        "prediction_date": (latest_row["Date"].iloc[0]),
        "model_name": model_name,
        "benchmark": benchmark,
        "model_filename": model_filename,
    }


# combine all prediction functions and run prediction
def run_prediction(page_data):
    # get ticker and model
    ticker = request.form.get("ticker","").upper().strip()
    selected_model_filename = request.form.get("prediction_model","").strip()
    # keep latest info on HTML
    page_data["ticker"] = ticker
    page_data["selected_model_id"] = (selected_model_filename)
    # check if ticker entered
    if not ticker:
        page_data["prediction_error"] = ("Please enter a stock ticker.")
        return
    # check if model selected
    if not selected_model_filename:
        page_data["prediction_error"] = ("Please select a model.")
        return

    # run prediction
    try:
        end_date = date.today()
        # get trading days
        trading_days = nyse.valid_days(start_date=end_date - pd.Timedelta(days=400),end_date=end_date)
        end_date = trading_days[-1].tz_localize(None)
        start_date = trading_days[-220].tz_localize(None) #use 220 days to give safety margin

        result = predict_stock(
            ticker=ticker,
            model_filename=(selected_model_filename),
            start_date=start_date,
            end_date=end_date,
        )

        predicted_class = result["predicted_class"]
        probability = result["probability"]
        model_name = result["model_name"]

        # send prediction results back to the HTML 
        page_data["selected_model"] = (model_name)
        page_data["prediction"] = (probability)
        page_data["prediction_date"] = (result["prediction_date"])
        page_data["prediction_threshold"] = result["threshold"]

        # display the prediction results
        if predicted_class == 1:
            page_data["predict_direction"] = ("Model suggests the stock may outperform in the next 100 trading days.")
        else:
            page_data["predict_direction"] = ("Model suggests the stock may underperform in the next 100 trading days.")

        # generate an AI explanation
        if probability is not None:
            try:
                page_data["ai_explanation"] = (
                    AI_explain(
                        ticker=ticker,
                        model_name=model_name,
                        predicted_class=(predicted_class),
                        probability=probability,
                        benchmark=result["benchmark"],
                        threshold=result["threshold"]
                    )
                )

            except Exception as ai_error:
                page_data["ai_explanation"] = (f"AI explanation was funavailable: {ai_error}")

    except Exception as error:
        page_data["prediction_error"] = (f"Prediction error: {error}")


# Scanning section------------------------------------------------------------------------------------------------
# scan through the entire index:
def scan_index(page_data):
    # get index name
    selected_index = request.form.get("scan_index","" ).strip()
    # get model name
    selected_model_filename = request.form.get("scan_model","").strip()

    # keep last selected options 
    page_data["selected_scan_index"] = selected_index
    page_data["selected_scan_model"] = selected_model_filename

    # error handling:
    if not selected_index:
        page_data["scan_error"] = "Please select an index."
        return

    if not selected_model_filename:
        page_data["scan_error"] = "Please select a model."
        return

    # get tickers 
    tickers = get_index_tickers(selected_index=selected_index)

    results = []
    # get dates
    end_date = date.today()
    trading_days = nyse.valid_days(start_date=end_date - pd.Timedelta(days=400),end_date=end_date)
    end_date = trading_days[-1].tz_localize(None)
    start_date = trading_days[-220].tz_localize(None) 
    # run scanning prediction
    for ticker in tickers:
        try:
            # get results
            result = predict_stock(
                ticker=ticker,
                model_filename=(selected_model_filename),
                start_date=start_date,
                end_date=end_date,
            )

            probability = result["probability"]
            predicted_class = result["predicted_class"]

            # only keep stocks classified as outperform
            if predicted_class == 1:
                results.append({"ticker": ticker, "probability": probability})
    
        except Exception as scan_error:
            print(f"Error: {scan_error}")

    results.sort(key=lambda x: x["probability"], reverse=True)
    # keep only upto 50 top results
    results = results[:50]
    # add rank number
    for rank, result in enumerate(results, start=1):
        result["rank"] = rank
    # indicate scan complete and if no results outperform then HTML will display message
    page_data["scan_completed"] = True

    page_data["scan_results"] = results
