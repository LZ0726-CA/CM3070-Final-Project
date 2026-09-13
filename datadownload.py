from pathlib import Path
import time
import yfinance as yf
import pandas as pd
import pandas_market_calendars as mc

# get market trading days
# reference:https://pandas-market-calendars.readthedocs.io
nyse = mc.get_calendar("NYSE")

# folder path
BASE_DIR = Path(__file__).resolve().parent
DOWNLOAD_FOLDER = BASE_DIR / "downloaded_data"
STOCK_DATA_FOLDER = DOWNLOAD_FOLDER / "stocks"
BENCHMARK_DATA_FOLDER = DOWNLOAD_FOLDER / "benchmarks"
# create folder to store stock and benchmark data
STOCK_DATA_FOLDER.mkdir(parents=True, exist_ok=True)
BENCHMARK_DATA_FOLDER.mkdir(parents=True, exist_ok=True)

# benchmark ticker and filename used for each index
BENCHMARK_CONFIGS = {
    "SP500": {"benchmark_ticker": "^GSPC", "benchmark_filename": "SP500_INDEX.csv"},
    "NASDAQ100": {"benchmark_ticker": "^NDX", "benchmark_filename": "NASDAQ100_INDEX.csv"},
    "DOWJONES": {"benchmark_ticker": "^DJI", "benchmark_filename": "DOWJONES_INDEX.csv"}
}
# create folder  to store data downloaded for model training
TRAINING_DATA_FOLDER = DOWNLOAD_FOLDER / "model_training_data"
TRAINING_DATA_FOLDER.mkdir(parents=True,exist_ok=True)

# check that both dates are valide
def validate_date_range(start_date, end_date):
    if not start_date or not end_date:
        raise ValueError("Both start date and end date are required.")

    start_timestamp = pd.Timestamp(start_date)
    end_timestamp = pd.Timestamp(end_date)

    if start_timestamp >= end_timestamp:
        raise ValueError("The start date must be earlier than the end date.")

    return start_timestamp, end_timestamp


# convert user input to match Yahoo Finance ticker format.
def normalize_ticker(ticker):
    ticker = ticker.upper().strip().replace(".", "-")

    if not ticker:
        raise ValueError("Ticker cannot be empty.")

    return ticker

# check if data already exist
def data_exist(file_path, start_date, end_date):
    try: 
        df = pd.read_csv(file_path)

        # some yfinance CSV files call the first date column "Price"
        # rename it so all files use the same "Date" column name
        if "Price" in df.columns:
            df = df.rename(columns={"Price": "Date"})

        if "Date" not in df.columns:
            raise ValueError("CSV file does not contain a Date column.")

        # remove extra yfinance header rows
        valid_rows = ((df["Date"] != "Date") & (df["Date"] != "Ticker") & (df["Date"] != "Price"))
        df = df[valid_rows].copy()

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        if df.empty:
            return False

        requested_start = pd.to_datetime(start_date)
        requested_end = pd.to_datetime(end_date)

        # get trading days within the requested period
        trading_days = nyse.valid_days(start_date=requested_start, end_date=requested_end)

        if len(trading_days) == 0:
            return False

        # first and last actual trading days
        requested_start = trading_days[0].tz_localize(None)
        requested_end = trading_days[-1].tz_localize(None)

        first_date = df["Date"].min()
        last_date = df["Date"].max()

        if first_date <= requested_start and last_date >= requested_end:
            return True

        else:
            return False
    
    except Exception:
        return False

    

# function to download market data 
def download_market_data(ticker, output_path, start_date, end_date):
    ticker = normalize_ticker(ticker)
    start_date, end_date = validate_date_range(start_date,end_date)
    if data_exist(output_path, start_date, end_date):
        return output_path

    # download data from yfinance 
    else: 
        try:
            df = yf.download(
                tickers=ticker,
                start=start_date,
                end=end_date + pd.Timedelta(days=1), # yf download is exclusive for end date
                interval="1d",
                auto_adjust=False,
                progress=False,
                actions=False,
                threads=False,
            )

        except Exception as error:
            raise RuntimeError(f"Download failed for {ticker}: {error}") from error

        if df.empty:
            raise ValueError(f"Can not locate data for {ticker} from {start_date} to {end_date}")

        # save data to folder
        output_path.parent.mkdir(parents=True,exist_ok=True,)
        df.to_csv(output_path)

        # small pause to reduce rate-limit problems
        time.sleep(0.5)
    # return location of the download data
    return output_path




# predicton data download section -----------------------------------------------------------------
# function to specifically download individual stock data
def stock_data_download(ticker, start_date, end_date) :
    ticker = normalize_ticker(ticker)
    # set download path
    output_path = (STOCK_DATA_FOLDER/ f"{ticker}.csv")
    # download
    return download_market_data(
        ticker=ticker, output_path=output_path,
        start_date = start_date, end_date = end_date
    )

# function to download data for selected benchmark 
def benchmark_data_download(selected_model, start_date, end_date):
    if selected_model not in BENCHMARK_CONFIGS:
        raise ValueError(f"Unknown model selection: {selected_model}")
    # get benchmark information
    config = BENCHMARK_CONFIGS[selected_model]
    benchmark_ticker = config["benchmark_ticker"]
    # set download path
    output_path = (BENCHMARK_DATA_FOLDER/ config["benchmark_filename"])
    # download
    return download_market_data(
        ticker=benchmark_ticker,
        output_path=output_path,
        start_date = start_date, end_date = end_date
    )

# download data for making prediction
def prediction_data(ticker, selected_model, start_date, end_date):
    stock_path = stock_data_download(ticker, start_date = start_date, end_date = end_date)
    benchmark_path = benchmark_data_download(selected_model, start_date = start_date, end_date = end_date)

    return stock_path, benchmark_path
# ----------------------------------------------------------------------------------------------------



# training data download section --------------------------------------------------------------------- 
# download benchmark data for training 
def training_benchmark_data_download(selected_index,start_date,end_date):
    if selected_index not in BENCHMARK_CONFIGS:
        raise ValueError(f"Unknown index selection: {selected_index}")
    # get benchmark info
    config = BENCHMARK_CONFIGS[selected_index]
    # set download path
    date_folder = f"{start_date}_{end_date}"
    output_path = (
        TRAINING_DATA_FOLDER
        / selected_index
        / date_folder
        / "benchmarks"
        / config["benchmark_filename"]
    )
    # download
    return download_market_data(
        ticker=config["benchmark_ticker"],
        output_path=output_path,
        start_date=start_date,
        end_date=end_date
    )

# function to download specific stock data for training
def training_stock_data_download(ticker,selected_index,start_date,end_date):
    ticker = normalize_ticker(ticker)
    # set download path
    date_folder = f"{start_date}_{end_date}"
    output_folder = (
        TRAINING_DATA_FOLDER
        / selected_index
        / date_folder
        / "stocks"
    )
    output_path = output_folder / f"{ticker}.csv"
    # download
    return download_market_data(
        ticker=ticker,
        output_path=output_path,
        start_date=start_date,
        end_date=end_date
    )

# function to download both benchmark and individual stock for training
def training_data_download(tickers,selected_index,start_date,end_date):
    if selected_index not in BENCHMARK_CONFIGS:
        raise ValueError(f"Unknown index selection: {selected_index}")

    # download benchmark data
    benchmark_path = training_benchmark_data_download(
        selected_index=selected_index,
        start_date=start_date,
        end_date=end_date
    )

    stock_paths = []
    failed_downloads = []
    # download data for each of the stocks in an index
    for ticker in tickers:
        try:
            stock_path = training_stock_data_download(
                ticker=ticker,
                selected_index=selected_index,
                start_date=start_date,
                end_date=end_date
            )

            stock_paths.append({
                "ticker": normalize_ticker(ticker),
                "path": stock_path
            })

        except Exception as error:
            failed_downloads.append({"ticker": ticker,"error": str(error)})

    if not stock_paths:
        raise ValueError("No stock data was downloaded successfully." )

    return {
        "stock_paths": stock_paths,
        "benchmark_path": benchmark_path,
    }
# --------------------------------------------------------------------------------------------------