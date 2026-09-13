import pandas as pd
import numpy as np

# clean the CSV and normalize the columns
def normalize_columns(df):
    df = df.copy()

    # some yfinance CSV files call the first date column "Price"
    # rename it so all files use the same "Date" column name
    if "Price" in df.columns:
        df = df.rename(columns={"Price": "Date"})

    if "Date" not in df.columns:
        raise ValueError("CSV file does not contain a Date column.")

    # remove extra yfinance header rows
    valide_rows = (df["Date"] != "Date") & (df["Date"] != "Ticker") & (df["Date"] != "Price")
    df = df[valide_rows].copy()

    # convert the date column into pandas datetime objects, invalid  values becomes NaT
    df["Date"] = pd.to_datetime(df["Date"],errors="coerce")

    # convert market-data columns to numbers, invalid  values becomes NaN
    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume"
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column],errors="coerce")

    # columns needed by stock and benchmark feature engineering
    required_columns = [
        "Date",
        "High",
        "Low",
        "Close",
        "Volume"
    ]
    # check if required columns are missing
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError("Missing required CSV columns: " + ", ".join(missing_columns))

    # remove rows with invalid data
    df = df.dropna(subset=required_columns)

    # sort by date and one row per date
    df = (df.sort_values("Date").drop_duplicates(subset="Date",keep="last"))
    df = df.reset_index(drop=True)

    return df

# process columns to get technical indicators
# calculate historical stock features used by the random-forest models
# formular reference:
# https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html
def extract_stock_features(df):
    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    daily_return = close.pct_change()
    high_low_ratio = high / low - 1

    # historical stock returns
    for window in [5, 10, 20, 30, 60, 120, 200]:
        df[f"return_{window}d"] = close.pct_change(window)

    # moving-average ratios, 
    # simple moving average (sma) is used
    for window in [5, 10, 20, 50, 100, 200]:
        moving_average = close.rolling(window).mean()
        df[f"sma{window}_ratio"] = (close / moving_average - 1)

    
    # relative volume
    for window in [5, 20, 60, 120, 200]:
        average_volume = volume.rolling(window).mean()
        df[f"volume_{window}d"] = volume / average_volume - 1


    # RSI(14) relative strength index (14 day window)
    rsi_window = 14
    price_change = close.diff()

    gain = price_change.clip(lower=0)
    loss = -price_change.clip(upper=0)

    average_gain = gain.ewm(alpha=1 / rsi_window,min_periods=rsi_window, adjust=False).mean()
    average_loss = loss.ewm(alpha=1 / rsi_window,min_periods=rsi_window,adjust=False).mean()

    relative_strength = (average_gain /average_loss.replace(0, np.nan))
    df["RSI14"] = (100 -100 / (1 + relative_strength))

    
    # MACD, Moving Average Convergence Divergence
    ema12 = close.ewm(span=12,adjust=False).mean()
    ema26 = close.ewm(span=26,adjust=False).mean()

    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9,adjust=False).mean()
    macd_histogram = macd - macd_signal

    # divide by price so it is comparable between stocks
    df["MACD_%"] = macd / close
    df["MACDSignal_%"] = macd_signal / close
    df["MACDHist_%"] = macd_histogram / close


    # volatility (calculates standard deviation)
    for window in [20, 30, 60, 120, 200]:
        df[f"volatility_{window}d"] = (daily_return.rolling(window).std())

    # high-low range
    df["high_low_ratio"] = high_low_ratio

    for window in [20, 30, 60, 120, 200]:
        df[f"high_low_ratio_{window}d_mean"] = (high_low_ratio.rolling(window).mean())

   
    # drawdown (peak-trough) and position within current range
    for window in [60, 120, 200]:
        rolling_high = close.rolling(window).max()
        rolling_low = close.rolling(window).min()
        rolling_range = (rolling_high - rolling_low).replace(0, np.nan)

        df[f"drawdown_{window}d"] = (close / rolling_high - 1)
        df[f"position_in_{window}d_range"] = ((close - rolling_low) / rolling_range)

    
    # Bollinger Band width
    bb_middle = close.rolling(20).mean()
    bb_std = close.rolling(20).std()

    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std

    # band width as a percentage of the current price
    df["Bollinger_Bands_WIDTH_%"] = ((bb_upper - bb_lower) / close)

    # rolling averages of Bollinger Band width
    for window in [30, 60, 120]:
        df[f"Bollinger_Bands_WIDTH_mean_{window}d"] = (df["Bollinger_Bands_WIDTH_%"].rolling(window).mean())

  
    # Average True Range in percentage
    previous_close = close.shift(1)

    true_range = pd.concat([high - low,(high - previous_close).abs(),(low - previous_close).abs()],axis=1).max(axis=1)

    for window in [30, 60, 120, 200]:
        atr = true_range.rolling(window).mean()
        df[f"Average_True_Range_{window}d(%)"] = (atr / close)

    
    # distance from recent lows
    for window in [120, 200]:
        rolling_low = close.rolling(window).min()
        df[f"dist_from_{window}d_low"] = (close / rolling_low - 1)

    
    # volatility-adjusted 200-day return
    df["return_200d_vol_adj"] = (df["return_200d"] /df["volatility_200d"].replace(0, np.nan))

    # replace invalid values
    df = df.replace([np.inf, -np.inf],np.nan)

    return df


# load benchmark
def load_benchmark_data(path, include_target=False):
    # read and normalize the benchmark data
    benchmark_df = pd.read_csv(path)
    benchmark_df = normalize_columns(benchmark_df)

    # benchmark closing prices and daily returns
    close = benchmark_df["Close"]
    daily_return = close.pct_change()
    # benchmark's daily high-low range
    high_low_ratio = (benchmark_df["High"]/ benchmark_df["Low"]- 1)
    # calculated benchmark features
    feature_data = {}

    # used only during model training
    # calculates the benchmark return
    if include_target:
        feature_data["benchmark_target_return_100d"] = (close.shift(-100) / close - 1)

    # historical benchmark returns
    for window in [30, 60, 120, 200]:
        feature_data[f"benchmark_return_{window}d"] = close.pct_change(window)

    # benchmark moving-average ratios
    for window in [50, 100, 200]:
        moving_average = close.rolling(window).mean()
        feature_data[f"benchmark_sma{window}_ratio"] = close / moving_average - 1

    # benchmark volatility
    for window in [30, 60, 120, 200]:
        feature_data[f"benchmark_volatility_{window}d"] = daily_return.rolling(window).std()

    # daily benchmark high-low range
    feature_data["benchmark_high_low_ratio"] = high_low_ratio

    # average benchmark high-low range
    for window in [30, 60, 120, 200]:
        feature_data[f"benchmark_high_low_ratio_{window}d_mean"] = high_low_ratio.rolling(window).mean()

    # benchmark drawdown and range position
    for window in [60, 120, 200]:
        rolling_high = close.rolling(window).max()
        rolling_low = close.rolling(window).min()
        rolling_range = (rolling_high - rolling_low).replace(0, np.nan)

        feature_data[f"benchmark_drawdown_{window}d"] = close / rolling_high - 1
        feature_data[f"benchmark_position_in_{window}d_range"] = (close - rolling_low) / rolling_range

    feature_df = pd.DataFrame(feature_data,index=benchmark_df.index)

    return pd.concat([benchmark_df[["Date"]],feature_df],axis=1)

# compare each stock's historical behaviour with the benchmark.
# these features are designed to compare the stock with benchmark, not a standard technical indicator
def add_relative_features(df):
    df = df.copy()

    # stock high-low range relative to benchmark
    for window in [30, 60, 120, 200]:
        df[f"rel_high_low_ratio_{window}d"] = (df[f"high_low_ratio_{window}d_mean"] - df[f"benchmark_high_low_ratio_{window}d_mean"])

    # stock volatility relative to benchmark
    for window in [30, 60, 120, 200]:
        df[f"rel_volatility_{window}d"] = (df[f"volatility_{window}d"]- df[f"benchmark_volatility_{window}d"])

    # stock historical return relative to benchmark
    for window in [120, 200]:
        df[f"rel_return_{window}d"] = (df[f"return_{window}d"] - df[f"benchmark_return_{window}d"])

    # replace invalid values
    df = df.replace([np.inf, -np.inf],np.nan)

    return df

# organize all available features into groups
# this helps user decide what to use for training the model 
FEATURE_GROUPS = {
    "Stock Returns": [
        "return_5d",
        "return_10d",
        "return_20d",
        "return_30d",
        "return_60d",
        "return_120d",
        "return_200d",
        "return_200d_vol_adj",
    ],

    "Moving averages": [
        "sma5_ratio",
        "sma10_ratio",
        "sma20_ratio",
        "sma50_ratio",
        "sma100_ratio",
        "sma200_ratio",
    ],

    "Stock Volume": [
        "volume_5d",
        "volume_20d",
        "volume_60d",
        "volume_120d",
        "volume_200d",
    ],

    "Stock Momentum": [
        "RSI14",
        "MACD_%",
        "MACDSignal_%",
        "MACDHist_%",
    ],

    "Stock Volatility": [
        "volatility_20d",
        "volatility_30d",
        "volatility_60d",
        "volatility_120d",
        "volatility_200d",
        "Bollinger_Bands_WIDTH_%",
        "Bollinger_Bands_WIDTH_mean_30d",
        "Bollinger_Bands_WIDTH_mean_60d",
        "Bollinger_Bands_WIDTH_mean_120d",
        "Average_True_Range_30d(%)",
        "Average_True_Range_60d(%)",
        "Average_True_Range_120d(%)",
        "Average_True_Range_200d(%)",
    ],

    "Stock Price position": [
        "drawdown_60d",
        "drawdown_120d",
        "drawdown_200d",
        "position_in_60d_range",
        "position_in_120d_range",
        "position_in_200d_range",
        "dist_from_120d_low",
        "dist_from_200d_low",
    ],

    "Benchmark Performance": [
        "benchmark_return_30d",
        "benchmark_return_60d",
        "benchmark_return_120d",
        "benchmark_return_200d",

        "benchmark_sma50_ratio",
        "benchmark_sma100_ratio",
        "benchmark_sma200_ratio",

        "benchmark_volatility_30d",
        "benchmark_volatility_60d",
        "benchmark_volatility_120d",
        "benchmark_volatility_200d",

        "benchmark_high_low_ratio",

        "benchmark_high_low_ratio_30d_mean",
        "benchmark_high_low_ratio_60d_mean",
        "benchmark_high_low_ratio_120d_mean",
        "benchmark_high_low_ratio_200d_mean",

        "benchmark_drawdown_60d",
        "benchmark_drawdown_120d",
        "benchmark_drawdown_200d",

        "benchmark_position_in_60d_range",
        "benchmark_position_in_120d_range",
        "benchmark_position_in_200d_range",
    ],

    "Relative Performance compare to Benchmark": [
        "rel_high_low_ratio_30d",
        "rel_high_low_ratio_60d",
        "rel_high_low_ratio_120d",
        "rel_high_low_ratio_200d",
        "rel_volatility_30d",
        "rel_volatility_60d",
        "rel_volatility_120d",
        "rel_volatility_200d",
        "rel_return_120d",
        "rel_return_200d",
    ],

    "High-low range": [
        "high_low_ratio",
        "high_low_ratio_20d_mean",
        "high_low_ratio_30d_mean",
        "high_low_ratio_60d_mean",
        "high_low_ratio_120d_mean",
        "high_low_ratio_200d_mean",
    ],
}


# validate the feature choices submitted by user
def validate_selected_features(selected_features):
    if not selected_features:
        raise ValueError("Please select at least one feature.")

    # return validated features
    return selected_features