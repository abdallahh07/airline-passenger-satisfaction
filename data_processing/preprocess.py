import pandas as pd
from utils.config import TARGET, ID_COL, CAT_COLS, RATING_COLS


def x_y(df):                                         # train (or orig)
    x = df.drop(columns=[ID_COL, TARGET], errors="ignore")
    y = df[TARGET].astype(int)
    return x, y


def imputer(df):
    df = df.copy()
    df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(0)
    if TARGET in df:                                 # test has no target
        df[TARGET] = df[TARGET].astype(int)
    for col in CAT_COLS:
        df[col] = df[col].astype("category")
    return df


def add_features(df):
    df = df.copy()
    df["customer_rating"] = df[RATING_COLS].sum(axis=1)
    df["delay_diff"] = df["Arrival Delay in Minutes"] + df["Departure Delay in Minutes"]
    df["airplane_comfort"] = (df["Inflight wifi service"] + df["Food and drink"]
                              + df["Seat comfort"] + df["Inflight entertainment"]
                              + df["Leg room service"])
    return df