import pandas as pd 
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent/"data"/"raw"

def load_data():
    train = pd.read_csv(DATA_DIR/"train.csv")
    test = pd.read_csv(DATA_DIR/"test.csv")
    orig = pd.read_csv(DATA_DIR/"orig.csv")
    return train,test,orig
