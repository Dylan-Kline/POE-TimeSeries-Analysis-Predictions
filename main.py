import numpy as np
import pandas as pd
from Preprocessing.DataHelper import *
from Visualization.DataVisualization import *

def main():
    df = pd.read_csv("data/unengineered_exalt_currency.csv")
    print(df.columns)

main()