import numpy as np
import pandas as pd
from Preprocessing.DataHelper import *

def main():
    df = pd.read_csv("data/unengineered_exalt_currency.csv", index_col='Date')
    df = DataHelper.extract_exalt_and_divine(df)
    df['Get'] = df['Get'].replace('Divine Orb', 'Exalted Orb')
    df.to_csv('data/unengineered_exalt_currency.csv')
    


main()