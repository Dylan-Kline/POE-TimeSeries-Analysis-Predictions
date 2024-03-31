import numpy as np
import pandas as pd
from Preprocessing.DataHelper import *

def main():
    df = DataHelper.update_data("data/currency.csv", "data/Affliction.currency.csv")
    print(df.columns)
    df = DataHelper.extract_exalt_and_divine(df)
    df['Get'] = df['Get'].replace('Divine Orb', 'Exalted Orb')
    exalts_df = df
    print(exalts_df)

main()