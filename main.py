import numpy as np
import pandas as pd
from Preprocessing.DataHelper import *

def main():
    df = pd.read_csv("data/unengineered_exalt_currency.csv")
    df['Get'] = df['Get'].replace('Divine Orb', 'Exalted Orb')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.to_csv('data/unengineered_exalt_currency.csv', index=False)
    print(df)
    


main()