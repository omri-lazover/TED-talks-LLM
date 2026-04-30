import pandas as pd
import numpy as np

ROWS_NUM = 500

def reduce_df(df, rows = 50):
    return df[:rows]

if __name__ == "__main__":
    df = pd.read_csv("ted_talks_en.csv")
    reduced_df = reduce_df(df, rows=ROWS_NUM)
    reduced_df.to_csv("ted_talks_en_reduced.csv", index=False)