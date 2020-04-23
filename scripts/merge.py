import os, glob
import pandas as pd

path = "/Users/chiragtubakad/Documents/DL/sentiment-classification-based-on-audio-visual-data/extracted_features/only-AV-All/"

allFiles = glob.glob(path + "/*.csv")
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_)
    list_.append(df)
frame = pd.concat(list_)

frame.reset_index(drop=True,inplace=True)
frame.drop(columns = ["intensity","actor","statement","mean_MFCC"],inplace=True)
frame = frame[[c for c in frame if c not in ['emotion']] + ['emotion']]

frame.to_csv("All-emotions.csv",index=False)
