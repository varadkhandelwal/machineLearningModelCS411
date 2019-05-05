import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def loadData(parameters):
    database = pd.read_csv("k201801.csv")
    Selected = database[database['category'] == parameters[0]] 
    Selected["launched"] = Selected["launched"].astype("datetime64")
    count = Selected["launched"].groupby(Selected["launched"].dt.year).count()

