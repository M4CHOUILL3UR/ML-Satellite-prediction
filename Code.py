# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 13:42:30 2025

@author: julien
"""

import pandas as pd
import math as m
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#%%
df_train = pd.read_csv("jan_train.csv")
df_train = df_train.drop("id", axis=1)
df_test = pd.read_csv("jan_test.csv")
df_test = df_test.drop("id", axis=1)
df_answer = pd.read_csv("answer_key.csv")

df_test.insert(2, "x", df_answer["x"])
df_test.insert(3, "y", df_answer["y"])
df_test.insert(4, "z", df_answer["z"])
df_test.insert(5, "Vx", df_answer["Vx"])
df_test.insert(6, "Vy", df_answer["Vy"])
df_test.insert(7, "Vz", df_answer["Vz"])

ids_train = df_train["sat_id"].unique()
ids_test = df_test["sat_id"].unique()
N_train = len(ids_train)
N_test = len(ids_train)

satellites_train = np.empty(N_train, dtype=object)
satellites_test = np.empty(N_test, dtype=object)

for i in range(N_train):
    satellites_train[i] = df_train[df_train["sat_id"] == i].drop("sat_id", axis=1)
for i in range(N_test):
    satellites_test[i] = df_test[df_test["sat_id"] == i].drop("sat_id", axis=1)

del df_answer
del df_train
del df_test

#%%
