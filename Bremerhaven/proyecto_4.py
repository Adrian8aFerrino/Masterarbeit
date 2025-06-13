# Part 1: Import libraries and data
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)

df_autoparts = pd.read_csv("/Users/ochoa/PycharmProjects/Bremerhaven/databases/agent_flow.csv",
                           delimiter=",", header=0)
