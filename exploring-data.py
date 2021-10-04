# First file for accessing the training dataset

# Import useful labraries
import pandas as pd
import json

# returns JSON object as a dictionary
f = open('data/train.json',)

# creating a dataframe
df = pd.DataFrame(json.load(f))

# visualising the first rows
print(df.head())
