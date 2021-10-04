# First file for accessing the training dataset

# Import useful labraries
import pandas as pd
import json

# Import Google Drive
from google.colab import drive
drive.mount('/content/drive')

# returns JSON object as a dictionary
f = open('/content/drive/MyDrive/Iceberg/train.json',) # The train.json file needs to be in an "Iceberg" directory

# creating a dataframe
df = pd.DataFrame(json.load(f))

# visualising the first rows
print(df.head())
