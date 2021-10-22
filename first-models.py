# import useful labraries
import json
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# import Google Drive
from google.colab import drive
drive.mount('/content/drive')

# returns JSON object as a dictionary
f = open('/content/drive/MyDrive/Iceberg/train.json',) # The train.json file needs to be in an "Iceberg" directory

# creating a dataframe
df = pd.DataFrame(json.load(f))

# visualising the first rows
print(df.head())

# selecting the prediction target and the model features
y = df["is_iceberg"]
df_features = ['band_1', 'band_2','inc_angle']
X = df[df_features]

# split data into training and validation data, for both features and target
# The split is based on a random number generator. 
# Supplying a numeric value to the random_state argument guarantees we get the same 
# split every time we run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# Define model
df_model = DecisionTreeRegressor()
# Fit model 
df_model.fit(train_X, train_y)

# get predicted values on validation data
val_predictions = df_model.predict(val_X)

print(mean_absolute_error(val_y, val_predictions))