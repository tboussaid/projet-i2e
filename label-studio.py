import pandas as pd
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# returns JSON object as a dictionary
f = open('data/train.json',) 

# creating a dataframe
df_raw = pd.DataFrame(json.load(f))

# visualising the first rows
print(df_raw.head())

def to_RGB(arr):
    normalized = (arr-np.min(arr))/(np.max(arr)-np.min(arr))
    im = Image.fromarray(plt.cm.jet(normalized, bytes=True))
    im = im.resize((300, 300), Image.ANTIALIAS)
    return (im)

for i in range(df_raw.shape[0]) :
    #f, (im1, im2) = plt.subplots(1, 2)
    im1 = to_RGB(np.array(df_raw.iloc[i,1]).reshape(75,75))
    im2 = to_RGB(np.array(df_raw.iloc[i,2]).reshape(75,75))
    im2 = im2.convert("RGB")
    #im1.imshow(np.array(df_raw.iloc[i,1]).reshape(75,75))
    #im2.imshow(np.array(df_raw.iloc[i,2]).reshape(75,75))
    #plt.show()
    im1.save("data/"+str(df_raw.iloc[i,0])+"_b1.png")
    im2.save("data/"+str(df_raw.iloc[i,0])+"_b2.png")