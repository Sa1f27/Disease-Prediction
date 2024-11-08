import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.ticker as ticker
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


import pandas as pd
pd.options.mode.chained_assignment = None
from tensorflow.keras.layers import InputLayer,Dense, Activation,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler


df=pd.read_csv(r'C:\Users\huzai\vs code projects\Disease-Prediction\dataset\heart.csv')

df.head()

df.nunique(axis=0)


Map categorical values to numerical values
data_mapping = {
    'RestingECG': {'Normal': 0, 'ST': 1, 'LHV': 2},
    'ST_Slope': {'Flat': 0, 'Up': 1, 'Down': 2},
    'ExerciseAngina': {'N': 0, 'Y': 1},
    'ChestPainType': {'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3},
    'Sex': {'M': 0, 'F': 1}
}
# Apply mappings to columns
for column, mapping in data_mapping.items():
    df[column] = df[column].map(mapping)


for i,j in enumerate(df["Sex"]):
    if j=="M":
        df["Sex"][i] = 0
    else:
        df["Sex"][i] = 1
        
for i,j in enumerate(df["ChestPainType"]):
    if j=="ATA":
        df["ChestPainType"][i] = 0
    elif j=="NAP":
        df["ChestPainType"][i] = 1
    elif j=="ASY":
        df["ChestPainType"][i] = 2
    else:
        df["ChestPainType"][i] = 3
        
for i,j in enumerate(df["RestingECG"]):
    if j=="Normal":
        df["RestingECG"][i] = 0
    elif j=="ST":
        df["RestingECG"][i] = 1
    else:
        df["RestingECG"][i] = 2

for i,j in enumerate(df["ST_Slope"]):
    if j=="Flat":
        df["ST_Slope"][i] = 0
    elif j=="Up":
        df["ST_Slope"][i] = 1
    else:
        df["ST_Slope"][i] = 2
        
for i,j in enumerate(df["ExerciseAngina"]):
    if j=="N":
        df["ExerciseAngina"][i] = 0
    else:
        df["ExerciseAngina"][i] = 1


X = df[["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS", "RestingECG", "MaxHR", 'ExerciseAngina', "Oldpeak", "ST_Slope"]].values
Y = df["HeartDisease"].values

# Standardize features
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

k=2
neigh=KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)
yhat=neigh.predict(X_test)

from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

# Save KNN model using pickle
with open('Heart_Disease_KNN.pkl', 'wb') as knn_pickle:
    pickle.dump(neigh, knn_pickle)

# Load KNN model and make predictions
loaded_knn_model = pickle.load(open('Heart_Disease_KNN.pkl', 'rb'))
knn_result = loaded_knn_model.predict(X_test)
print("Loaded KNN Model Test Accuracy:", metrics.accuracy_score(y_test, knn_result))

# Neural Network model
model = Sequential([
    Dense(8, activation="relu", input_dim=11),
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])
model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
# Train Neural Network model
model.fit(X_train, y_train, epochs=100, verbose=1)

# Evaluate Neural Network model
loss, acc = model.evaluate(X_test, y_test, verbose=1)
print(f"Neural Network Model Test Accuracy: {acc}")

# Save Neural Network model architecture and weights
model_json = model.to_json()
with open(r"C:/Users/huzai/vs code projects/Diseases/models/HeartDisease.json", "w") as json_file:
    json_file.write(model_json)
model.save(r"C:/Users/huzai/vs code projects/Diseases/models/HeartDisease.h5")
print("Neural Network model saved to disk")
