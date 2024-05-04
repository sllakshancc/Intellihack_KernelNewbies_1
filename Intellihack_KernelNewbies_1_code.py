# KernelNewbies

# Import libs
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

##### TASK 01 #####

# dataset
data = pd.read_csv("Crop_Dataset - Crop_Dataset.csv")

# extract features from the dataset
X = data.drop(columns=["Total_Nutrients","Temperature_Humidity","Log_Rainfall","Label","Label_Encoded"])
y = data["Label"]
X_cols = list(X.columns) 
X_array = X.values

# scaling numerical features
scalar = StandardScaler()
X_scaled = scalar.fit_transform(X_array, X_cols)

##### TASK 02 #####

# split the dataset to get training data and testing data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 0)

# initialize a RandomForestClassifier
model = RandomForestClassifier(n_estimators = 100, random_state = 0)

# train the model
model.fit(X_train, y_train)

##### TASK 03 #####

# model accuracy evaluation
y_predict = model.predict(X_test)
print("Accuracy of the Model:", accuracy_score(y_test, y_predict))

##### TASK 04 #####

# the joblib model
joblib.dump(model, 'crops_recommendation_model.joblib')
# load the model
crops_recommendation_model = joblib.load('crops_recommendation_model.joblib')

# predict top 3 crops based on user input
eval_environment = [[]]
features = X.columns
for i, feature in enumerate(features):
    value = float(input(f"{feature}: "))
    eval_environment[0].append(value)   

# numercial scaling of user input
eval_environment_scaled = scalar.transform(eval_environment)

# predict probabilities
predicted_probabilities = crops_recommendation_model.predict_proba(eval_environment_scaled)[0]
# get the top three
top_three_indices = predicted_probabilities.argsort()[-3:][::-1]
top_three_labels = [crops_recommendation_model.classes_[index] for index in top_three_indices]
top_three_probabilities = predicted_probabilities[top_three_indices]

# display the prediction
print("Top Three Crops that Matches Your Input:")
for label, probability in zip(top_three_labels, top_three_probabilities):
    print(f"{label}: {probability:.2f}")