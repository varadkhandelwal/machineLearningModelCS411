import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle

#### ML model
from sklearn import linear_model
from sklearn import model_selection

#### Loading dataset
database = pd.read_csv("k201801.csv")

#### Replacing countries with US/Canda and Others
database["Country_ML"] = database["country"].isin(['US', 'CA']) 
database["Country_ML"][database["Country_ML"] == True] = 1 

#### separating date into years and months (Year represents growth & months represent cyclicty in funding cycle)
Month = database["launched"].str.split("/", expand = True)
database["Months"] = Month.loc[:,0]

#### Doing OneHot encoding for categories
encoded_cat = pd.get_dummies(database['main_category'] ,prefix="")

#### Selecting relevant columns for ML
X = database.loc[:, ["Months", "backers", "Country_ML"]]
X = X.join(encoded_cat).values
Y = database["usd_goal_real"].values

#### Preparing dataset for training
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=42)

#### Applying different models and finding out the best one
lr = linear_model.LinearRegression()
lr.fit(X_train, y_train)

#### Saving the model into a file
pickle.dump(lr, open('model.pkl','wb'))


