import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("dataset_car_data.csv")
# print(df.head())
# print(df.shape)

# # categorical features
#
# # check null values
# print(df.isnull().sum())
#
# print(df.describe())
#
cols = list(df.columns)
# print(cols)

# remove car_name
mod_df = df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type',
             'Transmission', 'Owner']]

# add new feature
mod_df['Current_Year'] = 2020
# print(mod_df.head())

mod_df['Number_of_Years'] = mod_df['Current_Year'] - mod_df['Year']
# print(mod_df.head())

mod_df.drop(['Year','Current_Year'], axis=1, inplace=True)
# print(mod_df.head())

# drop first to avoid dummy variable trap (drops one feature if there are three features)
mod_df = pd.get_dummies(mod_df, drop_first=True)
# print(mod_df.head())
# print(mod_df.columns)

corr = mod_df.corr()
# print(corr)

# sns.pairplot(mod_df)
# plt.show()

# top_corr_features = corr.index
# plt.figure(figsize=(20,20))
# g = sns.heatmap(mod_df[top_corr_features].corr(), annot=True, cmap='RdYlGn')
# plt.show()

X = mod_df.iloc[:,1:]
y = mod_df.iloc[:,0]

# print(X.head())
# print(y.head())

# feature importance
from sklearn.ensemble import ExtraTreesRegressor

model = ExtraTreesRegressor()
model.fit(X, y)
# print(model.feature_importances_)

# visualize feature importances
# feat_imp = pd.Series(model.feature_importances_, index=X.columns)
# feat_imp.nlargest(n=5).plot(kind='barh')
# plt.show()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
# print(X_train.shape)

from sklearn.ensemble import RandomForestRegressor

rf_random = RandomForestRegressor()

# Hyperparameters
n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(start=5, stop=30, num=6)]
min_samples_split = [2,5,10,15,100]
min_samples_leaf = [1,2,5,10]

from sklearn.model_selection import RandomizedSearchCV

random_grid = {'n_estimators':n_estimators,
               'max_features':max_features,
               'max_depth':max_depth,
               'min_samples_split':min_samples_split,
               'min_samples_leaf':min_samples_leaf}

print(random_grid)

best_model = RandomizedSearchCV(estimator=rf_random, param_distributions=random_grid,scoring='neg_mean_squared_error',
                            n_iter=10, cv=5, verbose=2, random_state=4, n_jobs=10)
best_model.fit(X_train, y_train)

predictions = best_model.predict(X_test)
print(predictions)

# sns.distplot(y_test-predictions)
# plt.show()

# plt.scatter(y_test,predictions)
# plt.show()

import pickle

file = open('car_prediction_model.pkl', 'wb')
pickle.dump(best_model, file)












