import  DataUtil
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from CombinedAttributesAdder import CombinedAttributesAdder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from scipy import stats
import os
import joblib


housing = DataUtil.import_data()
training_set, test_set= DataUtil.get_data_set(housing)

housing = training_set.copy()

# Data visualization
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
 s=housing["population"]/100, label="population", figsize=(10,7),
 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,)
plt.legend()
#plt.show()

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
#plt.show()

housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
#plt.show()

#Attribute combination
# housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
# housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
# housing["population_per_household"]=housing["population"]/housing["households"]
# corr_matrix = housing.select_dtypes(include=['number']).corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))

#Data preparing
housing = training_set.drop("median_house_value", axis=1)
housing_labels = training_set["median_house_value"].copy()

#Data Cleaning
housing_num = housing.drop("ocean_proximity", axis=1)

#Prepare the data with pipline
num_pipeline = Pipeline([
 ('imputer', SimpleImputer(strategy="median")),
 ('attribs_adder', CombinedAttributesAdder()),
 ('std_scaler', StandardScaler()),
 ])

#housing_num_tr = num_pipeline.fit_transform(housing_num)  #process the pipline

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([                         #process all data including string columns
 ("num", num_pipeline, num_attribs),
 ("cat", OneHotEncoder(), cat_attribs),
])

housing_prepared = full_pipeline.fit_transform(housing)

#train
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# Try out
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]

some_data_prepared = full_pipeline.transform(some_data)

#print("Predicted values:",lin_reg.predict(some_data_prepared))
#print("Actual values:",list(some_labels))

#Test on training data
housing_predictions = lin_reg.predict(housing_prepared)
mse = mean_squared_error(housing_labels, housing_predictions)
smse= np.sqrt(mse)
print("Linear reg SMSE",smse)  # Too high(underfits)

# Try another models
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

#Test on training data
housing_predictions = tree_reg.predict(housing_prepared)
mse = mean_squared_error(housing_labels, housing_predictions)
smse= np.sqrt(mse)

#print("Predicted values:",tree_reg.predict(some_data_prepared))
#print("Actual values:",list(some_labels))
print("Decision Tree reg SMSE",smse) #Zero overfit

# Use validation sets
#randomly splits the training set into 10 distinct subsets called folds, then it
#trains and evaluates the Decision Tree models 10 times, picking a different fold for
#evaluation every time and training on the other 9 folds. The result is an array con
#taining the 10 evaluation scores
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
print("************************Decision Tree RMSE SCORES",tree_rmse_scores)
print("Mean SCORES",tree_rmse_scores.mean())
print("std SCORES",tree_rmse_scores.std())

#Try same with linear reg
scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
print("*************************Linear Reg RMSE SCORES",tree_rmse_scores)
print("Mean SCORES",tree_rmse_scores.mean())
print("std SCORES",tree_rmse_scores.std())

#Try change to RandomForest
forest_reg = RandomForestRegressor()
scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
scoring="neg_mean_squared_error", cv=10)
forst_rmse_scores = np.sqrt(-scores)
print("***************************forest Reg RMSE SCORES",tree_rmse_scores)
print("Mean SCORES",forst_rmse_scores.mean())
print("std SCORES",forst_rmse_scores.std())

#Save models
os.makedirs("models", exist_ok=True)

# Save the models
joblib.dump(lin_reg, "models/lin_reg.pkl")
joblib.dump(forest_reg, "models/forest_reg.pkl")
joblib.dump(tree_reg, "models/tree_reg.pkl")

# fine tune the parameters
#grid serach helps to try all the possible combinations of hyperparameters
param_grid = [
 {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
 {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
 scoring='neg_mean_squared_error', return_train_score=True)

grid_search.fit(housing_prepared, housing_labels)

#best result
print(grid_search.best_params_)
#best estimator
print(grid_search.best_estimator_)
#scores
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
   print(np.sqrt(-mean_score), params)

#check the feature importance
feature_importances = grid_search.best_estimator_.feature_importances_
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
print(sorted(zip(feature_importances, attributes), reverse=True))


#Test on the test set
finalModel = grid_search.best_estimator_
test_data = test_set.drop("median_house_value", axis=1)
test_labels = test_set["median_house_value"].copy()
test_prepared = full_pipeline.transform(test_data)

test_predictions = finalModel.predict(test_prepared)

test_mse = mean_squared_error(test_labels, test_predictions)
test_smse= np.sqrt(test_mse)

print("Test MSE",test_mse)
print("Test RMSE",test_smse)

#get 95% confidence instead of mse only
confidence = 0.95
squared_errors = (test_predictions - test_labels) ** 2
print(np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1, loc=squared_errors.mean(),
scale=stats.sem(squared_errors))))











