from sklearn.linear_model import  LinearRegression
from sklearn.tree import  DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV
import numpy as np

def train_all_models(housing_prepared,housing_labels):
    #train
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)

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
    print("***************************forest Reg RMSE SCORES", forst_rmse_scores)
    print("Mean SCORES",forst_rmse_scores.mean())
    print("std SCORES",forst_rmse_scores.std())

    return [lin_reg,tree_reg,forest_reg]


def grid_search_best(housing_prepared, housing_labels, model):
    # fine tune the parameters
    #grid serach helps to try all the possible combinations of hyperparameters
    param_grid = [
     {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
     {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]
    grid_search = GridSearchCV(model, param_grid, cv=5,
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

    # #check the feature importance
    # feature_importances = grid_search.best_estimator_.feature_importances_
    # extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
    # cat_encoder = full_pipeline.named_transformers_["cat"]
    # cat_one_hot_attribs = list(cat_encoder.categories_[0])
    # attributes = num_attribs + extra_attribs + cat_one_hot_attribs
    # print(sorted(zip(feature_importances, attributes), reverse=True))


    return grid_search.best_estimator_, grid_search.best_estimator_.feature_importances_