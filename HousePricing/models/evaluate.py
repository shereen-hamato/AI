import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from scipy import stats

def cross_validate_models(models, housing_prepared, housing_labels):
    # Use validation sets
    #randomly splits the training set into 10 distinct subsets called folds, then it
    #trains and evaluates the Decision Tree models 10 times, picking a different fold for
    #evaluation every time and training on the other 9 folds. The result is an array con
    #taining the 10 evaluation scores
    lin_reg, tree_reg, forest_reg = models
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
    scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
    scoring="neg_mean_squared_error", cv=10)
    forst_rmse_scores = np.sqrt(-scores)
    print("***************************forest Reg RMSE SCORES",tree_rmse_scores)
    print("Mean SCORES",forst_rmse_scores.mean())
    print("std SCORES",forst_rmse_scores.std())


def evaluate_on_test(best_model, full_pipeline, test_set):
    #Test on the test set
    finalModel = best_model
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
    return None