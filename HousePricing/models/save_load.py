import joblib


def save_models(models, best_model):
    joblib.dump(models[0], "models/lin_reg.pkl")
    # models[1] holds the DecisionTreeRegressor while models[2] holds the
    # RandomForestRegressor. The filenames were previously reversed, so we
    # swap them to store each estimator using the correct name.
    joblib.dump(models[1], "models/tree_reg.pkl")
    joblib.dump(models[2], "models/forest_reg.pkl")
    joblib.dump(best_model, "models/best_model.pkl")
