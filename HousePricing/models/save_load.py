import joblib


def save_models(models, best_model):
    joblib.dump(models[0], "models/lin_reg.pkl")
    joblib.dump(models[1], "models/forest_reg.pkl")
    joblib.dump(models[2], "models/tree_reg.pkl")
    joblib.dump(best_model, "models/best_model.pkl")
