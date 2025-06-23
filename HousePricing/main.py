from data import DataUtil
from models import train, evaluate, save_load
from pipeline import processing
from visualization import plots
import os

try:
    print("Starting main pipeline...")

    print("Import data...")
    housing = DataUtil.import_data()
    train_set, test_set = DataUtil.get_data_set(housing)

    print("Visualize data...")
    plots.visualize_data(train_set)

    print("Preparing data...")
    prepared_data, labels, full_pipeline = processing.prepare_data(train_set)

    print("Training data...")
    # Train models
    lin_reg, tree_reg, forest_reg = train.train_all_models(prepared_data, labels)

    # Evaluate
    print("Validate data...")
    evaluate.cross_validate_models([lin_reg, tree_reg, forest_reg], prepared_data, labels)

    # Fine-tune best models
    print("Fine tune data...")
    best_model, feature_importances = train.grid_search_best(prepared_data, labels, forest_reg)

    # Save
    print("Save model...")
    os.makedirs("models", exist_ok=True)
    save_load.save_models([lin_reg, tree_reg, forest_reg], best_model)

    # Test
    print("Evaluate on test data...")
    evaluate.evaluate_on_test(best_model, full_pipeline, test_set)
except Exception as e:
    print(f"An error occurred: {e}")
