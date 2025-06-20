from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from pipeline.feature_engineering import CombinedAttributesAdder


def prepare_data(training_set):
    # Data preparing
    housing = training_set.drop("median_house_value", axis=1)
    housing_labels = training_set["median_house_value"].copy()

    # Data Cleaning
    housing_num = housing.drop("ocean_proximity", axis=1)

    # Prepare the data with pipline
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

    # housing_num_tr = num_pipeline.fit_transform(housing_num)  #process the pipline

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    full_pipeline = ColumnTransformer([  # process all data including string columns
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

    housing_prepared = full_pipeline.fit_transform(housing)
    return housing_prepared, housing_labels, full_pipeline
