import os
import tarfile
import urllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("../datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()



def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)



def examine_data(housing):
    pd.set_option('display.max_columns', None)
    print('****************Read Head****************')
    print(housing.head())
    print('****************Read Info****************')
    print(housing.info())
    print('****************Description****************')
    print(housing.describe())
    print('****************Plot Histograms****************')
    housing.hist(bins=50, figsize=(20, 15))
    plt.show()


def import_data(plot=False):
    fetch_housing_data()
    housing = load_housing_data()
    if plot:
        examine_data(housing)
    return  housing


def get_data_set(housing):
    # x_train, x_test= train_test_split(housing,test_size=0.2, random_state=42)
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        training_set = housing.loc[train_index]
        test_set = housing.loc[test_index]

    for set_ in (training_set, test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    return training_set, test_set
