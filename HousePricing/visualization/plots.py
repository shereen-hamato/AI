import  matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


def visualize_data(train_set):
    # Data visualization
    train_set.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
     s=train_set["population"]/100, label="population", figsize=(10,7),
     c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,)
    plt.legend()
    #plt.show()

    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    scatter_matrix(train_set[attributes], figsize=(12, 8))
    #plt.show()

    train_set.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
    #plt.show()