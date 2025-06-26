import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, precision_recall_curve, \
    roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal



mnist = fetch_openml('mnist_784', version=1)
print(mnist.keys())

x, y = np.array(mnist['data']), np.array(mnist['target'])

y = y.astype(np.uint8)  # convert to int instead of string

print(x.shape)
print(y.shape)

some_digit = x[0]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap="binary")
plt.axis("off")
plt.show()

x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

# Training a Binary Classifier
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)



# choose the appropriate metric for your
#  task, evaluate your classifiers using cross-validation, select the precision/recall trade
# off that fits your needs, and use ROC curves and ROC AUC scores to compare vari
# ous models.


sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(x_train, y_train_5)

print(sgd_clf.predict([some_digit]))

# validation
validation_accuracy = cross_val_score(sgd_clf, x_train, y_train_5, cv=3, scoring="accuracy")
print("validation_accuracy ",
      validation_accuracy)  # [0.95035 0.96035 0.9604 ] which is misleading as the count of 5 in the set is pretty low

y_train_predict = sgd_clf.predict(x_train)
print("confusion_matrix: ", confusion_matrix(y_train_5, y_train_predict))
print("recall_score: ", recall_score(y_train_5, y_train_predict))
print("precision_score: ", precision_score(y_train_5, y_train_predict))

# F1 score is the harmonic mean of precision and recall. Unfortunately, you can’t have it both ways: increasing precision reduces recall, and
#  vice versa. This is called the precision/recall trade-off
print("f1_score: ", f1_score(y_train_5, y_train_predict))

# decision_function() method, which returns a score for each instance
y_score = sgd_clf.decision_function([some_digit])
print(y_score)
threshold = 0
print(y_score > threshold)

y_scores = cross_val_predict(sgd_clf, x_train, y_train_5, cv=3, method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]  # ~7816
y_train_pred_90 = (y_scores >= threshold_90_precision)

print("precision_score with 90% threshold", precision_score(y_train_5, y_train_pred_90))
print("recall_score with 90% threshold", recall_score(y_train_5, y_train_pred_90))

# The receiver operating characteristic (ROC) curve is another common tool used with binary classifiers.
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
plot_roc_curve(fpr, tpr)
plt.show()

# One way to compare classifiers is to measure the area under the curve (AUC). A per
# fect classifier will have a ROC AUC equal to 1, whereas a purely random classifier will
#  have a ROC AUC equal to 0.5.
print("SGDClassifier roc_auc_score", roc_auc_score(y_train_5, y_scores))

#Train with RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, x_train, y_train_5, cv=3, method="predict_proba")
y_scores_forest = y_probas_forest[:, 1]   # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)

plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show()

print("RandomForestClassifier roc_auc_score",roc_auc_score(y_train_5, y_scores_forest))

#Multiclass classifier
# Scikit-Learn detects when you try to use a binary classification algorithm for a multi
# class classification task, and it automatically runs OvR or OvO

svc_clf = SVC(random_state=42)
svc_clf.fit(x_train, y_train)
print(svc_clf.predict([some_digit]))
some_digit_score = svc_clf.decision_function([some_digit])
print("some_digit_score", some_digit_score)
print("max:", np.argmax(some_digit_score))
print("classes:", svc_clf.classes_)

#force to use OvR classifier
ovr_clf = OneVsRestClassifier(SVC(random_state=42))
ovr_clf.fit(x_train, y_train)
print("ovr_clf.predict([some_digit]): ",ovr_clf.predict([some_digit]))
print("len(ovr_clf.estimator): ",len(ovr_clf.estimators_))

# using RandomForestClassifier
forest_clf.fit(x_train, y_train)
print(forest_clf.predict([some_digit]))
print("forest_clf.predict_proba: ",forest_clf.predict_proba([some_digit]))
print("cross_val_predict(x_train: ",
      cross_val_predict(forest_clf, x_train, y_train_5, cv=3, method="predict_proba"))

#Imporve by scaling the input
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train.astype(np.float64))
y_train_pred = cross_val_predict(sgd_clf, x_train_scaled, y_train_5, cv=3)
print("cross_val_predict(x_train_scaled: ",  y_train_pred )

#Error analysis
confusion_matrix = confusion_matrix(y_train_5, y_train_pred)
print("confusion_matrix",confusion_matrix)
plt.matshow(confusion_matrix, cmap=plt.cm.gray)
plt.show()

#Multilabel Classification, e.g. classify odd and more than 7
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]
knn_clf = KNeighborsClassifier()
knn_clf.fit(x_train, y_multilabel)
print(knn_clf.predict([some_digit]))

# validation
y_train_knn_pred = cross_val_predict(knn_clf, x_train, y_multilabel, cv=3)

print(f1_score(y_multilabel, y_train_knn_pred, average="macro")) # Option is to give each label a weight equal to its support (i.e., the number of instances with that target label). To do this, simply set average="weighted"

# Multioutput Classification
noise = np.random.randint(0, 100, (len(x_train), 784))
x_train_mod = x_train + noise
noise = np.random.randint(0, 100, (len(x_test), 784))
X_test_mod = x_test + noise
y_train_mod = x_train
y_test_mod = x_test
knn_clf.fit(x_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[0]])
#plot_digit(clean_digit)



