import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt

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

sdg_clf = SGDClassifier(random_state=42)
sdg_clf.fit(x_train, y_train_5)

print(sdg_clf.predict([some_digit]))

#validation
validation_accuracy = cross_val_score(sdg_clf, x_train, y_train_5, cv=3, scoring="accuracy")
print("validation_accuracy ", validation_accuracy)  #[0.95035 0.96035 0.9604 ] which is misleading as the count of 5 in the set is pretty low

y_train_predict = sdg_clf.predict(x_train)
print("confusion_matrix: ",confusion_matrix(y_train_5, y_train_predict))
print("recall_score: ",recall_score(y_train_5, y_train_predict))
print("precision_score: ",precision_score(y_train_5,y_train_predict ))

# F1 score is the harmonic mean of precision and recall. Unfortunately, you can’t have it both ways: increasing precision reduces recall, and
#  vice versa. This is called the precision/recall trade-off
print("f1_score: ",f1_score(y_train_5,y_train_predict ))



