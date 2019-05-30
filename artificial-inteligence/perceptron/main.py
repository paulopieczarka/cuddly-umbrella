from votes_dataset import HouseVotes_DataSet
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# open and preprocess house-votes dataset
xs, ys, dataset = HouseVotes_DataSet()

class Perceptron:
  def __init__ (self):
    self.weights = None
    self.threshold = None

  def model (self, xs):
    return 1 if (np.dot(self.weights, xs) >= self.threshold) else 0

  def predict (self, xs):
    y = [self.model(x) for x in xs]
    return np.array(y)

  def fit (self, xs, ys, epochs = 1, lr = 1):
    self.weights = np.ones(xs.shape[1])
    self.threshold = 0

    accuracy = {}
    max_accuracy = 0
    wt_matrix = []

    # for all epochs
    for i in range(epochs):
      for x, y in zip(xs, ys):
        y_pred = self.model(x)
        if (y == 1 and y_pred == 0):
          self.weights = self.weights + lr * x
          self.threshold = self.threshold - lr * 1
        elif (y == 0 and y_pred == 1):
          self.weights = self.weights - lr * x
          self.threshold = self.threshold + lr * 1

      wt_matrix.append(self.weights)
      accuracy[i] = accuracy_score(self.predict(xs), ys)
      if (accuracy[i] > max_accuracy):
        max_accuracy = accuracy[i]
        chkptw = self.weights
        chkptb = self.threshold

    # checkpoint -> save the weights and threshold value
    self.weights = chkptw
    self.threshold = chkptb

    print(max_accuracy)

# train test split
x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size = 0.1, stratify = ys, random_state = 1)

perceptron = Perceptron()
perceptron.fit(x_train, y_train, 10000, .3)
# mat = perceptron.predict(xs)
