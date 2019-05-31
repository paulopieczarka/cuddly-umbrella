from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random as rd
import numpy as np

class Perceptron:
  def __init__ (self):
    self.weights = None
    self.threshold = None

  def model (self, xs):
    return 1 if (np.dot(self.weights, xs) >= self.threshold) else 0

  def predict (self, xs):
    y = [self.model(x) for x in xs]
    return np.array(y)

  def train (self, xs, ys, epochs = 1, learningRate = 1):
    self.weights = np.ones(xs.shape[1])
    self.threshold = 0

    accuracy = {}
    max_accuracy = 0
    wt_matrix = []

    print("-> Training perceptron...")

    # for all epochs
    for i in range(epochs):
      for x, y in zip(xs, ys):
        y_pred = self.model(x)
        if (y == 1 and y_pred == 0):
          self.weights = self.weights + learningRate * x
          self.threshold = self.threshold - learningRate * 1
        elif (y == 0 and y_pred == 1):
          self.weights = self.weights - learningRate * x
          self.threshold = self.threshold + learningRate * 1

      wt_matrix.append(self.weights)
      accuracy[i] = accuracy_score(self.predict(xs), ys)
      if (accuracy[i] > max_accuracy):
        max_accuracy = accuracy[i]
        chkptw = self.weights
        chkptb = self.threshold

    # checkpoint -> save the weights and threshold value
    self.weights = chkptw
    self.threshold = chkptb

    print("-> Done. Max accuracy = %s" %max_accuracy)

    #plot the accuracy values over epochs
    # plt.plot(np.array(list(accuracy.values())))
    # plt.xlabel("Epoch #")
    # plt.ylabel("Accuracy")
    # plt.ylim([0, 1])
    # plt.show()

class PerceptronPocketRatchet:
  def __init__ (self):
    self.weights = None
    self.run_ok = None
    self.corrent = None

  def classify(self, xs):
    return np.sign(np.dot(xs, self.weights))

  def update_weights (self, x, y):
    self.weights = np.add(self.weights, x * y)

  def prep_values (self, xs, ys):
    """
      Pocket with ratchet algorithm only works with -1, 0, 1 values.
      This mehtod coverts 0s to -1s.
    """
    xs[xs == 0] = -1
    ys[ys == 0] = -1
    return xs, ys

  def train (self, xs, ys, epochs=1):
    # fix dataset values
    xs, ys = self.prep_values(xs, ys)

    self.weights = np.zeros(xs.shape[1])
    self.run_ok = 0 # consecutive correct classifications usng perceptron
    self.num_ok = 0 # total of correct classifications

    iterations = 0
    pocket_weigths = np.zeros(xs.shape[1])
    pocket_run_ok = 0
    pocket_num_ok = 0

    print("-> Start pocket and ratchet training...")

    prediction = self.classify(xs)
    misclassified = [i for i in range(xs.shape[0]) if ys[i] != prediction[i]]

    while iterations < epochs:
      # stops when there's no misses
      if not misclassified:
        break

      iterations += 1
      random_sample = rd.choice(range(xs.shape[0])) # random pick a training example

      if random_sample not in misclassified:
        self.run_ok += 1
        if self.run_ok > pocket_run_ok:
          self.num_ok = xs.shape[0] - len(misclassified) # compute, by checking every training example
          if self.num_ok > pocket_num_ok: # ratchet and pocket it
            pocket_weigths = self.weights
            pocket_run_ok = self.run_ok
            pocket_num_ok = self.num_ok
      else:
        # change perceptron weights
        self.update_weights(xs[random_sample], ys[random_sample])
        self.run_ok = 0

      prediction = self.classify(xs)
      misclassified = [i for i in range(xs.shape[0]) if ys[i] != prediction[i]]

    # update weights with pocket_weights
    self.weights = pocket_weigths

    prediction = self.classify(xs)
    misclassified = [i for i in range(xs.shape[0]) if ys[i] != prediction[i]]

    accuracy = accuracy_score(prediction, ys)
    print("-> Done. Max accuracy = %s" % accuracy)
