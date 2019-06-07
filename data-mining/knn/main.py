import numpy as np
import math
from sklearn.metrics import accuracy_score
from votes_dataset import HouseVotes_DataSet
from sklearn.model_selection import train_test_split

# open and preprocess house-votes dataset
xs, ys, dataset = HouseVotes_DataSet()

# train test split
x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size = 0.25, stratify = ys, random_state = 1)

print(x_train)

def euclid_dist (x1, x2):
  return math.sqrt(np.sum([y**2 for y in (x1 - x2)]))

def calc_dists (x_test, k=1):
  dists = []
  for i in range(len(x_train)):
    x = x_train[i]
    dists.append([euclid_dist(x_test, x), y_train[i]])
  return sorted(dists, key=lambda d: d[0])[:k]

def knn_classify (x, k=1):
  dists = calc_dists(x, k)
  votes = [0, 0]
  for y in dists:
    votes[int(y[1])] += 1

  return votes.index(max(votes))

k_tests = [1, 5, 10, 20]

results = []
for k in k_tests:
  y_preds = []
  for x, y in zip(x_test, y_test):
    y_pred = knn_classify(x, k)
    y_preds.append(y_pred)
    # print(x, y_pred, y, "ok!" if y_pred == y else "fail!")

  score = accuracy_score(y_preds, y_test)
  results.append(score)
  print("Accuracy(k=%s) = %s" % (k, score))

print("Average Accuracy = %s" % np.average(results))
