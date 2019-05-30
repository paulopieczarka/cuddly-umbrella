from votes_dataset import HouseVotes_DataSet
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from perceptron import Perceptron, PerceptronPocketRatchet

# open and preprocess house-votes dataset
xs, ys, dataset = HouseVotes_DataSet()

# train test split
x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size = 0.25, stratify = ys, random_state = 1)

def run_perceptron_test ():
  print("\n-> POCKET PERCEPTRON!!")
  perceptron = Perceptron()
  perceptron.train(x_train, y_train, 10000, .3)

  y_pred_test = perceptron.predict(x_test)
  print("-> Model accuracy = %s" % accuracy_score(y_pred_test, y_test))

def run_pocket_perceptron_test ():
  perceptron = PerceptronPocketRatchet()
  perceptron.train(x_train, y_train, 10000)

# test
# run_perceptron_test()
run_pocket_perceptron_test()
