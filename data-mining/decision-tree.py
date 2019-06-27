import numpy as np
import math
from sklearn.metrics import accuracy_score
from votes_dataset import HouseVotes_DataSet
from sklearn.model_selection import train_test_split

# open and preprocess house-votes dataset
xs, ys, dataset = HouseVotes_DataSet()

# train test split
x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size = 0.25, stratify = ys, random_state = 1)

# print(x_train)

def gini_index (xs, ys):
  n_instances = sum([len(x) for x in xs])

  gini = 0
  for x in xs:
    size = len(x)

    if (size == 0):
      continue

    score = 0
    for y in ys:
      p = [row[-1] for row in x].count(y) / size
      score += p * p

    gini += (1.0 - score) * (size / n_instances)

  return gini

def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

def get_split(dataset):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset[0])-1):
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left)
		split(node['left'], max_depth, min_size, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth+1)

# Build a decision tree
def build_tree(train, max_depth, min_size):
	root = get_split(train)
	split(root, max_depth, min_size, 1)
	return root

def print_tree(node, depth=0):
	if isinstance(node, dict):
		print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
		print_tree(node['left'], depth+1)
		print_tree(node['right'], depth+1)
	else:
		print('%s[%s]' % ((depth*' ', node)))

# Make a prediction with a decision tree
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

def decision_tree(trainset, testset, max_depth, min_size):
	tree = build_tree(trainset, max_depth, min_size)
	predictions = list()
	for row in testset:
		prediction = predict(tree, row)
		predictions.append(prediction)
	return(predictions)

train_set = np.append(x_train, np.reshape(y_train, (len(y_train),1)), axis=1)
test_set = np.append(x_test, np.reshape(y_test, (len(y_test),1)), axis=1)

results = []
for i in range(10):
  y_pred = decision_tree(train_set, test_set, 10, 1)
  score = accuracy_score(y_pred, y_test)
  results.append(score)
  print("Accuracy(it=%s) = %s" % (i+1, score))

print("Average Accuracy = %s" % np.average(results))
