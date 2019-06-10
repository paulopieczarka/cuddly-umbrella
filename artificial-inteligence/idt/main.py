from votes_dataset import HouseVotes_DataSet
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# open and preprocess house-votes dataset
xs, ys, dataset = HouseVotes_DataSet()

# train test split
x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size = 0.25, stratify = ys, random_state = 1)

print(x_train)
