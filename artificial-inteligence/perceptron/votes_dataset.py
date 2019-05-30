import pandas as pd
import statistics as stcs


def HouseVotes_DataSet(filename='house-votes-84.csv'):
  # read database
  dataset = pd.read_csv(filename)

  x = dataset.iloc[:, 1:].values
  y = dataset.iloc[:, 0].values

  n_samples, n_instances = x.shape
  print(n_samples, "x", n_instances)

  # replace class name with ints
  # democrat = 0, republican = 1
  for i in range(n_samples):
    if y[i] == 'democrat':
      y[i] = 0
    elif y[i] == 'republican':
      y[i] = 1
  y = y.astype(int)

  # replace value with ints
  # y = 1, n = 0
  for i in range(n_samples):
    for j in range(n_instances):
      if ('y' in x[i][j]):
        x[i][j] = 1
      elif ('n' in x[i][j]):
        x[i][j] = 0

  # find each class median
  medians = []
  for j in range(n_instances):
    acceptable = []
    for i in range(n_samples):
      if (x[i][j] == 1 or x[i][j] == 0):
        acceptable.append(x[i][j])
    med = stcs.median(acceptable)
    medians.append(int(med))

  # replace missing values with median
  for i in range(n_samples):
    for j in range(n_instances):
      if (x[i][j] != 1 and x[i][j] != 0):
        x[i][j] = medians[j]
  x = x.astype(int)

  return (x, y, dataset)
