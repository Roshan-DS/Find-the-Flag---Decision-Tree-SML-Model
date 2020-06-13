import codecademylib3_seaborn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


# 1
flags = pd.read_csv("flags.csv", header = 0)
print(flags)

# 2
print(flags.columns)

# 3
print(flags.head)

# 4
labels = flags['Landmass']

# 5
data = flags[["Red", "Green", "Blue", "Gold",
 "White", "Black", "Orange",
 "Circles",
"Crosses","Saltires","Quarters","Sunstars",
"Crescent","Triangle"]]

# 6
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state = 1)

scores = []

for i in range(1,21):
  tree = DecisionTreeClassifier(random_state = 1, max_depth = i)


  train_model = tree.fit(train_data, train_labels)

  
  scores.append(tree.score(test_data,test_labels))

print(scores)

  
# 12
plt.plot(range(1,21), scores)
plt.show()

