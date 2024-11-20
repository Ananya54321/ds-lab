# Import necessary libraries
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


data = load_iris()
x = data.data

y = data.target
feature_names = data.feature_names

model = LogisticRegression(max_iter=500)

selector = RFE(estimator=model, n_features_to_select=2)
selector.fit(x, y)

print("Feature ranking:", selector.ranking_)

plt.figure(figsize=(13, 5))
plt.barh(feature_names, selector.ranking_)
plt.xlabel("Feature Ranking")
plt.ylabel("Features")
plt.title("Feature Importance Ranking using RFE")
plt.show()
