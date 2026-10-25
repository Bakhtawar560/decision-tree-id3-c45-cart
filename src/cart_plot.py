import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = pd.read_csv("data/play_tennis.csv")

X = pd.get_dummies(df.drop("Decision", axis=1))
y = df["Decision"]

cart = DecisionTreeClassifier(criterion="gini", random_state=42)
cart.fit(X, y)

plt.figure(figsize=(14, 7))
plot_tree(cart, feature_names=X.columns, class_names=list(cart.classes_), filled=True)
plt.savefig("cart_tree.png", dpi=150, bbox_inches="tight")
print("Saved cart_tree.png")