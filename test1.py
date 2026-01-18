from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

data = load_iris()
X = data.data
y = data.target

model = Pipeline([('pipe', StandardScaler()),('log', DecisionTreeClassifier())])
X_train, X_test, y_train, y_test = train_test_split(X, y)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=data.target_names))

plot_tree(model['log'], feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.title("Decision Tree Trained on Iris Dataset")
plt.show()
