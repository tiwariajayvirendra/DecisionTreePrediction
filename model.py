from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import pickle

# load dataSet 
iris = load_iris()

X=iris.data
y= iris.target

#Train Model 

model = DecisionTreeClassifier()
model.fit(X,y)

# Save Model
with open("decision_tree_model.pkl",'wb')as f:
    pickle.dump(model,f)
print("Model Saved Successfully!")