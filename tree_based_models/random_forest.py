##################################
from tree_based_models.decision_tree import DecisionTree
import numpy as np
from numpy.random import randint
from numpy import sqrt
from scipy.stats import mode
from numpy import log2

def Bootstrap(y,n_estimators): 
    num_instances = y.shape[0]
    indices = randint(0,high=num_instances,size=(n_estimators,num_instances))
    return indices

#################
class RandomForest:
    def __init__(self,n_estimators = 100, criterion = 'gini', max_depth = None, min_samples_split = 5, 
                 max_features = 'sqrt', random_state = None):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.estimators = []
    def fit(self,X,y):
        self.estimators = []
        if self.max_features == 'sqrt':
            n_features = round(sqrt(X.shape[1]))
        elif self.max_features == 'log2':
            n_features = round(log2(X.shape[1]))
        elif self.max_features == None:
            n_features = X.shape[1]
        np.random.seed(self.random_state)
        indices_array = Bootstrap(y,self.n_estimators)
        X_samples,y_samples = X[indices_array],y[indices_array] 
        for i in range(self.n_estimators):
            tree = DecisionTree(loss_func=self.criterion,max_depth = self.max_depth, 
                                min_samples_split = self.min_samples_split, 
                                max_features = n_features,random_state=i if self.random_state else None)
            tree.fit(X_samples[i],y_samples[i])
            self.estimators.append(tree)
    
    def predict(self,X):
        y_preds = np.array([tree.predict(X) for tree in self.estimators]) 
        y_preds = mode(y_preds)[0][0]
        return y_preds

    
if __name__ == "__main__":
    # Imports
    from sklearn import datasets
    from sklearn.model_selection import train_test_split


    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy


    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    clf = DecisionTree(max_depth=10)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy(y_test, y_pred)

    print("Accuracy:", acc)
