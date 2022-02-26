"""
This is decision tree built from scratch. Optimization supports Gini and Entropy. 
Note the random_forest module is built upon this module
"""

from scipy import stats
import pandas as pd
import numpy as np



def loss(y, loss_func='gini'):
    _, counts = np.unique(y, return_counts=True)
    prop = counts / np.sum(counts)
    if loss_func == 'entropy':
        return np.float64(-np.dot(prop, np.log2(prop)))
    elif loss_func == 'gini':
        return np.float64(np.dot(prop, 1 - prop))
    elif loss_func == 'mse':
        return np.float64(np.sum(y - np.mean(y)) ** 2 / y.shape[0])


class Node:
    def __init__(self, indices, parent=None):
        self.indices = indices
        self.left = None
        self.right = None
        self.split_feature = None
        self.split_value = None

        if parent:
            self.depth = parent.depth + 1
            self.X = parent.X
            self.y = parent.y
            self.label = stats.mode(self.y[indices])[0][0]
            self.mean = np.mean(self.y[indices])


def greedy_search(node, loss_func='gini', max_features=None):
    _, n_col = node.X.shape
    feature_idx = np.random.choice(n_col, np.minimum(n_col, max_features),
                                   replace=False) if max_features else np.arange(0, n_col)
    best_loss = np.inf
    best_col, best_val = None, None
    for i in feature_idx:
        X_i = node.X[node.indices, i]
        unique_values = np.unique(X_i)
        for val in unique_values:
            idx_left = node.indices[X_i <= val]
            idx_right = node.indices[X_i > val]
            if len(idx_left) == 0 or len(idx_right) == 0:
                continue
            loss_left = loss(node.y[idx_left], loss_func=loss_func)
            loss_right = loss(node.y[idx_right], loss_func=loss_func)
            len_left, len_right = idx_left.shape[0], idx_right.shape[0]
            loss_total = (len_left * loss_left + len_right * loss_right) / (len_left + len_right)
            if loss_total < best_loss:
                best_loss = loss_total
                best_col = i
                best_val = val
    return best_loss, best_col, best_val


class DecisionTree:
    '''
    The DecisionTree object contains fit and predict methods for decision tree algorithm.

    Parameters
    ----------

    loss_func : {"gini", "entropy","mse"}, default="gini"
        The loss function to greedily search at each node.  
        Use "gini" or "entropy" when the job is classification.
        Use "mse" when regression
        
    min_samples_split : int, default=5
        The minimum number of samples required to split an internal node
        
    max_depth : int, default=10
        The maximum depth a tree can grow.
    
    max_features : int, default=None
        The number of features considered at each split. 
        If None, then use all of the features. Otherwise, choose a subset of features randomly.
        
    random_state : int, default=None
        Whether to control the randomness of the algorithm (such as to control max_features)
        
        
    '''
    
    def __init__(self, loss_func='gini', min_samples_split=5,
                 max_depth=None, max_features=None, random_state=None):
        self.loss_func = loss_func
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.root = Node(np.arange(0, X.shape[0]), None)
        self.root.depth = 0
        self.root.X = X
        self.root.y = y
        self._grow_tree(self.root)
        return self

    def _grow_tree(self, node):
        if len(np.unique(node.y)) == 1 or node.depth == self.max_depth or len(node.indices) <= self.min_samples_split:
            return
        np.random.seed(self.random_state)
        cost, split_feature, split_value = greedy_search(node, self.loss_func, self.max_features)
        if np.isinf(cost):
            return
        test = node.X[node.indices, split_feature] <= split_value
        node.split_feature = split_feature
        node.split_value = split_value
        left = Node(node.indices[test], node)
        right = Node(node.indices[np.logical_not(test)], node)

        self._grow_tree(left)
        self._grow_tree(right)
        node.left = left
        node.right = right

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            node = self.root
            while node.left:
                if x[node.split_feature] <= node.split_value:
                    node = node.left
                else:
                    node = node.right
            y_pred[i] = node.mean if self.loss_func == 'mse' else node.label
        return y_pred


if __name__ == "__main__":
    df = pd.read_csv('titanic.csv', usecols=['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])
    df.isna().sum()

    y = df['Survived']
    X = df.drop('Survived', axis=1)
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    from sklearn.impute import SimpleImputer

    imp = SimpleImputer(strategy='mean')
    X_train[['Age']] = imp.fit_transform(X_train[['Age']])

    imp2 = SimpleImputer(strategy='most_frequent')
    X_train[['Embarked']] = imp2.fit_transform(X_train[['Embarked']])

    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import make_column_transformer

    ohe = OneHotEncoder()
    ct = make_column_transformer((ohe, ['Sex', 'Embarked']), remainder='passthrough')
    X_train_vect = ct.fit_transform(X_train)
    y_train = y_train.values

    X_test[['Age']] = imp.transform(X_test[['Age']])
    X_test[['Embarked']] = imp2.transform(X_test[['Embarked']])
    X_test_vect = ct.transform(X_test)
    y_test = y_test.values
    model = DecisionTree(loss_func='entropy', min_samples_split=30, max_depth=4, max_features=7, random_state=4)
    model.fit(X_train_vect, y_train)

    y_pred = model.predict(X_test_vect)

    accuracy = sum(y_pred == y_test) / len(y_test)
    print("Accuracy:", accuracy)

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

if __name__ == "__main__":
    # Imports
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error


    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy


    data = datasets.load_diabetes()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    clf = DecisionTree(loss_func='MSE')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = mean_squared_error(y_test, y_pred)

    print("MSE:", acc)
