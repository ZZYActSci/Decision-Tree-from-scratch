import pandas as pd
import numpy as np
from scipy import stats


def is_pure(y):
    return True if len(np.unique(y)) == 1 else False


def vote_class(y):
    return stats.mode(y)[0][0]


def entropy(y_left, y_right=None):
    if y_right is None:
        _, counts = np.unique(y_left, return_counts=True)
        prop = counts / np.sum(counts)
        return np.float64(-np.dot(prop, np.log2(prop)))
    else:
        n_l, n_r = len(y_left), len(y_right)
        return n_l / (n_l + n_r) * entropy(y_left) + n_r / (n_l + n_r) * entropy(y_right)


def gini(y_left, y_right=None):
    if y_right is None:
        _, counts = np.unique(y_left, return_counts=True)
        prop = counts / np.sum(counts)
        return np.float64(np.dot(prop, 1 - prop))
    else:
        n_l, n_r = len(y_left), len(y_right)
        return n_l / (n_l + n_r) * gini(y_left) + n_r / (n_l + n_r) * gini(y_right)


def calculate_loss(y_left, y_right=None, loss_func='gini'):  ### add loss_func into decision tree
    if loss_func == 'gini':
        return gini(y_left, y_right)
    elif loss_func == 'entropy':
        return entropy(y_left, y_right)
    else:
        print("Your loss function '{}' is not provided, please try either 'gini' or 'entropy'".format(loss_func))


def get_potential_splits(X, max_features=None):
    _, num_features = X.shape
    if max_features is not None:
        features_idx = np.random.choice(num_features, np.minimum(num_features, max_features), replace=False)
    else:
        features_idx = np.arange(0, num_features)
    potential_splits = {}
    for i in features_idx:
        unique_values = np.unique(X[:, i])
        potential_splits[i] = unique_values
    return potential_splits


def split_data(X, y, split_col, split_val):
    feature_col = X[:, split_col]
    if isinstance(split_val, str):
        idx_left = feature_col == split_val
        idx_right = feature_col != split_val
    else:
        idx_left = feature_col <= split_val
        idx_right = feature_col > split_val

    return X[idx_left], X[idx_right], y[idx_left], y[idx_right]


def determine_best_split(X, y, potential_splits, loss_func='gini'):
    loss_init = np.inf
    for idx in potential_splits:
        for val in potential_splits[idx]:
            _, _, y_left, y_right = split_data(X, y, split_col=idx, split_val=val)
            loss = calculate_loss(y_left, y_right, loss_func=loss_func)
            if loss < loss_init:
                loss_init = loss
                best_col = idx
                best_val = val
    return best_col, best_val


def decision_tree(X, y, depth=0, loss_func='gini', min_samples_split=5,
                  max_depth=100, max_features=None, *, feature_names=None):
    if depth == 0:
        if ((not isinstance(X, np.ndarray))
                or (not isinstance(y, np.ndarray))):
            X, y = np.array(X), np.array(y)
    if is_pure(y) or len(y) <= min_samples_split or (depth >= max_depth):
        return vote_class(y)
    else:
        depth += 1
        potential_splits = get_potential_splits(X, max_features=max_features)
        best_col, best_val = determine_best_split(X, y, potential_splits, loss_func=loss_func)
        X_left, X_right, y_left, y_right = split_data(X, y, best_col, best_val)
        if len(X_left) == 0 or len(X_right) == 0:
            return vote_class(y)

        if isinstance(best_val, str):
            question = "{} == {}".format(best_col if feature_names is None
                                         else feature_names[best_col], best_val)
        else:
            question = "{} <= {}".format(best_col if feature_names is None
                                         else feature_names[best_col], best_val)

        sub_tree = {question: []}

        yes_answer = decision_tree(X_left, y_left, depth, loss_func, min_samples_split,
                                   max_depth, max_features, feature_names=feature_names)
        no_answer = decision_tree(X_right, y_right, depth, loss_func, min_samples_split,
                                  max_depth, max_features, feature_names=feature_names)

        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)

        return sub_tree

    def traverse_tree(x, tree, feature_names=None):
        if not isinstance(tree, dict):
            return tree
        else:
            question = list(tree.keys())[0]
            name, sign, value = question.split()
            name = list(feature_names).index(name) if feature_names is not None else name
            if sign == '==':
                if x[int(name)] == value:
                    tree = list(tree[question])[0]
                else:
                    tree = list(tree[question])[1]
            else:
                if x[int(name)] <= float(value):
                    tree = list(tree[question])[0]
                else:
                    tree = list(tree[question])[1]
            return traverse_tree(x, tree, feature_names)

    def predict(X, tree, feature_names=None):
        if isinstance(X, pd.DataFrame):
            X = np.array(X)
        return [traverse_tree(x, tree, feature_names) for x in X]


