import numpy as np
np.random.seed(42)

chi_table = {0.01  : 6.635,
             0.005 : 7.879,
             0.001 : 10.828,
             0.0005 : 12.116,
             0.0001 : 15.140,
             0.00001: 19.511}



def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the gini impurity of the dataset.    
    """
    if np.size(data) == 0:
        return 0
    labels = data[:,-1]
    prob = np.sum(labels) / np.size(labels)
    gini = 1 - (prob ** 2 + (1 - prob) ** 2)
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns the entropy of the dataset.    
    """
    if np.size(data) == 0:
        return 0
    labels = data[:, -1]
    prob = np.sum(labels) / np.size (labels)
    if prob == 1 or prob == 0:
        return 0
    entropy = -(prob * np.log2([prob]) + (1 - prob) * np.log2([1 - prob]))
    entropy = entropy[0]
    return entropy

class DecisionNode:

    # This class will hold everything you require to construct a decision tree.
    # The structure of this class is up to you. However, you need to support basic 
    # functionality as described in the notebook. It is highly recommended that you 
    # first read and understand the entire exercise before diving into this class.
    
    def __init__(self, feature, value, data):
        self.feature = feature # column index of criteria being tested
        self.value = value # value necessary to get a true result
        self.children = []
        self.data = data
        
    def add_child(self, node):
        self.children.append(node)


def list_of_thresholds(data):
    """
    creates a matrix with the thresholds that we need to check. each column has the thresholds for the corresponding
    feature.

    Input:
    - data: the training dataset.

    Output: the matrix with the thresholds.
    """
    zeros = [np.zeros(np.size(data, 1))]
    sorted_data = np.sort(data, 0)
    data1 = np.concatenate((zeros, sorted_data), 0)
    data2 = np.concatenate((sorted_data, zeros), 0)
    avg = ((data1 + data2)/2)[1:np.size(sorted_data,0)]
    return avg

        
def find_best_threshold(data, thresholds, impurity):
    """
    for a specific feature finds the best threshold and it's gain

    Input:
    - data: the training dataset.
    - thresholds: a vector of thresholds
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.

    Output:
    - best_threshlod: the best thresholds found
    - max_gain: the gain of the thresholds found
    """
    max_gain = -1
    if (np.size(data) == 0):
        print ('data 0')
    current_impurity = impurity(data)
    best_threshold = -1
    
    for threshold in thresholds:
        relation = data[:,0] >= threshold
        bigger_data = data[relation,:]
        smaller_data = data[~relation,:]
        biggerThan = relation.sum()
        smallerThan = np.size(relation) - biggerThan
        gain = current_impurity - biggerThan / np.size(relation) * impurity(bigger_data) - smallerThan / np.size(relation) * impurity(smaller_data)
        if gain > max_gain:
            max_gain = gain
            best_threshold = threshold
    
    return best_threshold, max_gain

def find_best_feature(data, thresholds, impurity):
    """
    finds the best feature with the best threshold for the feature

    Input:
    - data: the training dataset.
    - thresholds: a thresholds matrix with the vector of thresholds for each feature
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.

    Output:
    - best_feature: the best feature found to split by
    - best_threshold: the best threshold to split by of the feature we found
    """
    best_feature = None
    max_gain = -1
    best_threshold = 0
    if len(thresholds.shape) == 2:
        for i in range(np.size(thresholds, 1)):
            threshold, gain = find_best_threshold(data[:,[i,-1]], thresholds[:,i], impurity)
            if gain > max_gain:
                best_feature = i
                max_gain = gain
                best_threshold = threshold
    else:
        for i in range(np.size(thresholds)):
            threshold, gain = find_best_threshold(data[:,[i,-1]], [thresholds[i]], impurity)
            if gain > max_gain:
                best_feature = i
                max_gain = gain
                best_threshold = threshold
       
    return best_feature, best_threshold
            
            
            
def split_data(data, threshold, feature):
    """
    splits the data to one data with feature bigger than threshold
    and one data with feature smaller than the threshold

    Input:
    - data: the training dataset.
    - threshold: the threshold we split by
    - feature: the feature we split by

    Output:
    - bigger_data: the instances of the data that their 'feature' feature is bigger than 'threshold'
    - smaller_data: the instances of the data that their 'feature' feature is smaller than 'threshold'
    """
    relation = data[:,feature] >= threshold
    bigger_data = data[relation]
    smaller_data = data[~relation]
    return bigger_data, smaller_data
    
def is_a_leaf(data):
    """
    checks weather a node with 'data' data is a leaf or not

    Input:
    - data: the dataset in a specific node

    Output: true if a leaf false otherwise
    """
    if np.size(data, 0) == np.sum(data[:,-1]):
        return True
    if np.sum(data[:,-1]) == 0:
        return True
    
    return False

def chi_square(node, data):
    """
    calculates the chi_square value of a specific node

    Input:
    - node: the node we calculates chi_square for
    - data: the dataset of 'node'

    Output: the chi square value of 'node'
    """
    chi = 0
    bigger_data,smaller_data = split_data(data, node.value, node.feature)
    d_bigger = np.size(bigger_data[:, -1])
    d_smaller = np.size(smaller_data[:, -1])
    p_true = np.sum(data[:, -1]) / np.size(data[:, -1])
    p_false = 1- p_true
    n_bigger = np.sum(bigger_data[:, -1])
    n_smaller = np.sum(smaller_data[:, -1])
    p_bigger = d_bigger - n_bigger
    p_smaller = d_smaller - n_smaller
    e0_bigger = d_bigger * p_false
    e1_bigger = d_bigger * p_true
    e0_smaller = d_smaller * p_false
    e1_smaller = d_smaller * p_true
    chi += ((p_bigger - e0_bigger) ** 2) / e0_bigger + ((n_bigger - e1_bigger) ** 2) / e1_bigger
    chi += ((p_smaller - e0_smaller) ** 2) / e0_smaller + ((n_smaller - e1_smaller) ** 2) / e1_smaller
    return chi
    
def build_tree(data, impurity, chi=0):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure. 

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.

    Output: the root node of the tree.
    """
    if len(data.shape) == 1:
        root = DecisionNode(data.size - 1, data[-1], data)
        return root
    if is_a_leaf(data):
        root = DecisionNode(np.size(data,1) - 1, data[0,-1], data)
        return root
    thresholds = list_of_thresholds(data[:,:-1])
    feature, value = find_best_feature(data, thresholds, impurity)
    root = DecisionNode(feature, value, data)
    if (chi_square(root, data) >= chi):
        biggerChild, smallerChild = split_data(data, value, feature)
        root.add_child(build_tree(biggerChild, impurity, chi))
        root.add_child(build_tree(smallerChild, impurity, chi))
    else:
        root.feature = np.size(data,1) - 1
        if np.sum(data[:, -1]) / np.size(data[:, -1]) >= 0.5:
            root.value = 1
        else:
            root.value = 0    
    return root

def predict(node, instance):
    """
    Predict a given instance using the decision tree

    Input:
    - root: the root of the decision tree.
    - instance: a row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.

    Output: the prediction of the instance.
    """
    if node.feature == np.size(instance) - 1:
        return node.value
    if instance[node.feature] >= node.value:
        return predict(node.children[0], instance)
    return predict(node.children[1], instance)

def calc_accuracy(node, dataset):
    """
    calculate the accuracy starting from some node of the decision tree using
    the given dataset.

    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated

    Output: the accuracy of the decision tree on the given dataset (%).
    """
    
    accuracy = 0.0
    for instance in dataset:
        prediction = predict(node, instance)
        if prediction == instance[-1]:
            accuracy += 1

    return accuracy / len(dataset)


def list_of_parents (node):
    """Return a list of all the parents who's children are leaves."""
    if node.children == []:
        return []
    if len(node.children[0].children) == 0 and len(node.children[1].children) == 0:
        return [node]
    elif len(node.children[0].children) == 0:
        return [node] + list_of_parents(node.children[1])
    elif len(node.children[1].children) == 0:
        return [node] + list_of_parents(node.children[0])
    return list_of_parents(node.children[0]) + list_of_parents(node.children[1])

def num_of_nodes (node):
    """ return the number of internal nodes in the tree of the node given """
    if node.children == []:
        return 0
    return num_of_nodes(node.children[0]) + num_of_nodes(node.children[1]) + 1

def post_pruning(node, data, accuracies, train_accuracies, numOfNodes):
    """
    removes nodes one by one from the tree 'node' untill we left only with the root.
    each removal of node is made by finding the best non-leaf node to remove by checking the accuracy of the 'data'
    data without the node.

    Input:
    - node: the root of the tree we're pruning
    - data: the test dataset
    - accuracies: the list of accuracies of the test dataset
    - train_accuracies: the list of accuracies of the train dataset
    - numOfNodes: the list of number of internal nodes in each prune

    Output: the lists of accuracies of test and training data and a list of the internal nodes in each prune
    """
    best_train_accuracy = 0
    best_accuracy = 0
    best_parent = None
    parents = list_of_parents(node)
    if parents == []:
        return accuracies, train_accuracies, numOfNodes
    for parent in parents:
        parentFeature = parent.feature
        parentValue = parent.value
        yesNoRelation = np.sum(parent.data[:,-1])/ np.size(parent.data[:,-1])
        parent.feature = np.size(parent.data,1) - 1
        if yesNoRelation >= 0.5:
            parent.value = 1
        else:
            parent.value = 0
        current_accuracy = calc_accuracy(node, data)
        train_accuracy = calc_accuracy(node, node.data)
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_parent = parent
            best_train_accuracy = train_accuracy
        parent.feature = parentFeature
        parent.value = parentValue
    yesNoRelation = np.sum(best_parent.data[:,-1])/ np.size(best_parent.data[:,-1])
    best_parent.feature = np.size(best_parent.data,1) - 1
    best_parent.children = []
    if yesNoRelation >= 0.5:
        best_parent.value = 1
    else:
        best_parent.value = 0
    current_numOfNodes = num_of_nodes(node)
        
    return post_pruning(node, data, np.append(accuracies,best_accuracy), np.append(train_accuracies, best_train_accuracy), np.append(numOfNodes,current_numOfNodes))

def print_tree(node, space=""):
    """
    prints the tree according to the example in the notebook

    Input:
    - node: a node in the decision tree

    This function has no return value
    """

    # Base case: we've reached a leaf
    if len(node.children) == 0 and node.value == 1:
        print (space + "leaf: [{"+str(node.value)+ ": " + str(np.sum(node.data[:,-1])) + "}]")
        return
    elif len(node.children) == 0:
        print (space + "leaf: [{"+str(node.value)+ ": " + str(np.size(node.data[:,-1]) - np.sum(node.data[:,-1]))+"}]")
        return

    # Print the question at this node
    print (space + "[" + str(node.feature) + " <= " + str(node.value) + "],")

    # Call this function recursively on the true branch
    print_tree(node.children[0], (space + "  "))
    # Call this function recursively on the false branch
    print_tree(node.children[1], (space + "  "))
    
    