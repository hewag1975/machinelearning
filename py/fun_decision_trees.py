
import numpy as np

def entropy(y):
    
    e = 0.
    if p > 0 and p < 1:
      p = sum(y) / len(y)
      e = -p1 * np.log2(p1) - (1 - p1)*np.log2(1-p1)
        
    return e

y = np.array([1,1,0,0,1,0,0,1,1,0])
entropy(y)


def split_dataset(X, node_indices, feature):
    l = []
    r = []
    
    for i, x in enumerate(node_indices):
      if X[x, feature] == 1:
        l.append(x)
      else:
        r.append(x)
        
    return l, r
  
X = np.array([
  [1,1,1]
  , [1,0,1]
  , [1,0,0]
  , [1,0,0]
  , [1,1,1]
  , [0,1,1]
  , [0,0,0]
  , [1,0,1]
  , [0,1,0]
  , [1,0,0]
])

split_dataset(X, range(9), 0)

def compute_information_gain(X, y, node_indices, feature):
    left_indices, right_indices = split_dataset(X, node_indices, feature)
    
    # Some useful variables
    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]
    
    # You need to return the following variables correctly
    information_gain = 0
    
    h_node = entropy(y_node)
    h_left = entropy(y_left)
    h_right = entropy(y_right)
    information_gain = h_node - (len(y_left) / len(y_node) * h_left + len(y_right) / len(y_node) * h_right)
    
    return information_gain

def get_best_split(X, y, node_indices):   
    num_features = X.shape[1]
    best_feature = -1
    max_gain = 0
    
    for feature in range(num_features): 
       info_gain = compute_information_gain(X, y, node_indices, feature)
       if info_gain > max_gain:  
           max_gain = info_gain
           best_feature = feature
   
    return best_featureX

y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
y = np.array([1, 0, 0, 1, 1, 1, 1, 1, 1, 1])
get_best_split(X, y, range(10))

def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth):
    """
    Build a tree using the recursive algorithm that split the dataset into 2 subgroups at each node.
    This function just prints the tree.
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
        branch_name (string):   Name of the branch. ['Root', 'Left', 'Right']
        max_depth (int):        Max depth of the resulting tree. 
        current_depth (int):    Current depth. Parameter used during recursive call.
   
    """ 

    # Maximum depth reached - stop splitting
    if current_depth == max_depth:
        formatting = " "*current_depth + "-"*current_depth
        print(formatting, "%s leaf node with indices" % branch_name, node_indices)
        return
   
    # Otherwise, get best split and split the data
    # Get the best feature and threshold at this node
    best_feature = get_best_split(X, y, node_indices) 
    
    formatting = "-"*current_depth
    print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))
    
    # Split the dataset at the best feature
    left_indices, right_indices = split_dataset(X, node_indices, best_feature)
    tree.append((left_indices, right_indices, best_feature))
    
    # continue splitting the left and the right child. Increment current depth
    build_tree_recursive(X, y, left_indices, "Left", max_depth, current_depth+1)
    build_tree_recursive(X, y, right_indices, "Right", max_depth, current_depth+1)
