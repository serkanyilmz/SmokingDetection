from graphviz import Digraph
from common.functions.evaluation_metrics import calculate_metrics
from common.functions.split_dataset import train_test_split

class Node:
    def __init__(self, index=None, value=None, left=None, right=None, label=None):
        self.index = index
        self.value = value
        self.left = left
        self.right = right
        self.label = label

def gini_index(groups, classes):
    n_instances = sum(len(group) for group in groups)
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        counts = {}
        for val in group:
            counts[val] = counts.get(val, 0) + 1
        score = sum((counts.get(cls, 0)/size)**2 for cls in classes)
        gini += (1 - score) * (size / n_instances)
    return gini


def test_split(index, value, dataset):
    left, right = [], []
    for row in dataset:
        if row["features"][index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

def get_best_split(dataset):
    class_values = list(set(row["label"] for row in dataset))
    b_index, b_value, b_score, b_groups = None, None, float('inf'), None
    n_features = len(dataset[0]["features"])

    for index in range(n_features):
        feature_values = sorted(set(row["features"][index] for row in dataset))
        split_candidates = [(feature_values[i] + feature_values[i+1])/2 for i in range(len(feature_values)-1)]

        for value in split_candidates:
            groups = test_split(index, value, dataset)
            gini = gini_index([[r["label"] for r in g] for g in groups], class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, value, gini, groups

    return Node(index=b_index, value=b_value, left=b_groups[0], right=b_groups[1])

def to_terminal(group):
    counts = {}
    for row in group:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    return max(counts, key=counts.get)

def split(node, max_depth, min_size, depth):
    left, right = node.left, node.right
    if not left or not right:
        node.label = to_terminal(left + right)
        node.left = node.right = None
        return
    if depth >= max_depth:
        node.left = Node(label=to_terminal(left))
        node.right = Node(label=to_terminal(right))
        return
    if len(left) <= min_size:
        node.left = Node(label=to_terminal(left))
    else:
        node.left = get_best_split(left)
        split(node.left, max_depth, min_size, depth + 1)
    if len(right) <= min_size:
        node.right = Node(label=to_terminal(right))
    else:
        node.right = get_best_split(right)
        split(node.right, max_depth, min_size, depth + 1)

def build_tree(train, max_depth, min_size):
    root = get_best_split(train)
    split(root, max_depth, min_size, 1)
    return root

def predict(node, row):
    while node.label is None:
        if row["features"][node.index] < node.value:
            node = node.left
        else:
            node = node.right
    return node.label

def add_nodes_edges(tree, feature_labels, dot=None, parent_id=None, edge_label="", node_counter=[0]):
    if dot is None:
        dot = Digraph()
    current_id = node_counter[0]
    node_counter[0] += 1

    if tree.label is None:
        feature_name = feature_labels.get(tree.index, f"X{tree.index}")
        label = f"{feature_name} < {tree.value:.2f}"
        dot.node(str(current_id), label)
        if parent_id is not None:
            dot.edge(str(parent_id), str(current_id), label=edge_label)
        add_nodes_edges(tree.left, feature_labels, dot, current_id, "True", node_counter)
        add_nodes_edges(tree.right, feature_labels, dot, current_id, "False", node_counter)
    else:
        dot.node(str(current_id), f"Leaf: {tree.label}", shape="box")
        if parent_id is not None:
            dot.edge(str(parent_id), str(current_id), label=edge_label)
    return dot

def set_header_labels(headers):
    if "ID" in headers: headers.remove("ID")
    if "DRK_YN" in headers: headers.remove("DRK_YN")

    feature_labels = {}
    for i, header in enumerate(headers[:-1]):  
        feature_labels[i] = header
    return feature_labels


def decision_tree(dataset, headers):

    feature_labels = set_header_labels(headers.copy())

    x = [row[:-1] for row in dataset]
    y = [row[-1] for row in dataset]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    train = [{"features": x_train[i], "label": y_train[i]} for i in range(len(x_train))]
    test = [{"features": x_test[i],  "label": y_test[i]} for i in range(len(x_test))]

    max_depth = 5
    min_size = 5
    tree = build_tree(train, max_depth, min_size)

    y_true = [row["label"] for row in test]
    predictions = [predict(tree, row) for row in test]

    print("\n--- DECISION TREE METRICS ---")
    calculate_metrics(y_true, predictions)

    dot = add_nodes_edges(tree, feature_labels)
    dot.render("decision_tree", format="png", cleanup=True)
    print("Decision tree görseli 'decision_tree.png' olarak kaydedildi.")