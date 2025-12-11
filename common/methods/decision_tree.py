from graphviz import Digraph
import random
from common.evaluation_metrics import calculate_metrics
from common.split_dataset import train_test_split

def gini_index(groups, classes):
    n_instances = sum([len(group) for group in groups])
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = sum(1 for row in group if row == class_val) / size
            score += p ** 2
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
        for row in dataset:
            groups = test_split(index, row["features"][index], dataset)
            gini = gini_index([[r["label"] for r in group] for group in groups], class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row["features"][index], gini, groups
    return {"index": b_index, "value": b_value, "groups": b_groups}


def to_terminal(group):
    labels = [row["label"] for row in group]
    return max(set(labels), key=labels.count)

def split(node, max_depth, min_size, depth):
    left, right = node["groups"]
    node.pop("groups")
    if not left or not right:
        node["left"] = node["right"] = to_terminal(left + right)
        return
    if depth >= max_depth:
        node["left"], node["right"] = to_terminal(left), to_terminal(right)
        return
    if len(left) <= min_size:
        node["left"] = to_terminal(left)
    else:
        node["left"] = get_best_split(left)
        split(node["left"], max_depth, min_size, depth + 1)
    if len(right) <= min_size:
        node["right"] = to_terminal(right)
    else:
        node["right"] = get_best_split(right)
        split(node["right"], max_depth, min_size, depth + 1)

def build_tree(train, max_depth, min_size):
    root = get_best_split(train)
    split(root, max_depth, min_size, 1)
    return root


def predict(node, row):
    if row["features"][node["index"]] < node["value"]:
        if isinstance(node["left"], dict):
            return predict(node["left"], row)
        else:
            return node["left"]
    else:
        if isinstance(node["right"], dict):
            return predict(node["right"], row)
        else:
            return node["right"]

 
def add_nodes_edges(tree, feature_labels, dot=None, parent=None, edge_label=""):
    if dot is None:
        dot = Digraph()

    if isinstance(tree, dict):
        feature_name = feature_labels.get(tree['index'], f"X{tree['index']}")
        label = f"{feature_name} < {tree['value']:.2f}"

        dot.node(str(id(tree)), label)

        if parent:
            dot.edge(str(id(parent)), str(id(tree)), label=edge_label)

        add_nodes_edges(tree["left"], feature_labels, dot, tree, "True")
        add_nodes_edges(tree["right"], feature_labels, dot, tree, "False")

    else:
        label = f"Leaf: {tree}"
        leaf_id = str(id(tree) + random.randint(0,1000))
        dot.node(leaf_id, label, shape="box")
        if parent:
            dot.edge(str(id(parent)), leaf_id, label=edge_label)
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

    dot = add_nodes_edges(tree, feature_labels)
    dot.render("decision_tree", format="png", cleanup=True)
    print("Decision tree görseli 'decision_tree.png' olarak kaydedildi.") 

    print("\n--- DECISION TREE METRICS ---")
    calculate_metrics(y_true, predictions)