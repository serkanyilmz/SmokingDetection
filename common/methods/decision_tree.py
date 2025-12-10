import csv
import random
from graphviz import Digraph

# -------------------------
# 1. CSV VERİSİNİ OKUMA
# -------------------------
def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = []
        for row in reader:
            data.append(row)
        return data

# -------------------------
# 2. KATEGORİK VERİLERİ SAYISALLAŞTIRMA
# -------------------------
def label_encode_column(data, column):
    unique_vals = list({row[column] for row in data})
    mapping = {val: i for i, val in enumerate(unique_vals)}
    for row in data:
        row[column] = mapping[row[column]]
    return mapping

# -------------------------
# 3. SAYIYA ÇEVİRME
# -------------------------
def convert_numeric(data):
    for row in data:
        for key in row:
            try:
                row[key] = float(row[key])
            except:
                pass

# -------------------------
# 4. ÖZELLİK NORMALİZASYONU (0-1)
# -------------------------
def normalize_features(dataset):
    n_features = len(dataset[0]["features"])
    mins = [min(row["features"][i] for row in dataset) for i in range(n_features)]
    maxs = [max(row["features"][i] for row in dataset) for i in range(n_features)]

    for row in dataset:
        row["features"] = [
            (row["features"][i] - mins[i]) / (maxs[i] - mins[i] + 1e-8)
            for i in range(n_features)
        ]

# -------------------------
# 4. GINI HESABI
# -------------------------
def gini_index(groups, classes):
    # groups: [sol, sağ] list of lists of labels
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

# -------------------------
# 5. EN İYİ BÖLME
# -------------------------
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

# -------------------------
# 6. Düğüm oluşturma (rekürsif)
# -------------------------
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

# -------------------------
# 7. Karar Ağacı oluşturma
# -------------------------
def build_tree(train, max_depth, min_size):
    root = get_best_split(train)
    split(root, max_depth, min_size, 1)
    return root

# -------------------------
# 8. Tahmin yapma
# -------------------------
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

# -------------------------
# 9. Tahmin olasılıkları (ROC-AUC)
# -------------------------
def predict_prob(node, row):
    # leaf node: return 0 veya 1 olasılık
    if row["features"][node["index"]] < node["value"]:
        if isinstance(node["left"], dict):
            return predict_prob(node["left"], row)
        else:
            # leaf node olasılık: sadece sınıf 0 veya 1
            return 1.0 if node["left"] == 1 else 0.0
    else:
        if isinstance(node["right"], dict):
            return predict_prob(node["right"], row)
        else:
            return 1.0 if node["right"] == 1 else 0.0

# -------------------------
# 10. METRİKLER (önceki koddan)
# -------------------------
def accuracy(y_true, y_pred):
    correct = sum(1 for i in range(len(y_true)) if y_true[i] == y_pred[i])
    return correct / len(y_true)

def precision(y_true, y_pred):
    tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
    fp = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
    return tp / (tp + fp) if (tp + fp) != 0 else 0

def recall(y_true, y_pred):
    tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
    fn = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)
    return tp / (tp + fn) if (tp + fn) != 0 else 0

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) != 0 else 0

def confusion_matrix(y_true, y_pred):
    tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
    tn = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 0)
    fp = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
    fn = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)
    return [[tn, fp], [fn, tp]]

def roc_auc(y_true, y_prob):
    thresholds = sorted(set(y_prob), reverse=True)
    tpr_list = []
    fpr_list = []

    for thresh in thresholds:
        y_pred = [1 if p >= thresh else 0 for p in y_prob]
        tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
        fn = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)
        fp = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
        tn = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 0)

        tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) != 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    auc = 0
    for i in range(1, len(tpr_list)):
        auc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2
    return auc


# -------------------------
# 12. KARAR AĞACI GÖRSELLEŞTİRME
# -------------------------
def add_nodes_edges(tree, dot=None, parent=None, edge_label=""):
    if dot is None:
        dot = Digraph()
    if isinstance(tree, dict):
        label = f"X{tree['index']} < {tree['value']:.2f}"
        dot.node(str(id(tree)), label)
        if parent:
            dot.edge(str(id(parent)), str(id(tree)), label=edge_label)
        add_nodes_edges(tree["left"], dot, tree, "True")
        add_nodes_edges(tree["right"], dot, tree, "False")
    else:
        label = f"Leaf: {tree}"
        leaf_id = str(id(tree) + random.randint(0,1000))
        dot.node(leaf_id, label, shape="box")
        if parent:
            dot.edge(str(id(parent)), leaf_id, label=edge_label)
    return dot


# -------------------------
# 11. ANA PROGRAM
# -------------------------
data = read_csv("../../dataset1 Smoking/smoking.csv")

# ID kolonunu çıkar
if "ID" in data[0]:
    for row in data:
        row.pop("ID")

# Label encode kategorik
for col in data[0].keys():
    try:
        float(data[0][col])
    except:
        if col != "smoking":
            label_encode_column(data, col)

convert_numeric(data)

# VERİYİ FEATURES + LABEL AYIR
dataset = []
for row in data:
    label = int(row["smoking"])
    row.pop("smoking")
    features = list(row.values())
    dataset.append({"features": features, "label": label})

# Normalize
normalize_features(dataset)

# Train/test böl
random.shuffle(dataset)
split_idx = int(len(dataset) * 0.8)
train = dataset[:split_idx]
test = dataset[split_idx:]

# Decision Tree oluştur
max_depth = 5
min_size = 5
tree = build_tree(train, max_depth, min_size)

# Tahmin
y_true = [row["label"] for row in test]
predictions = [predict(tree, row) for row in test]
y_prob = [predict_prob(tree, row) for row in test]

# METRİKLERİ YAZDIR
print("Accuracy:", accuracy(y_true, predictions))
print("Precision:", precision(y_true, predictions))
print("Recall:", recall(y_true, predictions))
print("F1-Score:", f1_score(y_true, predictions))
print("Confusion Matrix:", confusion_matrix(y_true, predictions))
print("ROC-AUC:", roc_auc(y_true, y_prob))

# AĞACI ÇİZ
dot = add_nodes_edges(tree)
dot.render("decision_tree", format="png", cleanup=True)
print("Decision tree görseli 'decision_tree.png' olarak kaydedildi.")