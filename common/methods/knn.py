import csv
import math
import random
import matplotlib.pyplot as plt 

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
# 2. KATEGORİK VERİLERİ SAYISALLAŞTIRMA (Label Encoding)
# -------------------------
def label_encode_column(data, column):
    unique_vals = list({row[column] for row in data})
    mapping = {val: i for i, val in enumerate(unique_vals)}
    for row in data:
        row[column] = mapping[row[column]]
    return mapping

# -------------------------
# 3. SAYIYA ÇEVRİLEBİLENLERİ ÇEVİRME
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
# 4. ÖKLİD UZAKLIĞI
# -------------------------
def euclidean(a, b):
    return math.sqrt(sum((a[i] - b[i])**2 for i in range(len(a))))

# -------------------------
# 5. KNN
# -------------------------
def knn_predict(train, test_row, k):
    distances = []
    for row in train:
        dist = euclidean(row["features"], test_row["features"])
        distances.append((dist, row["label"]))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]

    # çoğunluk oyu
    votes = {}
    for _, label in neighbors:
        votes[label] = votes.get(label, 0) + 1

    return max(votes, key=votes.get)

# -------------------------
# Accuracy
# -------------------------
def accuracy(y_true, y_pred):
    correct = sum(1 for i in range(len(y_true)) if y_true[i] == y_pred[i])
    return correct / len(y_true)

# -------------------------
# Precision
# -------------------------
def precision(y_true, y_pred):
    tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
    fp = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
    return tp / (tp + fp) if (tp + fp) != 0 else 0

# -------------------------
# Recall
# -------------------------
def recall(y_true, y_pred):
    tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
    fn = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)
    return tp / (tp + fn) if (tp + fn) != 0 else 0

# -------------------------
# F1-Score
# -------------------------
def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) != 0 else 0

# -------------------------
# Confusion Matrix
# -------------------------
def confusion_matrix(y_true, y_pred):
    tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
    tn = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 0)
    fp = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
    fn = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)
    return [[tn, fp], [fn, tp]]

# -------------------------
# ROC-AUC
# -------------------------
def roc_auc(y_true, y_prob):
    # y_prob: 1 olma olasılıkları
    # Önce threshold değerlerine göre TPR ve FPR hesaplanacak
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

    # AUC: trapez yöntemiyle yaklaşık hesap
    auc = 0
    for i in range(1, len(tpr_list)):
        auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2
    return auc


# -------------------------
# ANA PROGRAM
# -------------------------

data = read_csv("../../dataset1 Smoking/smoking.csv")

# ID kolonunu at
if "ID" in data[0]:
    for row in data:
        row.pop("ID")

# SAYISAL OLMAYAN TÜM SÜTUNLARI LABEL-ENCODE ET
for col in data[0].keys():
    try:
        float(data[0][col])
    except:
        if col != "Smoking Status":  # hedef değişkeni sonra özel alacağız
            label_encode_column(data, col)

# numeric'e çevirmeyi tamamla
convert_numeric(data)

# -------------------------
# VERİYİ FEATURES + LABEL OLARAK AYIR
# -------------------------

dataset = []
for row in data:
    label = int(row["smoking"])  # hedef değişken
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

# -------------------------
# KNN ÇALIŞTIR
# -------------------------
k = 5
predictions = []

for row in test:
    predictions.append(knn_predict(train, row, k))

# ROC-AUC için 1 olma olasılığı (komşu ortalaması)
y_prob = []
for row in test:
    distances = []
    for tr in train:
        dist = euclidean(tr["features"], row["features"])
        distances.append((dist, tr["label"]))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    prob = sum(label for _, label in neighbors) / k
    y_prob.append(prob)

y_true = [row["label"] for row in test]

# -------------------------
# METRİKLERİ YAZDIR
# -------------------------
print("Accuracy:", accuracy(y_true, predictions))
print("Precision:", precision(y_true, predictions))
print("Recall:", recall(y_true, predictions))
print("F1-Score:", f1_score(y_true, predictions))
print("Confusion Matrix:", confusion_matrix(y_true, predictions))
print("ROC-AUC:", roc_auc(y_true, y_prob))


# ===========================
# 🔥 ROC AUC GRAFİĞİ (EKLENEN KISIM)
# ===========================

# ROC için threshold'lara göre değer tekrar hesaplanır
thresholds = sorted(set(y_prob), reverse=True)
tpr_list = []
fpr_list = []

for thresh in thresholds:
    y_pred = [1 if p >= thresh else 0 for p in y_prob]

    tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
    fn = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)
    fp = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
    tn = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 0)

    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0   # True Positive Rate
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0   # False Positive Rate

    tpr_list.append(tpr)
    fpr_list.append(fpr)

# -------------------------
# GRAFİK ÇİZİMİ
# -------------------------
plt.figure(figsize=(6,5))
plt.plot(fpr_list, tpr_list, marker='o', label=f'KNN ROC Curve (AUC={roc_auc(y_true,y_prob):.3f})')
plt.plot([0,1], [0,1], 'r--', label="Rastgele Tahmin (Random Baseline)")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - KNN")
plt.legend()
plt.grid(True)
plt.show()