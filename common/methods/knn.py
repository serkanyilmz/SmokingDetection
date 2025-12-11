import math
from common.evaluation_metrics import calculate_metrics
from common.split_dataset import train_test_split

def euclidean(a, b):
    return math.sqrt(sum((a[i] - b[i])**2 for i in range(len(a))))

def knn_predict(train, test_row, k):
    distances = []
    for row in train:
        dist = euclidean(row["features"], test_row["features"])
        distances.append((dist, row["label"]))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]

    votes = {}
    for _, label in neighbors:
        votes[label] = votes.get(label, 0) + 1

    return max(votes, key=votes.get)


def knn(dataset, k=5):

    x = [row[:-1] for row in dataset]
    y = [row[-1] for row in dataset]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    train = [{"features": x_train[i], "label": y_train[i]} for i in range(len(x_train))]
    test = [{"features": x_test[i],  "label": y_test[i]} for i in range(len(x_test))]
    predictions = []
    y_prob = []

    for row in test:
        predictions.append(knn_predict(train, row, k))

        # ROC için Probability hesabı
        distances = []
        for tr in train:
            dist = euclidean(tr["features"], row["features"])
            distances.append((dist, tr["label"]))
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:k]

        prob = sum(label for _, label in neighbors) / k
        y_prob.append(prob)

    y_true = [row["label"] for row in test]

    print("\n--- KNN METRICS ---")
    calculate_metrics(y_true, predictions)
    #calculate_metrics(y_true, predictions, y_prob)



"""
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
"""