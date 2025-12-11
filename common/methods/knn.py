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

    for row in test:
        predictions.append(knn_predict(train, row, k))

    y_true = [row["label"] for row in test]

    print("\n--- KNN METRICS ---")
    calculate_metrics(y_true, predictions)
