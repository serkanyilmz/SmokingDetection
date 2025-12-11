import math

from common.evaluation_metrics import calculate_metrics
from common.split_dataset import train_val_test_split


def sigmoid(z):
    if z < -700: return 0
    if z > 700: return 1
    return 1.0 / (1.0 + math.exp(-z))


def train_logistic_regression_model(train_data, learning_rate=0.01, epochs=100):
    if not train_data:
        return [], 0.0

    n_features = len(train_data[0]) - 1
    weights = [0.0] * n_features
    bias = 0.0

    for epoch in range(epochs):
        for row in train_data:
            features = row[:-1]
            target = row[-1]

            z = bias
            for i in range(n_features):
                z += weights[i] * features[i]
            y_pred = sigmoid(z)
            error = target - y_pred

            bias = bias + (learning_rate * error)
            for i in range(n_features):
                weights[i] = weights[i] + (learning_rate * error * features[i])

    return weights, bias

def predict_probability(row, weights, bias):
    z = bias
    for i in range(len(weights)):
        z += weights[i] * row[i]

    return sigmoid(z)

def logistic_regression(dataset):
    X = [row[:-1] for row in dataset]
    y = [row[-1] for row in dataset]

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, train_size=0.6, val_size=0.2, random_state=42
    )
    train_set_combined = [x + [label] for x, label in zip(X_train, y_train)]

    best_weights, best_bias = train_logistic_regression_model(
        train_set_combined,
        learning_rate=0.1,
        epochs=200
    )

    print("Tahminler yapılıyor...")
    predictions = []

    for row in X_test:
        prob_score = predict_probability(row, best_weights, best_bias)
        predictions.append(prob_score)

    print("\n--- LOGISTIC REGRESSION METRICS ---")
    calculate_metrics(y_test, predictions)

