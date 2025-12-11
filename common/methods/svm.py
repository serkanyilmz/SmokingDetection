import random
from common.functions.evaluation_metrics import calculate_metrics
from common.functions.split_dataset import train_val_test_split

def dot_product(w, x):
    return sum(wj * xj for wj, xj in zip(w, x))

def train_linear_svm_model(train_data, lr=0.01, epochs=100, lambd=1e-3, random_state=None):
    if random_state is not None:
        random.seed(random_state)

    n_samples = len(train_data)
    n_features = len(train_data[0]) - 1

    w = [0.0] * n_features
    b = 0.0

    for _ in range(epochs):
        random.shuffle(train_data)

        for row in train_data:
            features = row[:-1]
            target = row[-1]

            y_i = 1 if target == 1 else -1

            score = dot_product(w, features) + b

            margin = y_i * score

            w = [(1 - lr * lambd) * wj for wj in w]

            if margin < 1:
                w = [wj + lr * y_i * xij for wj, xij in zip(w, features)]

                b = b + lr * y_i

    return w, b

def predict_one(row, w, b):
    score = dot_product(w, row) + b
    return 1.0 if score >= 0 else 0.0


def svm(dataset):
    print("--- SVM (Pegasos Linear) Başlatılıyor ---")

    X = [row[:-1] for row in dataset]
    y = [row[-1] for row in dataset]
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, train_size=0.6, val_size=0.2, random_state=42
    )

    train_set_combined = [x + [label] for x, label in zip(X_train, y_train)]

    # 4. Modeli Eğit
    print("Model eğitiliyor...")
    best_w, best_b = train_linear_svm_model(
        train_set_combined,
        lr=0.01,
        epochs=100,
        lambd=1e-3,
        random_state=42
    )
    predictions = []
    for row in X_test:
        pred = predict_one(row, best_w, best_b)
        predictions.append(pred)

    print("\n=== SVM SONUÇLARI ===")
    calculate_metrics(y_test, predictions)