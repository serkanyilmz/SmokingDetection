# svm_smoking.py
# Kütüphanesiz Linear SVM örneği
# Kaggle veri setleri: smoking.csv ve smoking_drinking_dataset.csv

import random

# --- CSV yükleme ---
def load_csv(filename, delimiter=","):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
        header = lines[0].split(delimiter)
        for line in lines[1:]:
            parts = line.split(delimiter)
            data.append(parts)
    return header, data

# --- Train-test split (stratified) ---
def train_test_split_stratified(X, y, test_size=0.3, random_state=None):
    if random_state is not None:
        random.seed(random_state)
    idx0 = [i for i, v in enumerate(y) if v == 0]
    idx1 = [i for i, v in enumerate(y) if v == 1]
    random.shuffle(idx0); random.shuffle(idx1)
    split0 = int(len(idx0) * (1 - test_size))
    split1 = int(len(idx1) * (1 - test_size))
    train_idx = idx0[:split0] + idx1[:split1]
    test_idx = idx0[split0:] + idx1[split1:]
    random.shuffle(train_idx); random.shuffle(test_idx)
    X_train = [X[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]
    return X_train, X_test, y_train, y_test

# --- Min-max scaler ---
def min_max_scale(X):
    n_features = len(X[0])
    min_vals = [min(row[i] for row in X) for i in range(n_features)]
    max_vals = [max(row[i] for row in X) for i in range(n_features)]
    ranges = [(max_vals[i] - min_vals[i]) if (max_vals[i] - min_vals[i]) != 0 else 1.0 for i in range(n_features)]
    X_scaled = [[(row[i] - min_vals[i]) / ranges[i] for i in range(n_features)] for row in X]
    return X_scaled, min_vals, max_vals

def min_max_scale_transform(X, min_vals, max_vals):
    n_features = len(X[0])
    ranges = [(max_vals[i] - min_vals[i]) if (max_vals[i] - min_vals[i]) != 0 else 1.0 for i in range(n_features)]
    return [[(row[i] - min_vals[i]) / ranges[i] for i in range(n_features)] for row in X]

# --- Confusion matrix ---
def confusion_matrix_manual(y_true, y_pred):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    return tp, tn, fp, fn

# --- Linear SVM (Pegasos tarzı) ---
class LinearSVM:
    def __init__(self, lr=0.01, epochs=100, lambd=1e-3, random_state=None):
        self.lr = lr
        self.epochs = epochs
        self.lambd = lambd
        self.random_state = random_state
        self.w = []
        self.b = 0.0

    def _to_pm1(self, y):
        return [1 if val == 1 else -1 for val in y]

    def fit(self, X, y):
        if self.random_state is not None:
            random.seed(self.random_state)
        n_samples = len(X)
        n_features = len(X[0])
        self.w = [0.0] * n_features
        self.b = 0.0
        y_pm1 = self._to_pm1(y)

        for _ in range(self.epochs):
            indices = list(range(n_samples))
            random.shuffle(indices)
            for i in indices:
                x_i = X[i]
                y_i = y_pm1[i]
                score = sum(wj * xij for wj, xij in zip(self.w, x_i)) + self.b
                margin = y_i * score
                # L2 shrink
                self.w = [(1 - self.lr * self.lambd) * wj for wj in self.w]
                if margin < 1:
                    self.w = [wj + self.lr * y_i * xij for wj, xij in zip(self.w, x_i)]
                    self.b = self.b + self.lr * y_i

    def predict(self, X):
        preds = []
        for row in X:
            score = sum(wj * xij for wj, xij in zip(self.w, row)) + self.b
            preds.append(1 if score >= 0 else 0)
        return preds

    def score(self, X, y):
        preds = self.predict(X)
        correct = sum(1 for yt, yp in zip(y, preds) if yt == yp)
        return correct / len(y)

# --- MAIN fonksiyonları ---
def run_smoking_dataset():
    header, rows = load_csv("smoking.csv")
    X, y = [], []
    for row in rows:
        features = [float(val) for val in row[:-1]]  # son sütun hariç
        target = int(row[-1])  # son sütun smoking
        X.append(features)
        y.append(target)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split_stratified(X, y, test_size=0.3, random_state=1)
    X_train, min_vals, max_vals = min_max_scale(X_train_raw)
    X_test = min_max_scale_transform(X_test_raw, min_vals, max_vals)

    model = LinearSVM(lr=0.05, epochs=150, lambd=1e-3, random_state=42)
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    y_pred = model.predict(X_test)
    tp, tn, fp, fn = confusion_matrix_manual(y_test, y_pred)

    print("=== Smoking.csv Dataset ===")
    print(f"Train accuracy: {train_acc:.4f}")
    print("Confusion (tp, tn, fp, fn):", tp, tn, fp, fn)
    print("Test accuracy:", (tp + tn) / (tp + tn + fp + fn))

def run_smoking_drinking_dataset():
    header, rows = load_csv("smoking_drinking_dataset.csv")
    X, y = [], []
    for row in rows:
        # smoking sütunu hedef, diğerleri özellik
        # smoking sütunu header'da hangi indexteyse bulalım
        smoking_idx = header.index("smoking")
        features = [float(val) for i, val in enumerate(row) if i != smoking_idx]
        target = int(row[smoking_idx])
        X.append(features)
        y.append(target)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split_stratified(X, y, test_size=0.3, random_state=1)
    X_train, min_vals, max_vals = min_max_scale(X_train_raw)
    X_test = min_max_scale_transform(X_test_raw, min_vals, max_vals)

    model = LinearSVM(lr=0.05, epochs=150, lambd=1e-3, random_state=42)
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    y_pred = model.predict(X_test)
    tp, tn, fp, fn = confusion_matrix_manual(y_test, y_pred)

    print("=== Smoking & Drinking Dataset ===")
    print(f"Train accuracy: {train_acc:.4f}")
    print("Confusion (tp, tn, fp, fn):", tp, tn, fp, fn)
    print("Test accuracy:", (tp + tn) / (tp + tn + fp + fn))

if __name__ == "__main__":
    run_smoking_dataset()
    run_smoking_drinking_dataset()