import math
from common.evaluation_metrics import calculate_metrics
from common.split_dataset import train_val_test_split

def calculate_mean(numbers):
    return sum(numbers) / float(len(numbers))


def calculate_stdev(numbers):
    avg = calculate_mean(numbers)
    variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance) + 1e-9


def calculate_gaussian_probability(x, mean, stdev):
    if stdev == 0: return 0
    exponent = math.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

def separate_by_class(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if class_value not in separated:
            separated[class_value] = []
        separated[class_value].append(vector)
    return separated


def summarize_dataset(dataset):
    summaries = []
    for column in zip(*dataset):
        summaries.append((calculate_mean(column), calculate_stdev(column)))
    del summaries[-1]
    return summaries


def train_naive_bayes_model(train_data):
    separated = separate_by_class(train_data)
    model_summaries = {}
    for class_value, rows in separated.items():
        model_summaries[class_value] = summarize_dataset(rows)
    return model_summaries

def calculate_class_likelihoods(row, model_summaries):
    probabilities = {}
    for class_value, class_summaries in model_summaries.items():
        probabilities[class_value] = 1
        for i in range(len(class_summaries)):
            mean, stdev = class_summaries[i]
            x = row[i]
            probabilities[class_value] *= calculate_gaussian_probability(x, mean, stdev)
    return probabilities

def naive_bayes(dataset):
    X = [row[:-1] for row in dataset]
    y = [row[-1] for row in dataset]

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, train_size=0.6, val_size=0.2, random_state=42
    )
    train_set_combined = [x + [label] for x, label in zip(X_train, y_train)]

    print("Model eğitiliyor...")
    model_summaries = train_naive_bayes_model(train_set_combined)

    predictions = []

    for row in X_test:
        likelihoods = calculate_class_likelihoods(row, model_summaries)

        likelihood_0 = likelihoods.get(0.0, 0)
        likelihood_1 = likelihoods.get(1.0, 0)

        total_likelihood = likelihood_0 + likelihood_1

        if total_likelihood == 0:
            prob_score = 0.0
        else:
            # P(1) = L(1) / (L(0) + L(1))
            prob_score = likelihood_1 / total_likelihood
        predictions.append(prob_score)

    print("\n--- NAIVE BAYES METRICS ---")
    calculate_metrics(y_test, predictions)