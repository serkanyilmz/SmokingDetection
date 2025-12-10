import csv
import math
from common.evaluation_metrics import (
    get_accuracy,
    get_confusion_matrix,
    get_precision,
    get_recall,
    get_f1_score)

def load_csv(filename):
    dataset = []
    try:
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            headers = next(reader)  # Skip header
            for row in reader:
                if row:
                    dataset.append(row)
        return dataset
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return []


def encode_and_convert_data(dataset):
    """
    Specific preprocessing for the Smoking Dataset.
    1. Deletes ID column (Index 0).
    2. Encodes Gender, Oral, Tartar to numbers.
    3. Converts everything else to floats.
    """
    processed_data = []

    # Define mappings for text columns
    # Based on your data: gender (index 1), oral (23), tartar (25)
    # Note: We shift indices by -1 because we will delete the ID column first

    for row in dataset:
        # 1. Remove ID (First column)
        # We create a new row starting from index 1 to the end
        new_row = row[1:]

        # Now the indices have shifted left by 1:
        # old_gender (1) -> new_gender (0)
        # old_oral (23) -> new_oral (22)
        # old_tartar (25) -> new_tartar (24)

        # 2. Handle Gender (Index 0 in new_row)
        if new_row[0] == 'M':
            new_row[0] = 1.0
        elif new_row[0] == 'F':
            new_row[0] = 0.0

        # 3. Handle Oral (Index 22 in new_row)
        if new_row[22] == 'Y':
            new_row[22] = 1.0
        elif new_row[22] == 'N':
            new_row[22] = 0.0

        # 4. Handle Tartar (Index 24 in new_row)
        if new_row[24] == 'Y':
            new_row[24] = 1.0
        elif new_row[24] == 'N':
            new_row[24] = 0.0

        # 5. Convert EVERYTHING to float
        # This loop goes through every item and forces it to be a number
        clean_row = []
        for val in new_row:
            try:
                clean_row.append(float(val))
            except ValueError:
                # If data is empty or corrupted, default to 0.0
                clean_row.append(0.0)

        processed_data.append(clean_row)

    return processed_data


# ---------------------------------------------------------
# 2. NAIVE BAYES ALGORITHM (GAUSSIAN)
# ---------------------------------------------------------

class GaussianNaiveBayes:
    def __init__(self):
        self.summaries = {}

    def separate_by_class(self, dataset):
        separated = {}
        for i in range(len(dataset)):
            vector = dataset[i]
            class_value = vector[-1]  # Last column is target
            if class_value not in separated:
                separated[class_value] = []
            separated[class_value].append(vector)
        return separated

    def mean(self, numbers):
        return sum(numbers) / float(len(numbers))

    def stdev(self, numbers):
        avg = self.mean(numbers)
        variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
        # Add a tiny epsilon to avoid division by zero later if stdev is 0
        return math.sqrt(variance) + 1e-9

    def summarize_dataset(self, dataset):
        summaries = []
        for column in zip(*dataset):
            summaries.append((self.mean(column), self.stdev(column)))
        del summaries[-1]  # Delete summary of class column
        return summaries

    def fit(self, train_data):
        separated = self.separate_by_class(train_data)
        self.summaries = {}
        for class_value, rows in separated.items():
            self.summaries[class_value] = self.summarize_dataset(rows)

    def calculate_probability(self, x, mean, stdev):
        if stdev == 0: return 0
        exponent = math.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    def calculate_class_probabilities(self, row):
        probabilities = {}
        for class_value, class_summaries in self.summaries.items():
            probabilities[class_value] = 1
            for i in range(len(class_summaries)):
                mean, stdev = class_summaries[i]
                x = row[i]
                probabilities[class_value] *= self.calculate_probability(x, mean, stdev)
        return probabilities

    def predict(self, row):
        probabilities = self.calculate_class_probabilities(row)
        best_label, best_prob = None, -1
        for class_value, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        return best_label

# ---------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------

def main():
    print("1. Loading Data...")
    raw_train = load_csv("../../smokingtrain.csv")
    raw_val = load_csv("../../smokingvalidation.csv")
    raw_test = load_csv("../../smokingtest.csv")

    if not raw_train:
        print("Data not found!")
        return

    print("2. Preprocessing Data (Encoding 'F'/'M', 'Y'/'N' and removing ID)...")
    # Apply the specific encoding function to all datasets
    train_data = encode_and_convert_data(raw_train)
    val_data = encode_and_convert_data(raw_val)
    test_data = encode_and_convert_data(raw_test)

    print(f"   Rows processed: {len(train_data)}")
    print(f"   Columns (features + target): {len(train_data[0])}")

    print("3. Training Model...")
    model = GaussianNaiveBayes()
    model.fit(train_data)

    print("4. Evaluating...")
    val_actual = [row[-1] for row in val_data]
    val_predictions = []

    for row in val_data:
        pred = model.predict(row)
        val_predictions.append(pred)

    accuracy = get_accuracy(val_actual, val_predictions)
    tp, tn, fp, fn = get_confusion_matrix(val_actual, val_predictions)
    precision = get_precision(tp, fp)
    recall = get_recall(tp, fn)
    f1 = get_f1_score(precision, recall)

    print("-" * 30)
    print(f"Accuracy:  {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("-" * 30)
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")


if __name__ == "__main__":
    main()