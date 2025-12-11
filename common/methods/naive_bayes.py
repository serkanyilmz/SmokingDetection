import math

from common.evaluation_metrics import calculate_metrics
from common.split_dataset import train_val_test_split


def naive_bayes(dataset):

    X = [row[:-1] for row in dataset]
    y = [row[-1] for row in dataset]
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, train_size=0.6, val_size=0.2, random_state=42
    )

    print(f"Eğitim Verisi: {len(X_train)} satır")
    print(f"Test Verisi:   {len(X_test)} satır")

    # Eğitim verisini tekrar birleştir (Naive Bayes sınıfın fit fonksiyonu tüm seti ister)
    # Senin yazdığın 'fit' fonksiyonu X ve y'yi birleşik bekliyor:
    train_set_combined = [x + [label] for x, label in zip(X_train, y_train)]

    # 6. MODEL EĞİTİMİ
    print("Model eğitiliyor...")
    model = GaussianNaiveBayes()
    model.fit(train_set_combined)

    # 7. TAHMİN (TEST SETİ ÜZERİNDE)
    print("Tahminler yapılıyor...")
    predictions = []

    for row in X_test:
        # ROC-AUC için sadece 0 veya 1 demek yetmez, '1 olma ihtimalini' bulmalıyız.
        # Senin sınıfındaki 'calculate_class_probabilities' fonksiyonunu kullanıyoruz.
        probs = model.calculate_class_probabilities(row)

        # probs çıktısı şöyledir: {0.0: 0.00042, 1.0: 0.00008} (Örnek)
        # Bu değerler normalize edilmemiştir (Likelihood).
        # Olasılığa çevirmek için: P(1) = L(1) / (L(0) + L(1))

        likelihood_0 = probs.get(0.0, 0)
        likelihood_1 = probs.get(1.0, 0)

        total_likelihood = likelihood_0 + likelihood_1

        if total_likelihood == 0:
            prob_score = 0.0
        else:
            prob_score = likelihood_1 / total_likelihood

        predictions.append(prob_score)

    # 8. METRİKLERİ HESAPLA
    print("\n--- SONUÇLAR ---")
    calculate_metrics(y_test, predictions)


# --- Sınıf Tanımı (Dosya 1'den alınan, scriptin çalışması için gerekli) ---
class GaussianNaiveBayes:
    def __init__(self):
        self.summaries = {}

    def separate_by_class(self, dataset):
        separated = {}
        for i in range(len(dataset)):
            vector = dataset[i]
            class_value = vector[-1]
            if class_value not in separated:
                separated[class_value] = []
            separated[class_value].append(vector)
        return separated

    def mean(self, numbers):
        return sum(numbers) / float(len(numbers))

    def stdev(self, numbers):
        avg = self.mean(numbers)
        variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
        return math.sqrt(variance) + 1e-9

    def summarize_dataset(self, dataset):
        summaries = []
        for column in zip(*dataset):
            summaries.append((self.mean(column), self.stdev(column)))
        del summaries[-1]
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