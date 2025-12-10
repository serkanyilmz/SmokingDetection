import csv
import math
from common.evaluation_metrics import (
    get_accuracy,
    get_confusion_matrix,
    get_precision,
    get_recall,
    get_f1_score
)

# ---------------------------------------------------------
# 1. VERİ YÜKLEME VE ÖN İŞLEME
# ---------------------------------------------------------

def load_csv(filename):
    dataset = []
    try:
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            headers = next(reader)
            for row in reader:
                if row:
                    dataset.append(row)
        return dataset
    except FileNotFoundError:
        print(f"Hata: '{filename}' bulunamadı.")
        return []


def encode_and_convert_data(dataset):
    """
    ID'yi siler, metinleri sayıya çevirir (Encoding).
    """
    processed_data = []
    for row in dataset:
        # 1. ID Sütununu (ilk sütun) atıyoruz
        new_row = row[1:]

        # 2. Cinsiyet (Index 0) - M:1, F:0
        if new_row[0] == 'M':
            new_row[0] = 1.0
        elif new_row[0] == 'F':
            new_row[0] = 0.0

        # 3. Oral (Index 22) - Y:1, N:0
        if new_row[22] == 'Y':
            new_row[22] = 1.0
        elif new_row[22] == 'N':
            new_row[22] = 0.0

        # 4. Tartar (Index 24) - Y:1, N:0
        if new_row[24] == 'Y':
            new_row[24] = 1.0
        elif new_row[24] == 'N':
            new_row[24] = 0.0

        # 5. Hepsini Float Yap
        clean_row = []
        for val in new_row:
            try:
                clean_row.append(float(val))
            except ValueError:
                clean_row.append(0.0)

        processed_data.append(clean_row)

    return processed_data


def normalize_dataset(dataset):
    """
    Min-Max Normalizasyonu: Tüm verileri 0 ile 1 arasına sıkıştırır.
    Lojistik Regresyon için BU ÇOK ÖNEMLİDİR.
    """
    # Hedef sütun (son sütun) hariç min ve max değerleri bul
    if not dataset: return
    n_cols = len(dataset[0]) - 1  # Son sütun label, onu normalize etme

    minmax = []
    for i in range(n_cols):
        col_values = [row[i] for row in dataset]
        minmax.append([min(col_values), max(col_values)])

    for row in dataset:
        for i in range(n_cols):
            min_val = minmax[i][0]
            max_val = minmax[i][1]
            # Bölme hatasını önle
            if max_val - min_val == 0:
                row[i] = 0.0
            else:
                row[i] = (row[i] - min_val) / (max_val - min_val)


# ---------------------------------------------------------
# 2. LOJİSTİK REGRESYON ALGORİTMASI
# ---------------------------------------------------------

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.bias = 0.0

    def sigmoid(self, z):
        """
        Sigmoid Aktivasyon Fonksiyonu:
        Herhangi bir sayıyı 0 ile 1 arasına sıkıştırır (Olasılık verir).
        """
        # Taşmayı (Overflow) önlemek için koruma
        if z < -700: return 0
        if z > 700: return 1
        return 1.0 / (1.0 + math.exp(-z))

    def fit(self, train_data):
        """
        Stokastik Gradient Descent (SGD) kullanarak eğitimi gerçekleştirir.
        """
        n_features = len(train_data[0]) - 1
        self.weights = [0.0] * n_features
        self.bias = 0.0

        for epoch in range(self.epochs):
            sum_error = 0
            for row in train_data:
                features = row[:-1]
                target = row[-1]  # Gerçek değer (0 veya 1)

                # 1. Tahmin Yap (y_pred)
                # z = w1*x1 + w2*x2 + ... + bias
                z = self.bias
                for i in range(n_features):
                    z += self.weights[i] * features[i]

                y_pred = self.sigmoid(z)

                # 2. Hatayı Hesapla
                error = target - y_pred
                sum_error += error ** 2

                # 3. Ağırlıkları Güncelle (Gradient Descent)
                # w_yeni = w_eski + (learning_rate * hata * x)
                self.bias = self.bias + (self.learning_rate * error)

                for i in range(n_features):
                    self.weights[i] = self.weights[i] + (self.learning_rate * error * features[i])

            # İlerleme durumunu görmek istersen:
            # print(f"Epoch {epoch}, Error: {sum_error:.4f}")

    def predict(self, row):
        """
        Bir satır için tahmin yapar. Olasılık > 0.5 ise 1 döner.
        """
        features = row[:-1]
        z = self.bias
        for i in range(len(features)):
            z += self.weights[i] * features[i]

        prob = self.sigmoid(z)
        return 1.0 if prob >= 0.5 else 0.0


# ---------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------

def main():
    print("1. Veri Yükleniyor...")
    raw_train = load_csv("../../smokingtrain.csv")
    raw_val = load_csv("../../smokingvalidation.csv")

    if not raw_train:
        print("Veri bulunamadı!")
        return

    print("2. Veri İşleniyor (Encoding)...")
    train_data = encode_and_convert_data(raw_train)
    val_data = encode_and_convert_data(raw_val)

    # Lojistik Regresyon için ÖNEMLİ ADIM: Normalizasyon
    print("3. Veri Normalize Ediliyor (0-1 Aralığı)...")
    normalize_dataset(train_data)
    normalize_dataset(val_data)

    print(f"   Eğitim Seti Boyutu: {len(train_data)}")

    print("4. Model Eğitiliyor (Gradient Descent)...")
    # Learning rate (öğrenme hızı) çok yüksekse hata artar, çok düşükse öğrenmez.
    # 0.01 veya 0.1 genellikle iyidir.
    model = LogisticRegression(learning_rate=0.12, epochs=200)
    model.fit(train_data)

    print("5. Değerlendirme Yapılıyor...")
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
    print(f"Confusion Matrix: TP:{tp}, TN:{tn}, FP:{fp}, FN:{fn}")


if __name__ == "__main__":
    main()