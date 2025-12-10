import csv

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
