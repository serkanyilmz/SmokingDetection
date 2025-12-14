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
        return dataset, headers
    except FileNotFoundError:
        print(f"Error: '{filename}' can't found.")
        return []
