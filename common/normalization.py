def normalize_dataset(dataset):
    """
    Min-Max Normalizasyonu: Tüm verileri 0 ile 1 arasına sıkıştırır.
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

    return dataset