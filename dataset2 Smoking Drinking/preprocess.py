def encode_and_convert_data(dataset):
    processed_data = []

    for row in dataset:
        new_row = row.copy()

        # 1. sex -> Male:1, Female:0
        if new_row[0] == 'Male':
            new_row[0] = 1.0
        elif new_row[0] == 'Female':
            new_row[0] = 0.0
        else:
            new_row[0] = 0.0

        # 2. DRK_YN (son sütun) -> Y:1, N:0
        if new_row[-1] == 'Y':
            new_row[-1] = 1.0
        elif new_row[-1] == 'N':
            new_row[-1] = 0.0
        else:
            new_row[-1] = 0.0

        # 3. Tüm alanları float'a çevir
        clean_row = []
        for val in new_row:
            try:
                clean_row.append(float(val))
            except ValueError:
                clean_row.append(0.0)

        processed_data.append(clean_row)

    return processed_data