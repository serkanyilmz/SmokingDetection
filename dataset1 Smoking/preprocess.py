def encode_and_convert_data(dataset):
    """
    ID'yi siler, metinleri sayıya çevirir (Encoding).
    """
    processed_data = []
    for row in dataset[0]:
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