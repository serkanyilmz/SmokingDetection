def encode_and_convert_data(dataset):
    processed_data = []

    for row in dataset[0]:
        print(row)
        # 1. Son sütunu atıyoruz
        new_row = row[:-1]

        # 1. sex -> Male:1, Female:0
        if new_row[0] == 'Male':
            new_row[0] = 1.0
        elif new_row[0] == 'Female':
            new_row[0] = 0.0
        else:
            new_row[0] = 0.0

        # önceden içip bırakanlar da 1
        if new_row[-1] == "1.0":
            new_row[-1] = 0.0
        elif new_row[-1] == "2.0" or new_row[-1] == "3.0":
            new_row[-1] = 1.0
        else:
            new_row[-1] = 0.0

        # 3. Tüm alanları float'a çevir
        clean_row = []
        for val in new_row:
            print(val)
            try:
                clean_row.append(float(val))
            except ValueError:
                clean_row.append(0.0)

        processed_data.append(clean_row)

    return processed_data