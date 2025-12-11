from common.normalization import normalize_dataset
from common.read_dataset import load_csv
from preprocess import encode_and_convert_data

dataset = load_csv('./smoking_driking_dataset_Ver01.csv')

dataset = encode_and_convert_data(dataset)
print(dataset[0])
dataset = normalize_dataset(dataset)
print(dataset[0])

print(dataset[0])
print(dataset[1])
print(len(dataset))
#data=datasetioku
#
#data = preprocess(data)
#
#acc pred = decision_tree(data)
#show_result(acc, red)
#
#acc pred = knn(data)
#show_result(acc, red)