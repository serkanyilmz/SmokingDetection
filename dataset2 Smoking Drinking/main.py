from common.methods.logistic_regression import logistic_regression
from common.methods.naive_bayes import naive_bayes
from common.normalization import normalize_dataset
from common.read_dataset import load_csv
from preprocess import encode_and_convert_data

dataset = load_csv('./smoking_driking_dataset_Ver01.csv')

dataset = encode_and_convert_data(dataset)
dataset = normalize_dataset(dataset)

#naive_bayes(dataset)
logistic_regression(dataset)

#data=datasetioku
#
#data = preprocess(data)
#
#acc pred = decision_tree(data)
#show_result(acc, red)
#
#acc pred = knn(data)
#show_result(acc, red)