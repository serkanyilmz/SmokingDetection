from common.methods.decision_tree import decision_tree
from common.methods.knn import knn
from common.methods.naive_bayes import naive_bayes
from common.normalization import normalize_dataset
from common.read_dataset import load_csv
from preprocess import encode_and_convert_data

dataset, headers = load_csv('./smoking_driking_dataset_Ver01.csv')
dataset = encode_and_convert_data(dataset)
dataset = normalize_dataset(dataset)

naive_bayes(dataset)
knn(dataset)
decision_tree(dataset, headers)
