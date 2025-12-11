from common.methods.naive_bayes import naive_bayes
from common.methods.svm import svm
from common.functions.normalization import normalize_dataset
from common.functions.read_dataset import load_csv
from preprocess import encode_and_convert_data

dataset = load_csv('./smoking_driking_dataset_Ver01.csv')
dataset = encode_and_convert_data(dataset)
dataset = normalize_dataset(dataset)

dataset = dataset[:200]

#naive_bayes(dataset)
#decision_tree(dataset)
naive_bayes(dataset)