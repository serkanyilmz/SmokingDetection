from common.methods.decision_tree import decision_tree
from common.methods.knn import knn
from common.methods.logistic_regression import logistic_regression
from common.methods.naive_bayes import naive_bayes
from common.methods.svm import svm
from preprocess import encode_and_convert_data
from common.functions.normalization import normalize_dataset
from common.functions.read_dataset import load_csv

dataset, headers = load_csv("smoking.csv")
dataset = encode_and_convert_data(dataset)
dataset = normalize_dataset(dataset)

knn(dataset)
logistic_regression(dataset)
naive_bayes(dataset)
svm(dataset)
decision_tree(dataset, headers)