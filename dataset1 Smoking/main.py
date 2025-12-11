from common.methods.knn import knn
from common.methods.logistic_regression import logistic_regression
from common.methods.svm import svm
from preprocess import encode_and_convert_data
from common.normalization import normalize_dataset
from common.methods.naive_bayes import naive_bayes
from common.read_dataset import load_csv
from common.methods.naive_bayes import naive_bayes
from common.read_dataset import load_csv
from common.methods.decision_tree import decision_tree


dataset = load_csv("smoking.csv")
dataset = encode_and_convert_data(dataset)
dataset = normalize_dataset(dataset)

dataset = dataset[:200]

svm(dataset)