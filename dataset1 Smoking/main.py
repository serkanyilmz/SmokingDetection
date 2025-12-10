from preprocess import encode_and_convert_data
from common.normalization import normalize_dataset
from common.read_dataset import read_dataset
from common.methods.decision_tree import decision_tree
from common.methods.knn import knn
from common.methods.logistic_regression import logistic_regression
from common.methods.svm import linear_svm
from common.methods.naive_bayes import naive_bayes


data = read_dataset("smoking.csv")
# data bölme
proced_data = encode_and_convert_data(data)
normalized_data = normalize_dataset(proced_data)


acc pred = decision_tree(data)
show_result(acc, red)

acc pred = knn(data)
show_result(acc, red)