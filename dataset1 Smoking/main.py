from preprocess import encode_and_convert_data
from common.normalization import normalize_dataset
from common.read_dataset import load_csv
from common.methods.knn import knn
from common.methods.naive_bayes import naive_bayes
from common.methods.decision_tree import decision_tree

dataset = load_csv("smoking.csv")
dataset = encode_and_convert_data(dataset)
dataset = normalize_dataset(dataset)

naive_bayes(dataset)
knn(dataset)
decision_tree(dataset)
