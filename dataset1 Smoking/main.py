from common.methods.decision_tree import decision_tree
from common.methods.svm import svm
from preprocess import encode_and_convert_data
from common.functions.normalization import normalize_dataset
from common.functions.read_dataset import load_csv

dataset = load_csv("smoking.csv")
print(len(dataset[1]))
dataset = encode_and_convert_data(dataset)
dataset = normalize_dataset(dataset)

dataset = dataset[:200]

decision_tree(dataset)