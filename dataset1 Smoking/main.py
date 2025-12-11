from preprocess import encode_and_convert_data
from common.normalization import normalize_dataset
from common.methods.naive_bayes import naive_bayes
from common.read_dataset import load_csv

dataset = load_csv("smoking.csv")
dataset = encode_and_convert_data(dataset)
dataset = normalize_dataset(dataset)

naive_bayes(dataset)

