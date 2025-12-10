
data=datasetioku

data = preprocess(data)

acc pred = decision_tree(data)
show_result(acc, red)

acc pred = knn(data)
show_result(acc, red)