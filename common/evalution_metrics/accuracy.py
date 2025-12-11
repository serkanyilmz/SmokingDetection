def get_accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        pred_label = 1.0 if predicted[i] >= 0.5 else 0.0
        if actual[i] == pred_label:
            correct += 1
    return correct / float(len(actual)) * 100.0
