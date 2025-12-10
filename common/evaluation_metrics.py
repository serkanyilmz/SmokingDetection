def get_accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def get_confusion_matrix(actual, predicted):
    tp = tn = fp = fn = 0
    for i in range(len(actual)):
        if actual[i] == 1.0 and predicted[i] == 1.0:
            tp += 1
        elif actual[i] == 0.0 and predicted[i] == 0.0:
            tn += 1
        elif actual[i] == 0.0 and predicted[i] == 1.0:
            fp += 1
        elif actual[i] == 1.0 and predicted[i] == 0.0:
            fn += 1
    return tp, tn, fp, fn


def get_precision(tp, fp):
    if (tp + fp) == 0:
        return 0
    return tp / (tp + fp)


def get_recall(tp, fn):
    if (tp + fn) == 0:
        return 0
    return tp / (tp + fn)


def get_f1_score(precision, recall):
    if (precision + recall) == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)