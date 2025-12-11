def get_confusion_matrix(actual, predicted):
    tp = tn = fp = fn = 0
    for i in range(len(actual)):
        pred_label = 1.0 if predicted[i] >= 0.5 else 0.0

        if actual[i] == 1.0 and pred_label == 1.0:
            tp += 1
        elif actual[i] == 0.0 and pred_label == 0.0:
            tn += 1
        elif actual[i] == 0.0 and pred_label == 1.0:
            fp += 1
        elif actual[i] == 1.0 and pred_label == 0.0:
            fn += 1
    return [[tn, fp], [fn, tp]]
