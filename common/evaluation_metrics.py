def get_accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        pred_label = 1.0 if predicted[i] >= 0.5 else 0.0
        if actual[i] == pred_label:
            correct += 1
    return correct / float(len(actual)) * 100.0


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


def get_roc_auc(actual, predicted):
    data = list(zip(actual, predicted))
    data.sort(key=lambda x: x[1], reverse=True)

    tp = 0
    fp = 0
    total_pos = sum(actual)
    total_neg = len(actual) - total_pos

    if total_pos == 0 or total_neg == 0:
        return 0.0

    fpr_prev = 0
    tpr_prev = 0
    auc = 0

    for i in range(len(data)):
        if data[i][0] == 1.0:
            tp += 1
        else:
            fp += 1

        tpr = tp / total_pos
        fpr = fp / total_neg

        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2

        fpr_prev = fpr
        tpr_prev = tpr

    return auc


def calculate_metrics(actual, predicted):
    cm = get_confusion_matrix(actual, predicted)
    tn, fp = cm[0]
    fn, tp = cm[1]

    accuracy = get_accuracy(actual, predicted)
    precision = get_precision(tp, fp)
    recall = get_recall(tp, fn)
    f1 = get_f1_score(precision, recall)
    roc_auc = get_roc_auc(actual, predicted)

    print(f"Accuracy:                           {accuracy:.2f}%")
    print(f"Precision:                          {precision:.4f}")
    print(f"Recall:                             {recall:.4f}")
    print(f"F1-Score:                           {f1:.4f}")
    print(f"ROC-AUC:                            {roc_auc:.4f}")
    print(f"Confusion Matrix tn, fp, fn, tp:    {cm}")