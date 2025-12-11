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
