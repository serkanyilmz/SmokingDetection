def get_recall(tp, fn):
    if (tp + fn) == 0:
        return 0
    return tp / (tp + fn)
