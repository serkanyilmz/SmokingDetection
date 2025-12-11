def get_precision(tp, fp):
    if (tp + fp) == 0:
        return 0
    return tp / (tp + fp)
