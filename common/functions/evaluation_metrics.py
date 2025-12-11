from common.evalution_metrics.accuracy import get_accuracy
from common.evalution_metrics.confusion_matrix import get_confusion_matrix
from common.evalution_metrics.f1_score import get_f1_score
from common.evalution_metrics.precision import get_precision
from common.evalution_metrics.recall import get_recall
from common.evalution_metrics.roc_auc import get_roc_auc


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