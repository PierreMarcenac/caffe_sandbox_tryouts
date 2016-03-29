from nose.tools import assert_equals
import metrics
import numpy as np

class TestROCCurve:
    def test_correct_classification(self):
        # Case of 3 classes which were correctly classified
        classes = range(3)
        y_true = np.array([0, 1, 2])
        y_score = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])
        fpr, tpr, auc = metrics.roc_curve_(y_true, y_score, classes)

        # Ideal case
        fpr_awaited = 0 # no false positive
        tpr_awaited = 1 # only true positives
        auc_awaited = 1 # area of (1-0)x1

        # Test equality for all class in classes
        for class_ in classes:
            assert_equals(fpr_awaited, fpr[class_][0])
            assert_equals(tpr_awaited, tpr[class_][0])
            assert_equals(auc_awaited, auc[class_])
