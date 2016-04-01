from nose.tools import assert_equals
import metrics
from sklearn.metrics import roc_curve
import numpy as np

class TestROCCurve:
    def test_correct_classification(self):
        """
        Test on known results for a simple optimal classification:
        verify that the curve is optimal (iff area under the curve is 1)
        """
        # Case of 3 classes which were correctly classified
        classes = range(3)
        y_true = np.array([0, 1, 2])
        y_score = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])
        _, _, auc = metrics.roc_curve_(y_true, y_score, classes)

        # Ideal case
        auc_wanted = 1

        # Test equality for all class in classes
        for class_ in classes:
            assert_equals(auc[class_], auc_wanted)

    def test_pos_label(self):
        """
        Test the use of parameter post_label as opposed
        to using a binary representation (0,1)
        """
        # Binary implementation of metrics.roc_curve_
        def roc_curve_bin(y_true, y_score, classes):
            fpr_bin = dict()
            tpr_bin = dict()
            for class_ in classes:
                # Binary representation of y_true
                y_true_class = [i==class_ for i in y_true]
                # ROC
                fpr_bin[class_], tpr_bin[class_], _ = roc_curve(y_true_class, y_score[:, class_])
            return fpr_bin, tpr_bin

        # Generate rates with boths methods
        classes = range(3)
        y_true = np.array([0, 1, 2])
        y_score = np.array([[5, 5, 0],
                            [1, 6, 1],
                            [5, 2, 3]])
        fpr, tpr, _ = metrics.roc_curve_(y_true, y_score, classes)
        fpr_bin, tpr_bin = roc_curve_bin(y_true, y_score, classes)

        # Check consistency
        for class_ in classes:
            for i, _ in enumerate(fpr[class_]):
                assert_equals(fpr[class_][i], fpr_bin[class_][i])
                assert_equals(tpr[class_][i], tpr_bin[class_][i])

    def test_softmax(self):
        """
        Do we have to normalize y_score using softmax?
        """
        def softmax(x):
            e_x = np.exp(x)
            return e_x / e_x.sum()

        # Normalized/not normalized
        classes = range(3)
        y_true = np.array([0, 1, 2])
        y_score = np.array([[5, 5, 0],
                            [1, 6, 1],
                            [5, 2, 3]])
        y_score_softmax = np.array([softmax(score) for score in y_score])

        fpr, tpr, _ = metrics.roc_curve_(y_true, y_score, classes)
        fpr_norm, tpr_norm, _ = metrics.roc_curve_(y_true, y_score_softmax, classes)

        # Check consistency
        for class_ in classes:
            for i,_ in enumerate(fpr[class_]):
                assert_equals(fpr_norm[class_][i], fpr[class_][i])
                assert_equals(tpr_norm[class_][i], tpr[class_][i])
