from nose.tools import assert_equals
import accuracy
import numpy as np

class TestAccuracy:
    def test_ideal_situation(self):
        """
        Ideal case with accuracy 1
        """
        for n in range(2, 100):
            y_true = np.identity(n)
            y_pred = np.identity(n)
            assert_equals(accuracy.hamming_accuracy(y_true, y_pred), 1)

    def test_all_wrong(self):
        """
        Wrong classification with zero accuracy
        """
        for n in range(2, 100):
            y_true = np.ones((n,n))
            y_pred = np.zeros((n,n))
            assert_equals(accuracy.hamming_accuracy(y_true, y_pred), 0)

    def test_compare_to_sklearn(self):
        """
        Comparison using calculations by hand
        """
        for n in range(2, 100):
            # Multi-label classification notation with binary label indicators
            y_true = np.random.randint(2, size=(n,n))
            y_pred = np.random.randint(2, size=(n,n))
            # Hamming accuracy by hand
            hamming_loss = 0.
            for i in range(n):
                for j in range(n):
                    hamming_loss += (y_true[i,j] != y_pred[i,j])
            # Compare both
            acc_sklearn = 1. - hamming_loss/n**2
            acc_hamming = accuracy.hamming_accuracy(y_true, y_pred)
            assert_equals(acc_hamming, acc_sklearn)
