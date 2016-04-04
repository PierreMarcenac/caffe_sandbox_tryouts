import numpy as np
from sklearn.metrics import hamming_loss

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def hamming_accuracy_test(solver):
    """
    Run hamming_accuracy on a test batch for mnist
    """
    solver.test_nets[0].forward()
    y_true = solver.test_nets[0].blobs['label'].data
    y_prob = solver.test_nets[0].blobs['score'].data
    y_pred = np.array([[int(prob>=0) for prob in preds] for preds in y_prob])
    return hamming_accuracy(y_true, y_pred)

def hamming_accuracy_test_twoears(solver):
    """
    Run hamming_accuracy on a test batch for two!ears sound data
    """
    solver.test_nets[0].forward()
    y_true = render_to_matrix(solver.test_nets[0].blobs['label'].data)
    y_prob = solver.test_nets[0].blobs['score'].data
    y_pred = np.array([[int(prob>=0) for prob in preds] for preds in y_prob])
    return hamming_accuracy(y_true, y_pred)

def render_to_matrix(v):
    """v collection of vertical vectors"""
    n, _, p, _ = v.shape
    return v.reshape((n,p))

def hamming_accuracy(y_true, y_pred):
    """
    hamming_accuracy = 1 - hamming_loss
    """
    return 1 - hamming_loss(y_true, y_pred)
