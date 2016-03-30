import numpy as np
from sklearn.metrics import hamming_loss

def softmax(x):
    """
    Softmax on vector x
    """
    e_x = np.exp(x)
    return e_x / e_x.sum()

def hamming_loss_test(solver):
    """
    Run Hamming loss on a test batch
    """
    solver.test_nets[0].forward()
    y_true = solver.test_nets[0].blobs['label'].data
    y_prob = np.array([softmax(label) for label in solver.test_nets[0].blobs['score'].data])
    y_pred = np.array([[0.+(prob==np.max(label)) for prob in label] for label in y_prob])
    return hamming_loss(y_true, y_pred)

def main():
    solver = ""

    ### TODO: ecrire les tests pour hamming loss: situation ideale, situation ou tout est faux, exemples de scipy, comparer directement avec accuracy de scipy
