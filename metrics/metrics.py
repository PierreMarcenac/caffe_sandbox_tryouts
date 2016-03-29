import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def roc_curve_(y_true, y_score, classes):
    """
    Outputs false positives and true positives rates to build the roc curve
    (receiver operating characteristics) and area under the roc curve
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for class_ in classes:
        fpr[class_], tpr[class_], _ = roc_curve(y_true, y_score[:, class_],
                                                pos_label=class_)
        roc_auc[class_] = auc(fpr[class_], tpr[class_])
    return fpr, tpr, roc_auc

def plot_roc_curve(y_true, y_score, classes):
    """
    Plot roc curve for each class of classes
    """
    fpr, tpr, roc_auc = roc_curve_(y_true, y_score, classes)
    plt.figure()
    for class_ in classes:
        plt.plot(fpr[class_], tpr[class_], label=class_)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('')
    plt.legend(loc="lower right")
    pass

# TODO: implement hamming loss here

def main():
    """
    Case in point for computing the metrics
    """
    # Load modules
    import caffe
    import h5py
    import numpy as np
    from eval.inference import infer_to_h5_fixed_dims

    # Find paths to db/etc
    def path_to(path):
        return base_path + path
    base_path = '/mnt/scratch/pierre/caffe_sandbox_tryouts/'
    fpath_net = path_to('learning_curve/prototxt/net1_test.prototxt')
    fpath_weights = path_to('learning_curve/snapshots/net1_snapshot_iter_10000.caffemodel')
    fpath_db = path_to('inference/mnist_%s_train_lmdb')
    fpath_h5 = path_to('inference/db.h5')

    # Make 1000 inferences and storing keys in an hdf5 if build_db activated
    build_db = True # change to True when needed
    if build_db:
        net = caffe.Net(fpath_net, fpath_weights, caffe.TEST) # or caffe.TRAIN
        keys = ["label", "score"]
        n = 50
        infer_to_h5_fixed_dims(net, keys, n, fpath_h5, preserve_batch=True)

    # Define y_true, y_score and y_pred
    f = h5py.File(fpath_h5, "r")
    y_true = f["label"].value[0]
    y_score = f["score"].value[0]
    y_pred = [np.argmax(y) for y in y_score]

    # Accuracy
    from sklearn.metrics import accuracy_score
    print "y_true", y_true
    print "y_pred", y_pred
    print "y_score", y_score
    accuracy = accuracy_score(y_true, y_pred)
    print "Accuracy:", accuracy

    # ROC curve
    classes = range(10)
    plot_roc_curve(y_true, y_score, classes)
    plt.show()

    # Scikit-learn report
    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred)
    print report

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    confusion = confusion_matrix(y_true, y_pred, labels=classes)
    print "Confusion matrix:\n", confusion

if __name__=="__main__":
    main()
