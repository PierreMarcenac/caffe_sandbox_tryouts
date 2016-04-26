from learning_curve.mnist_train_test import *
import lmdb, h5py
from caffe.io import datum_to_array

process_names = ["net_multi"] # "net_single"

for process_name in process_names:
    accuracy = False
    if "multi" in process_name:
        accuracy = True
    print process_name
    log_name = "logs/caffe_" + process_name + ".log"
    solver_name = "prototxt/"+process_name+"_solver.prototxt"
    print "Start training process:", process_name
    s = caffe_pb2.SolverParameter()
    train_test_net_python(solver_name, log_name, accuracy=accuracy, debug=False)
    print "Stop training"

	print_learning_curve(process_name, log_name, "figs/", accuracy=True)
