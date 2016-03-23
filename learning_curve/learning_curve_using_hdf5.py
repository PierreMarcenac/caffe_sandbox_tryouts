# MNIST - MULTI-CLASS MULTI-LABEL CLASSIFICATION

# Convert input lmdb files to hdf5 files
from models_comparison_with_caffe_sandbox import *
import lmdb, h5py
from caffe.io import datum_to_array

def silent_remove(filename):
    try:
        os.remove(filename)
    except OSError:
	pass

# Scalar is transformed to a vector of length lg with a 1 at position scalar and zeros otherwise
def vectorize(scalar, lg): # DONT KNOW THE PROPER DIMENSION FOR LABEL SPACE
    vec = np.zeros(lg)
    vec[scalar][0][0] = 1
    return vec

def vectorize2(scalar, lg):
    vec = np.zeros(lg)
    vec[scalar] = 1
    return vec

fpath_db = "/mnt/scratch/pierre/caffe_sandbox_tryouts/mnist/mnist_{0}_{1}/"

# Make a copy of lmdb and vectorize labels to allow multi-label classification
def make_hdf5(phase, size):
    fpath_hdf5_phase = (fpath_db+"mnist_{0}.h5").format(phase, "hdf5")
    fpath_lmdb_phase = fpath_db.format(phase, "lmdb")
    # lmdb
    lmdb_env = lmdb.open(fpath_lmdb_phase)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe.proto.caffe_pb2.Datum()
    # hdf5
    silent_remove(fpath_hdf5_phase)
    f = h5py.File(fpath_hdf5_phase, "w")
    f.create_dataset("data", (size, 1, 28, 28), dtype="int32")
    f.create_dataset("label", (size, 10), dtype="int32")
    # write and normalize
    for key, value in lmdb_cursor:
        datum.ParseFromString(value)
	key = int(key)
        label = int(datum.label)
        image = caffe.io.datum_to_array(datum)
 	image = image/255.
	# --> permute images from 3D (channel x column x row) to 4D (column x row x channel x numbers of images)
	# write images in hdf5 db
        f["data"][key] = image
	# --> permute labels from 1D scalar to 2D (labels x number of labels)
	# write label in hdf5 db
	#f["label"][key] = np.array([[vectorize(label, (10, 1, 1))]])
	f["label"][key] = np.array(vectorize2(label, 10))
    # close all working files/environments
    f.close()
    lmdb_cursor.close()
    lmdb_env.close()
    pass

train_size = 60000
test_size = 10000
train_name = "train"
test_name = "test"

# Render copy
make_hdf5(train_name, train_size)
print "Copy of train set in hdf5: done"
make_hdf5(test_name, test_size)
print "Copy of test set in hdf5: done"


# Net architecture to handle hdf5 inputs (use SigmoidCrossEntropyLoss)
def net_hdf5(hdf5, batch_size):
    n = caffe.NetSpec()
    n.data, n.label = L.HDF5Data(batch_size=batch_size, source=hdf5, ntop=2)
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.relu1 = L.ReLU(n.pool1, in_place=True)
    n.conv2 = L.Convolution(n.relu1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.relu2 = L.ReLU(n.pool2, in_place=True)
    n.fc1 =   L.InnerProduct(n.relu2, num_output=500, weight_filler=dict(type='xavier'))
    n.score =   L.InnerProduct(n.fc1, num_output=10, weight_filler=dict(type='xavier'))
    #n.accuracy = L.Accuracy(n.score, n.label, include=[dict(phase=caffe.TEST)])
    n.loss =  L.SigmoidCrossEntropyLoss(n.score, n.label)
    return n.to_proto()

# Time stamp
ts = time.time()
time_stamp = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')

# Log/fig names with time stamp
net_prefix = "net_hdf5"
process_name = net_prefix + "_" + time_stamp
log_name = log_path + process_name + ".log"
print "Training process:", process_name

# New cpp solver
s = caffe_pb2.SolverParameter()

# Make prototxts
train_net_path, test_net_path, solver_config_path = get_locations(net_prefix)
fpath_train_list = (fpath_db+"h5list").format("train", "hdf5")
fpath_test_list = (fpath_db+"h5list").format("test", "hdf5")
with open(train_net_path, "w") as f:
    f.write(str(net_hdf5(fpath_train_list, 64)))
with open(test_net_path, "w") as f:
    f.write(str(net_hdf5(fpath_test_list, 100)))
make_solver(s, net_prefix, train_net_path, test_net_path, solver_config_path)

# Solve neural net and write to log
out = OutputGrabber(sys.stderr)
out.start()
print "Start training"
train_test_net_python(solver_config_path, 10000)
out.stop(log_name)
print "Stop training"

# Plot
#print_learning_curve(log_name, process_name)
