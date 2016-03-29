import h5py

db_path = "/mnt/scratch/pierre/caffe_sandbox_tryouts/mnist/mnist_{0}_{1}/mnist_{0}.h5"

def augment_hdf5(phase, size):
    hdf5_db_path = db_path.format(phase, "hdf5")
    hdf5_augmented_db_path = db_path.format(phase, "hdf5_augmented")
    # Copier la non augmentee en la augmentee...
    f = h5py.File(hdf5_augmented_db_path, 'w')
    label, data = f['label'], f['data']
    for i, d in enumerate(f['data']):
        f['data'][i] = d

        # Ne pas append au hdf5 mais en fair eplusisuers et les lister
