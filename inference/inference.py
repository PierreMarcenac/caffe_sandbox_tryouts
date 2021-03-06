'''
Created on Dec 3, 2015
@author: kashefy
'''
import numpy as np
import h5py
import lmdb
import caffe
from iow import read_lmdb, to_lmdb
from iow.lmdb_utils import MAP_SZ, IDX_FMT
from blobs.mat_utils import expand_dims

def infer_to_h5_fixed_dims(net, keys, n, dst_fpath, preserve_batch=False):
    """
    Run network inference for n batches and save results to file
    """
    dc = {k:[] for k in keys}
    for _ in range(n):
        d = forward(net, keys)
        for k in keys:
            if preserve_batch:
                dc[k].append(np.copy(d[k]))
            else:
                dc[k].extend(np.copy(d[k]))
            
    with h5py.File(dst_fpath, "w") as f:
        for k in keys:
            f[k] = dc[k]
            
    return [len(dc[k]) for k in keys]

def infer_to_lmdb(net, keys, n, dst_prefix):
    """
    Run network inference for n batches and save results to an lmdb for each key.
    Lower time complexity but much higher space complexity.
    
    Not recommended for large datasets or large number of keys
    See: infer_to_lmdb_cur() for slower alternative with less memory overhead
    
    lmdb cannot preserve batches
    """
    dc = {k:[] for k in keys}
    for _ in range(n):
        d = forward(net, keys)
        for k in keys:
            dc[k].extend(np.copy(d[k].astype(float)))
          
    for k in keys:
        to_lmdb.arrays_to_lmdb(dc[k], dst_prefix % (k,))
            
    return [len(dc[k]) for k in keys]

def infer_to_lmdb_cur(net, keys, n, dst_prefix):
    '''
    Run network inference for n batches and save results to an lmdb for each key.
    Higher time complexity but lower space complexity.
    
    Recommended for large datasets or large number of keys
    See: infer_to_lmdb() for faster alternative but with higher memory overhead
    
    lmdb cannot preserve batches
    '''
    dbs = {k : lmdb.open(dst_prefix % (k,), map_size=MAP_SZ) for k in keys}
    
    if len(keys) == 1:
        key_ = keys[0]
        num_written = _infer_to_lmdb_cur_single_key(net, key_, n, dbs[key_])
    else:
        num_written = _infer_to_lmdb_cur_multi_key(net, keys, n, dbs)
                    
    for k in keys:
        dbs[k].close()
        
    return num_written

def _infer_to_lmdb_cur_single_key(net, key_, n, db):
    '''
    Run network inference for n batches and save results to an lmdb for each key.
    Higher time complexity but lower space complexity.
    
    Takes advantage if there is only a single key
    '''
    idx = 0
    
    with db.begin(write=True) as txn:
        for _ in range(n):
            d = forward(net, [key_])
            l = []
            l.extend(d[key_].astype(float))
                    
            for x in l:
                x = expand_dims(x, 3)
                txn.put(IDX_FMT.format(idx), caffe.io.array_to_datum(x).SerializeToString())
                idx += 1
    return [idx]

def _infer_to_lmdb_cur_multi_key(net, keys, n, dbs):
    '''
    Run network inference for n batches and save results to an lmdb for each key.
    Higher time complexity but lower space complexity.
    
    See _infer_to_lmdb_cur_single_key() if there is only a single key
    '''
    idxs = [0] * len(keys)
    
    for _ in range(n):
        d = forward(net, keys)
        for ik, k in enumerate(keys):
            
            with dbs[k].begin(write=True) as txn:
            
                l = []
                l.extend(d[k].astype(float))
                        
                for x in l:
                    x = expand_dims(x, 3)
                    txn.put(IDX_FMT.format(idxs[ik]), caffe.io.array_to_datum(x).SerializeToString())
                    
                    idxs[ik] += 1
    return idxs

def forward(net, keys):
    '''
    Perform forward pass on network and extract values for a set of responses
    '''
    net.forward()
    return {k : net.blobs[k].data for k in keys}

def est_min_num_fwd_passes(fpath_net, mode_str):
    """
    if multiple source for same mode, base num_passes on last
    fpath_net -- path to network definition
    mode_str -- train or test?
    
    return
    minimum no. of forward passes to cover training set 
    """
    from proto.proto_utils import Parser
    np = Parser().from_net_params_file(fpath_net)
    
    num_passes = 0
    
    for l in np.layer:
        if 'data' in l.type.lower() and mode_str.lower() in l.data_param.source.lower():
            num_entries = read_lmdb.num_entries(l.data_param.source)
            num_passes = int(num_entries / l.data_param.batch_size)
            if num_entries % l.data_param.batch_size != 0:
                print("WARNING: db size not a multiple of batch size. Adding another fwd. pass.")
                num_passes += 1
            print("%d fwd. passes with batch size %d" % (num_passes, l.data_param.batch_size))
            
    return num_passes
            
def response_to_lmdb(fpath_net,
                     fpath_weights,
                     keys,
                     dst_prefix,
                     modes=[caffe.TRAIN, caffe.TEST],
                     ):
    """
    keys -- name of responses to extract. Must be valid for all requested modes
    """
    out = dict.fromkeys(modes)
    
    for m in modes:
        num_passes = est_min_num_fwd_passes(fpath_net, ['train', 'test'][m])
        out[m] = infer_to_lmdb(caffe.Net(fpath_net, fpath_weights, m),
                               keys,
                               num_passes,
                               dst_prefix + '%s_' + ['train', 'test'][m] + '_lmdb')
    return out

if __name__ == '__main__':
    
    base_path = '/mnt/scratch/pierre/caffe_sandbox_tryouts/'
    
    def path_to(path):
	return base_path + path

    fpath_net = path_to('learning_curve/prototxt/net0_test.prototxt')
    fpath_weights = path_to('learning_curve/snapshots/net0_snapshot_iter_10000.caffemodel')
    fpath_db = path_to('inference/mnist_%s_train_lmdb')
    
    net = caffe.Net(fpath_net, fpath_weights, caffe.TRAIN)
    keys = ['fc2', 'fc1']
    x = infer_to_lmdb_cur(net, keys, 2, fpath_db)
    
    import os
    print 'Do lmdbs exist?'
    print [os.path.isdir(fpath_db % (k,)) for k in keys]
    print 'Number of entries in lmdbs:'
    print [read_lmdb.num_entries(fpath_db % (k,)) for k in keys]

    # Here you need to compute accuracy, roc, precision recall, confusion matrix
    
    pass
