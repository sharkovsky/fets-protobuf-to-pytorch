from proto import load_proto, collaborator_aggregator_interface_pb2
import numpy as np
import os
import pickle

base_dir = '/home/fcremone/fl-experiments/fets/pretrained-brats/Models/fets_consensus_models/0/'

weights = dict()
for fpath in os.listdir(base_dir):
    print('Loading ', fpath)
    if fpath == 'ModelHeader.pbuf' or fpath == 'ExtraModelInfo.pbuf' or fpath[-5:] != '.pbuf':
        print('Skipped')
        continue
    tp = load_proto(os.path.join(base_dir, fpath), collaborator_aggregator_interface_pb2.TensorProto)
    weight = np.frombuffer(tp.data_bytes, dtype=np.float32)
    weight = weight.reshape(tp.shape)
    weights[tp.name] = weight

with open(os.path.join(base_dir, 'new_model_state_dict.pkl'), 'wb') as f:
    pickle.dump(weights, f)

