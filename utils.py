import json
import yaml
import pickle
import numpy as np


def load_json(filename):
    with open(filename, 'r') as f:
        #return yaml.safe_load(f)
        return json.load(f)

def save_json(dictionary, filename):
    with open(filename, 'w') as f:
        json.dump(dictionary, f)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def split_batch(data, ndevice):

    tb = len(data)

    split_batch = int(tb / ndevice)

    split_data = [data[i * split_batch : (i+1) * split_batch] for i in range(ndevice)]

    return split_data 
    
def load_string_from_h5(dataset, key):

    return

def write_object_to_h5(dataset, key, obj):

    return

def save_split_pickle(obj, filename, start, end):
    with open('{}-{}-{}'.format(filename, start, end), 'wb') as f:
        pickle.dump(obj, f)

    
if __name__ == '__main__':
    
    pass

def assign(g, node, data):
    data = np.array(data) if type(data) != np.ndarray else data
    if node in list(g.keys()):
        if g[node][...].shape == data.shape:
            g[node][...] = data
        else:
            del g[node]
            g[node] = data
    else:
        g[node] = data

