# Two functions to save and load a python object
# A. Ireson
import pickle
import json
import numpy as np

def save(obj,name):
    if not(name.lower()[-4:]=='.pkl'): name=name+'.pkl'
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load(name):
    if not(name.lower()[-4:]=='.pkl'): name=name+'.pkl'
    with open(name, 'rb') as f:
        return pickle.load(f)

def dict2json(d,fn):
    # Save a dictionary to a json text file
    for key in d:
        if isinstance(d[key],np.ndarray):
            d[key]=d[key].tolist()
    f=open(fn,'w')
    f.write(json.dumps(d,indent=2))
    f.close()

def json2dict(fn):
    # Load a json text file into a dictionary variable
    f=open(fn,'r')
    d=json.load(f)
    f.close()
    for key in d:
        if isinstance(d[key],list):
            d[key]=np.array(d[key])
    return d

