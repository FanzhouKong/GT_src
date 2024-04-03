import json
import hashlib
import numpy as np
import pickle
import datetime
import base64


def smart_io(filename, mode):
    if filename.endswith(".gz"):
        import gzip
        f = gzip.open(filename, mode)
    else:
        f = open(filename, mode)
    return f


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, bytes):
            return base64.b64encode(obj).decode("utf-8")
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat()
        elif np.isnan(obj):
            return None
        return super(NumpyEncoder, self).default(obj)


def md5_for_file(filename):
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def md5_for_object(object):
    hash_md5 = hashlib.md5()
    if isinstance(object, str):
        hash_md5.update(object.encode("utf-8"))
    elif isinstance(object, bytes):
        hash_md5.update(object)
    else:
        hash_md5.update(pickle.dumps(object))
    return hash_md5.hexdigest()
