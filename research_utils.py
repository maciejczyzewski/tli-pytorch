import hashlib
import json


def create_hash(d):
    _hash = json.dumps(d, sort_keys=True)
    return str(int(hashlib.sha1(_hash.encode("utf-8")).hexdigest(), 16) % (10 ** 8))


class __lazy_init:
    def __init__(self, func, params=None):
        self.func = func
        self.name = func.__name__
        if not params:
            params = {}
        self.params = params
        try:
            self.hash = self.name + "/" + create_hash(params)
        except:
            self.hash = self.name

    def __getitem__(self, name):
        return self.params[name]

    def __call__(self, *args, **kwargs):
        _func_params = {
            key: self.params[key]
            for key in self.func.__code__.co_varnames[: self.func.__code__.co_argcount]
            if key in self.params
        }
        _func_params = {**_func_params, **kwargs}
        return self.func(*args, **_func_params)
