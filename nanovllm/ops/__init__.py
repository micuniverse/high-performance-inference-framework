from .build import load_ops

_ops = None

def get_ops():
    global _ops
    if _ops is None:
        _ops = load_ops()
    return _ops