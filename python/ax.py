from contextlib import contextmanager
from arrayx import Backend


@contextmanager
def context():
    try:
        Backend.init()
        yield
    finally:
        Backend.cleanup()
