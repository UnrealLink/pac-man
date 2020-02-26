import numpy as np

def index_sum(idx1, idx2):
    """
    Sum two tuples
    """
    return (idx1[0] + idx2[0], idx1[1] + idx2[1])

class InvalidIndex(Exception):
    pass