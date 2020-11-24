
def flattenIterable(*args):
    """
    Flatten any combination of iterables
    
    Returns:
       Yield a generator of unpacked elements
    """
    return (res for elem in args for res in
            (flattenIterable(*elem) if isinstance(elem, (tuple, list, dict)) else (elem,)))