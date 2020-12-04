

def modulo(series: pd.Series, n: float) ->  pd.Series:
    """
    Modulo operation that works on real numbers, designed to work with pandas udf
    
    Parameters:
        series (pd.Series): pandas Series that contains numbers to compute
        n (float): divisor
    
    Returns:     
        pandas Series
    """
    
    res = np.fmod(series, n)
    res[res < 0] = res + n
    return res

## mad