


def random_hexcolor(n=1, seed=None):
    """
    Get random colors in hexadecimal format
    
    Parameters:
       n: number of columns to return
    
    Returns:     
       List of hexadecimal colors
    """
    
    import random
    random.seed(seed)
    def hexcolor(): return f"#{random.randint(0, 0xFFFFFF):06x}"
    return hexcolor() if n == 1 else [hexcolor() for i in range(n)]