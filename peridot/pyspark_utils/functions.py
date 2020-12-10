from peridot.default import *

from pyspark.sql.functions import *

## new SQL column functions like

def arrayFunc(col, func, returntype=DoubleType(), **params):
    """
    Apply a python function to a Spark ArrayType column via a pandas_udf 
    
    Parameters:
        col: column name
        func: function to be applied
    
    Returns:     
        Spark DoubleType column
    
    Examples:
        >>> lambda col: F.arrayFunc(col, lambda arr: arr[-win_len:], ArrayType(DoubleType()))
        >>> F.arrayFunc(F.arrayWithoutNone(F.col(col_1), F.col(col_2), ...), np.max)
        >>> F.arrayFunc("col", lambda arr: np.median(np.abs(arr - np.median(arr)))) #median absolute deviation
        >>> df = sc.parallelize([[.1, 1, 1], [.2, 2, 1], [.3, 3, 1], [.4, 4, 1], [.5, 5, 1], [.6, 6, 1],  [.1, 7, 2], [.2, 8, 2],
                                [.3, 9, 2], [.4, 10, 2], [.3, 11, 2], [.4, 12, 2], [None, 13, 2], [None, None, None], 
                                [.1, 14, 3], [.2, 15, 3], [.3, 16, 3], [.4, 17, 3]]).toDF(("PM_RING", "date", "ring"))
        >>> display(df.withColumn("test", F.arrayFunc(F.collect_list(F.least("PM_RING", "date", "ring"))
                    .over(Window.rowsBetween(-2, Window.currentRow)), np.min)))
    """
    
    len0_var = [] if type(returntype) == pyspark.sql.types.ArrayType else np.nan
    
    def array_func(s: pd.Series) -> pd.Series:
        return s.apply(lambda arr: func(arr, **params) if len(arr) != 0 else len0_var)
    return array_func(col)

# F.arrayFunc = arrayFunc


def arrayWithoutNone(func):
    """
    Decorator over F.array to remove None values
    
    Parameters:
        func: F.array by default
    
    Returns:     
        Function
    """

    def wrapper(*args):
        return F.array_except(func(*args), F.array(F.lit(None)))
    return wrapper

# F.arrayWithoutNone = arrayWithoutNone(F.array)


def cumsum(col, orderby, partitionby=None):
    """
    Computes the cumulative sum of a Pyspark column
    
    Parameters:
        col: Spark column or column name
        orderby: definition of the ordering
        partitionby: definition of the partitioning (default: None)
    
    Returns:     
        Spark column
    """

    col = F.col(col) if type(col) == str else col
    win = (Window.partitionBy(partitionby).orderBy(orderby) 
                 .rangeBetween(Window.unboundedPreceding, Window.currentRow))
    return F.sum(col).over(win)

# F.cumsum = cumsum


def diff(col, orderby, offset=1, partitionby=[]):
    """
    Calculates the difference of a Spark column element compared with another element, according to an offfset
    
    Parameters:
        col: Spark column or column name
        offset: number of rows to shift for calculating difference (default: 1)
        orderby: definition of the ordering
        partitionby: definition of the partitioning (default: no partition)
    
    Returns:     
        Spark column
    """
    col = F.col(col) if type(col) == str else col
    return col - F.lag(col, offset).over(Window.partitionBy(partitionby).orderBy(orderby))

# F.diff = diff


def incrementalCut(col, expr, orderby, partitionby=None):
    """
    Time series segmentor based on variation of consecutives values
    Returns a unique id for each interval without variation
    
    Parameters:
        col: Spark column or column name
        expr (str): expression to be evaluated as a condition for group change
        orderby: definition of the ordering
        partitionby: definition of the partitioning (default: no partition)
    
    Returns:     
        Spark column
    """

    col = F.col(col) if type(col) == str else col
    win = Window.partitionBy(partitionby).orderBy(orderby) if partitionby else Window.orderBy(orderby)
    return (F.last(F.when(eval("(col - F.lag(col).over(win))" + expr) |
                          (F.row_number().over(win) == F.lit(1)),
                          F.monotonically_increasing_id() + F.lit(1))
                    .otherwise(F.lit(None)), ignorenulls=True)
            .over(win.rowsBetween(Window.unboundedPreceding, Window.currentRow)))

# F.incrementalCut = incrementalCut


def mapPandas(col, func, returntype, **params):
    """
    Apply a pandas Series function to a Spark column

    Parameters:
        col (str or Column): column, column name, to be processed
        func (function): function to be applied
        params (kwargs): parameters of the func

    Returns:     
        Spark Column
    
    Examples:
        >>> display(spark.range(5, numPartitions=1)
                         .select(F.col("id"),
                                 F.mapPandas("id", lambda series: series.apply(lambda val: pd.Timestamp(val, unit="s")
                                                                                           .strftime("%Y-%m-%d %H:%M:%S")),
                                             returntype=StringType()).alias("date"),
                                 F.mapPandas("id", pd.Series.diff, returntype=IntegerType(), periods=2).alias("diff")))
    """
    
    def wrapper(s: pd.Series) -> pd.Series:
        return F.pandas_udf(lambda col: func(col, **params),
                            returnType=returntype)(_colAsStringOrColumn(col))
    return wrapper(col)

# F.mapPandas = mapPandas
## A généraliser avec F.arrayFunc ?


def mapWithDict(col, dico, returntype):
    """
    Map values of a dataframe column from a dictionary

    Parameters:
        col: Spark column or column name
        dico (dict): dictionary to use as mapping
    
    Returns:     
        Spark column
    """

    #Eventuellemnt mettre returntype à None,
    # et calculer le mode des types (retourner un message informatif si le returntype est différent)
    # spark_output_type = pyspark.sql.types._type_mappings[set(map(type, dico.values())).pop()].typeName()
    return F.mapPandas(col, pd.Series.map, arg=dico, returntype=returntype)

# F.mapWithDict = mapWithDict


## Private functions

def _colAsStringOrColumn(colArg):
    """
    Returns a Spark column object whatever the argument by which it was passed

    Parameters:
        colArg (str or Column): function input to be formatted

    Returns:     
        Spark column
    """

    return F.col(colArg) if isinstance(colArg, str) else colArg