from peridot.default import *

from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.column import Column, _to_java_column, _to_seq

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


def cumsum(col, orderby, partitionby=F.lit(True)):
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
        col: Spark column or column name (can be either numerical, string, stuct, expression...)
        expr (str): sign of comparison to be evaluated as a condition for id change
        orderby: definition of the ordering
        partitionby: definition of the partitioning (default: no partition)
    
    Returns:     
        Spark column
    """

    col = _colAsStringOrColumn(col)
    win = Window.partitionBy(partitionby).orderBy(orderby) if partitionby else Window.orderBy(orderby)
    return (F.last(F.when(eval(f"col {expr} F.lag(col).over(win)") |
                          (F.row_number().over(win) == F.lit(1)),
                          F.monotonically_increasing_id() + F.lit(1))
                    .otherwise(F.lit(None)), ignorenulls=True)
            .over(win.rowsBetween(Window.unboundedPreceding, Window.currentRow)))

# F.incrementalCut = incrementalCut


def mapPandas(cols, func, returntype, row_func=False, **params):
    """
    Apply a pandas Series function to one or more Spark columns
    Can be used as an aggregation function and mixed with the native functions
    by first performing a collect_list (possibly over a partition)

    Parameters:
        cols (str or column or list): column, column name or list of them, to be processed
        func (function): function to be applied
        returntype (pyspark.sql.types): type of the returned Spark column
        row_func (bool): for simplification purposes, when the func expression is ambiguous or can not be applied directly on a series,
        this variable is designed to underneath modify the function and take a pandas Series or a pandas Row as input
        params (kwargs): parameters of the func

    Returns:     
        Spark Column
    
    Examples:
        >>> display(spark.createDataFrame([(1, 1.0), (1, 2.0), (2, 3.0), (2, 5.0), (2, 10.0)], ("id", "v"))
                    .withColumn("tutu", F.mapPandas(F.collect_list("v").over(Window.partitionBy("id")),
                                                    lambda s: s.apply(np.mean), DoubleType())))
        >>> display(spark.range(5, numPartitions=1)
                         .select(F.col("id"),
                                 F.mapPandas([F.col("id"), "id"], lambda cols: np.add(*cols), returntype=IntegerType()),
                                 F.mapPandas([F.sqrt("id")], np.round, returntype=DoubleType(), decimals=2).alias("sqrt_round"),
                                 F.mapPandas("id", lambda series: series.apply(lambda val: pd.Timestamp(val, unit="s")
                                                                                           .strftime("%Y-%m-%d %H:%M:%S")),
                                             returntype=StringType()).alias("date"),
                                 F.mapPandas("id", pd.Series.diff, returntype=IntegerType(), periods=2).alias("diff")))
        >>> display(sc.parallelize([[.1, 1, 1], [.2, 2, 1], [.3, 3, 1], [.4, 4, 1], [.5, 5, 1], [.6, 6, 1], [.1, 7, 2], [.2, 8, 2],
                                    [.3, 9, 2], [.4, 10, 2], [.3, 11, 2], [.4, 12, 2], [None, 13, 2], [None, None, None], 
                                    [.1, 14, 3], [.2, 15, 3], [.3, 16, 3], [.4, 17, 3]]).toDF(("PM_RING", "date", "ring"))
                    .withColumn("list", F.collect_list(F.least("PM_RING", "date", "ring")).over(Window.rowsBetween(-2, Window.currentRow)))
                    .withColumn("min2", F.mapPandas("list", np.min, row_func=True, returntype=FloatType(), initial=.35))
                    .withColumn("diff2", F.mapPandas("min2", pd.Series.diff, returntype=FloatType(), periods=2))
                    .withColumn("datetime", F.mapPandas(F.when(F.col("date").isNotNull(), F.col("date")).otherwise(F.lit(0)),
                                                        lambda series: series.apply(lambda val: pd.Timestamp(val, unit="s")
                                                                                    .strftime("%Y-%m-%d %H:%M:%S")), returntype=StringType()))
                    .withColumn("datetime2", F.mapPandas(F.when(F.col("date").isNotNull(), F.col("date")).otherwise(F.lit(0)),
                                                         lambda val, unit: pd.Timestamp(val, unit=unit).strftime("%Y-%m-%d %H:%M:%S"),
                                                         returntype=StringType(), row_func=True, unit="h")))
        >>> display(spark.createDataFrame([(1, 1.0), (1, 2.0), (2, 3.0), (2, 5.0), (2, 10.0)], ("id", "v"))
                    .groupby("id").agg(F.max("v"),
                                       *[F.mapPandas(F.collect_list("v"),
                                                                 lambda s: s.apply(func), DoubleType()).alias(name)
                                                     for func, name in [(np.mean, "mean"), (np.median, "median"), (np.max, "max"),
                                                                        (partial(np.quantile, q=0.5), "median")]]))
    """

    @pandas_udf(returntype)
    def wrapper(iterator: Iterator[Tuple[pd.Series, ...]]) -> Iterator[pd.Series]:
        for columns in iterator:
            yield (lambda cols: pd.concat([*cols], axis=1).apply(lambda row: func(row, **params), axis=1) if row_func
                                else func(cols, **params))(columns)
    return wrapper(*_colsAsListOfColumns(cols))

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


def median(col):
    return _sqlFunc("percentile", [_colAsStringOrColumn(col), F.lit(0.5)])

F.median = median


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


def _sqlFunc(func, *params):
    """
    SQL aggregate function called by name to run over groubBy

    Parameters:
        func (str): SQL function name

    Returns:     
        Spark column
    """
    
    return Column(spark.sparkContext._jvm.org.apache.spark.sql.functions.callUDF(func,
                                                                                 _to_seq(sc, *params, _to_java_column)))