from ..default import *
from ..python_utils.iterables import flattenIterable
from ..databricks_utils.utils import save_dataframe

from __future__ import annotations
from pyspark.sql import DataFrame as OriginalSparkDataFrame

from typing import Iterator, Iterable
import types

## An extended version of Spark's DataFrames
class DataFrame(OriginalSparkDataFrame):
    def __init__(self, df):
        super(self.__class__, self).__init__(df._jdf, df.sql_ctx)
    
    
    def applyPandasFunc(self, pd_func, col, partitionby, new_col=None, orderby=None, ascending=True, **params):
        """
        Apply a built-in pandas function (such as interpolation, ewm, rolling) or a lambda function for each group of a spark dataframe
        If pd_func is rolling, it is assumed to find a roll_agg_func variable of type function in params to specify the aggregation

        Parameters:
            pd_func (str or function): name of a pandas built-in function or lambda/numpy/scipy to be applied on col
            col (str or list of str): column, column name or list of one of them, to be processed
            new_col (str or list of str): column name or list of column names, as output of pd_func (optional, will overwrite the input column(s) if None)
            partitionby (str or list of str): column name or list of column names to create groups
            orderby (str or list of str): column name or list of column names to sort the wrapped pdf (groups)
            ascending (bool): to sort ascending or descending (default, True)
            params (kwargs): parameters of the pd_func

        Returns:     
            Spark DataFrame
        
        Examples:
            >>> df = spark.createDataFrame([(0, 1, 21., 12, 0., 1), (2, 2, 1., 12, 0.78, 2), (3, 2, None, 12, 1.57, 3),
                                            (2, 1, np.nan,  12, 5.49, 4), (3, 1, np.nan, 12, 6.28, 5), (5, 1, 30., 13, 0., 6),
                                            (1, 1, 30., 13, 0.78, 7), (1, 2, 75., 13, 1.57, 8), (10, 2, np.nan, 13, 5.49, 9),
                                            (100, 2, 98., 13, 6.28, 10)], ("order", "id", "age", "group", "angle", "num"))
            >>> display(df
                        .applyPandasFunc(pd.DataFrame.interpolate, "age", new_col="age_new", orderby="order", partitionby="id",
                         method="ffill")
                        .applyPandasFunc("interpolate", "age", new_col="age_new_2", orderby="order", partitionby="id", method="bfill")
                        .applyPandasFunc("cumsum", ["age", "id"], new_col=["age_new_3", "id_new_3"], orderby="order", ascending=False,
                         partitionby=["group", "age_new_2"], skipna=True)
                        .applyPandasFunc(pd.DataFrame.cumsum, ["age", "id"], new_col=["age_new_4", "id_new_4"], orderby="order",
                         ascending=False, partitionby=["group", "age_new_2"], skipna=True)
                        .applyPandasFunc("rolling", ["age_new", "id"], new_col=["age_new_5", "id_new_5"], orderby="order", partitionby="id",
                         window=3, win_type='gaussian', roll_agg_func="sum(std=2)")
                        .applyPandasFunc("std", "age_new", partitionby="id_new_3")
                        .applyPandasFunc(np.unwrap, "angle", new_col="unwrap", orderby="num", partitionby="group", axis=0)
                        .applyPandasFunc(np.unwrap, ["angle"], new_col=["unwrap_2"], orderby="num", partitionby="group", axis=0)
                        .orderBy("group", "num"))
        """
    
        def wrapper(pdf: pd.DataFrame) -> pd.DataFrame:
#             pdf[new_col].fillna(np.nan, inplace=True)
            if orderby:
                pdf.sort_values(by=orderby, ascending=ascending, inplace=True,
                                ignore_index=True) #groupby is causing a shuffle (ignore_index is the key feature)
            if isinstance(pd_func, str):
                pdf[new_col] = eval(f"pdf{new_col if len(new_col) == 1 else [new_col]}.{pd_func}(**{params})\
                                    {f'.{roll_agg_func}' if roll_agg_func else ''}")
            elif isinstance(pd_func, types.FunctionType):
                pdf[new_col] = eval(f"pd_func(pdf{new_col if len(new_col) == 1 else [new_col]}, **params)")
            return pdf

        ## pd_func extra parameters
        roll_agg_func = params.pop("roll_agg_func", None)
    
        ## Update schema: add new columns if necessary and returns names as strings for all columns
        self, col, new_col = self._pandasUpdateSchema(col, new_col)

        return self.groupBy(partitionby).applyInPandas(wrapper, schema=self.schema)


    def flatStruct(self, cols=None, remove_tree=False):
        """
        Flatten Spark Struct Columns
        
        Parameters:
            cols: List of Spark Columns
            remove_tree (bool): Whether or not to keep the name of the parents column
        
        Returns:     
            Modified input DataFrame
        
        Examples:
            >>> df = (df.withColumn("TEST", F.struct(F.lit(12).alias("Col_12"), F.lit(24).alias("Col_24")))
                        .withColumn("ANOTHER_TEST", F.struct(F.lit("A").alias("Col_A"), F.lit("B").alias("Col_B"))))
            >>> display(df.flatStruct(["ANOTHER_TEST"], remove_tree=False))
        """   
        
        if not cols:
            cols = self.columns
        ## Flatten Spark Struct columns (recursive)
        def flatten_allstruct(self, prefix="", remove_tree=False):
            res=[]
            for elem in self:
                col_name = f"{prefix}{elem.name}"
                if isinstance(elem.dataType, StructType):
                    res += elem.dataType.flatten_allstruct(f"{col_name}.", remove_tree=remove_tree)
                else:
                    res.append(F.col(col_name).alias((col_name if '.' not in col_name else col_name.split('.')[-1]) if remove_tree else col_name))
            return res
        StructType.flatten_allstruct = flatten_allstruct
        ## Flatten specific Spark Struct columns using flatten_allstruct
        def flatten_struct(self, cols, prefix="", remove_tree=False):
            res=[]
            res.extend([F.col(elem.name) for elem in self if elem.name not in cols])
            return res + StructType([elem for elem in self if elem.name in cols]).flatten_allstruct(remove_tree=remove_tree)
        StructType.flatten_struct = flatten_struct
        return self.select(self.schema.flatten_struct(cols=cols, remove_tree=remove_tree))


    def mapPandasFunc(self, pd_func, col, new_col=None, **params):
        """
        Apply a pandas function to column(s) of a Spark Dataframe

        Parameters:
            pd_func (str or function): function designed to run on pandas DataFrame
            col (str or list of str): column, column name or list of one of them, to be processed
            new_col (str or list of str): column name or list of column names, as output of pd_func (optional, will overwrite the input column(s) if None)
            params (kwargs): parameters of the pd_func

        Returns:     
            Spark DataFrame
        
        Examples:
            >>> df = spark.createDataFrame([(0, 1, 21., 12, 0., 1), (2, 2, 1., 12, 5.78, 2), (3, 2, None, 12, 1.57, 3),
                                            (2, 1, np.nan,  12, -5.49, 4), (3, 1, np.nan, 12, 6.28, 5), (5, 1, 30., 13, 0., 6),
                                            (1, 1, 30., 13, -0.78, 7), (1, 2, 75., 13, 1.57, 8), (10, 2, np.nan, 13, 5.49, 9),
                                            (100, 2, 98., 13, 46.28, 10)], ("order", "id", "age", "group", "angle", "num"))
            >>> display(df
                        .mapPandasFunc(lambda series: modulo(series, 2 * np.pi), "angle", "angle_mod")
                        .mapPandasFunc(modulo, F.col("angle"), "angle_mod_2", n=2*np.pi)
                        .mapPandasFunc(modulo, ["order", "id"], [F.col("ord_mod"), "id_mod"], n=2))
        """
        
        def wrapper(itr: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
            for pdf in itr:
                pdf[new_col] = pd_func(pdf[new_col], **params)
                yield pdf
        
        ## Update schema: add new columns if necessary and returns names as strings for all columns
        self, col, new_col = self._pandasUpdateSchema(col, new_col)
        
        return self.mapInPandas(wrapper, schema=self.schema)

    
    def melt(self: DataFrame, id_vars: Iterable[str], value_vars: Iterable[str],
            var_name: str="variable", value_name: str="value") -> DataFrame:
        """
        Unpivot a Pyspark Dataframe from wide to long format
        
        Credit:
            https://stackoverflow.com/questions/41670103/how-to-melt-spark-dataframe

        Parameters:
            id_vars: list of column names to use as identifier variables
            value_vars: list of column names to unpivot
            var_name: name of the column to be used for the variables column (default: "variable")
            value_name: name of the column to be used for the values  column (default: "value")
    
        Returns:     
            Unpivoted Dataframe
        """

        _vars_and_vals = F.array(*(F.struct(F.lit(col).alias(var_name), F.col(col).alias(value_name)) for col in value_vars))
        _tmp = self.withColumn("_vars_and_vals", F.explode(_vars_and_vals))
        cols = id_vars + [F.col("_vars_and_vals")[x].alias(x) for x in [var_name, value_name]]
        return _tmp.select(*cols)


    save = save_dataframe


    def stack(self, cols, stack_cols_dict={"key": "key", "value": "value"},
            rename=lambda col: re.sub(r"(\d+)", "", col).rstrip('_'), remove_tree=False):
        """
        Stack a Pyspark Dataframe from one column, several columns or
        a list of several columns that operate together
        
        Parameters:
            cols: Spark column or list of Spark columns or list of list of Spark columns
            stack_cols_dict: mapping dictionary to rename the columns
            rename: regex to be applied to cols to obtain the same name
    
        Returns:     
            Stacked Pyspark Dataframe
        """

        if isinstance(cols, str):
            return (self.selectExpr(*[col for col in self.columns if col != cols],
                    f"stack(1, '{cols}', {cols}) as ({stack_cols_dict['key']}, {stack_cols_dict['value']})"))
        elif isinstance(cols, list):
            if isinstance(cols[0], str):
                stack_cols = ", ".join([f"'{cols[i]}', {cols[i]}" for i, _ in enumerate(cols)])
            return (self
                    .selectExpr(*[col for col in self.columns if col not in cols],
                                f"stack({len(cols)}, {stack_cols}) as ({stack_cols_dict['key']}, {stack_cols_dict['value']})")
                    .filter(F.col(stack_cols_dict['value']).isNotNull()))
        ## fonctionne sur des noms de colonnes indexés avec un entier débutant à 1 (à généraliser en remplaçant le i du enumerate par une liste d'id => ammènerait à modifier la regex en passant la liste des id ou en générant la liste à partir d'une regex)
        elif isinstance(cols[0], (list, tuple)):
            stack_cols = ", ".join([f"'{i+1}', struct({', '.join([f'{col} as {rename(col)}' for col in cols_i])})" for i, cols_i in enumerate(cols)])
            self = (self.selectExpr(*[col for col in self.columns if col not in flattenIterable(cols)],
                                    f"stack({len(cols)}, {stack_cols}) as ({stack_cols_dict['key']}, {stack_cols_dict['value']})"))
            new_cols = re.sub(r"(:\w+|.*<)|>", "", self.select(stack_cols_dict['value']).schema.simpleString()).split(",")
            self = self.filter(F.coalesce(*[F.col(stack_cols_dict['value']).getItem(col) for col in new_cols]).isNotNull())
            return self.flatStruct([stack_cols_dict["value"]], remove_tree=remove_tree)


    def toList(self, col):
        """
        Transform a column to a list
        
        Parameters:
        col: Spark column
        
        Returns:     
        Python list
        """
        
        return list(self.select(col).toPandas()[col])


    def toLists(self, cols):
        """
        Transform each column to a list
        
        Parameters:
        cols: List of Spark columns
        
        Returns:     
        Tupple of Python lists
        """
        
        return tuple(self.toList(col) for col in cols)


    def unstack(self, index, pivot, fields=None, subpivot_val=None,
                agg_func=lambda col: F.first(col, ignorenulls=True), mapping=(lambda col: F.col(col))):
        """
        Unstack Pyspark Dataframe
        
        Parameters:
            index: List of Spark columns to consider as index
            pivot: Pyspark column name to consider as pivot
            fields: List of Pyspark column names to pivot (optional, all if None)
            subpivot_val: Limitation (sub-selection) of values to pivot (optional)
            agg_func: Function to be applied to the pivot results (default: first)
            mapping: Function to be applied to the pivot's resulting column names (default: values of the pivot column)
        
        Returns:     
            Unstacked Pyspark Dataframe around pivot column name
        """
        
        mapping = mapping(pivot)
        if not fields:
            fields = list(set(self.columns).difference(set(list(index) + [pivot])))
        return (self
                .withColumn(pivot, mapping)
                .groupBy(index) 
                .pivot(pivot, values=subpivot_val)
                .agg(*[agg_func(col).alias(col) for col in fields]))


    def transforms(self, func_lst):
        """
        Chain custom transformations with transform method applied on a list of functions
        
        Parameters:
        func_lst: list of functions that take and return a Spark DataFrame
        
        Returns:     
        DataFrame
        """

        for func in func_lst:
            self = self.transform(func)
        return self


    def withColumns(self, cols, func, new_cols=None):
        """
        Returns a new DataFrame by adding columns or replacing existing columns after applying the function 
        
        Parameters:
        cols: list of existing Spark columns
        func: function to apply on cols
        new_cols: optional list of columns to create (default: None, using cols to store results)
        
        Returns:     
        DataFrame
        """

        if not new_cols:
            new_cols = cols
        for col, new_col in zip(cols, new_cols):
            # self = self.withColumn(new_col, func(F.col(col) if type(col) == str else col))
            self = self.withColumn(new_col, func(col))
        return self
      

    ## Private functions

    def _pandasUpdateSchema(self, col, new_col):
        """
        Returns an updated schema of the input Spark DataFrame, according pandas function API to be applied
        Includes the formatting of the columns to be processed

        Parameters:
            self (Spark DataFrame): Spark DataFrame to update
            col (str or Column or list): pandas function API input to be formatted
            new_col (str or Column or list): pandas function API ouput to be formatted

        Returns:     
            Tuple of updated Spark DataFrame, input and output columns to process 
        """
        
        col = _colsAsListOfStrings(col)
        if new_col is None:
            new_col = col
        else:
            new_col = _colsAsListOfStrings(new_col)
            self = eval(f"self.withColumns({col}, lambda col: F.col(col), {new_col})")
        return self, col, new_col


## Private functions

def _colsAsListOfStrings(colArg):
    """
    Formats a column(s) input such as Spark column or Spark column name or list of these, as a list of column name(s)

    Parameters:
        colArg (str or Column or list): function input to be formatted

    Returns:     
        List of column names
    """

    toString = lambda cols: [elem if not isinstance(elem, pyspark.sql.column.Column)
                            else elem.__repr__().split("'")[1] for elem in cols]
    if not isinstance(colArg, list):
        colArg = [colArg]
    return toString(colArg)


## Modifies the methods that return Spark DataFrames to instanciate the extended peridot Spark DataFrame
def _wrapDataFrameMethods(cls, DataFrame_methods_wrapper):
    for key, value in cls.__dict__.items():
        if hasattr(value, '__call__'):
            setattr(cls, key, _dfMethodsDecorator(value))

def _dfMethodsDecorator(func):
    def call(*args, **kwargs):
        result = func(*args, **kwargs)
        if type(result) is pyspark.sql.dataframe.DataFrame:
            return DataFrame(result) 
        else:
            return result
    return call

for cla in [pyspark.sql.session.SparkSession, pyspark.sql.readwriter.DataFrameReader, DataFrame, OriginalSparkDataFrame]:
    _wrapDataFrameMethods(cla, _dfMethodsDecorator)
