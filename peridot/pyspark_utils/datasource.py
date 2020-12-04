from ..default import *

def load_dataset(datasource, cols, tz, timestamp_var="recordDate",
                 additional_filters=[], additional_casts={}, persist=False, **kwargs):
    """
    Loads data from a data source, including conversion from timestamp to date and pre-processing by filtering and casting
    
    Parameters:
        datasource: Hive table name
        persist: whether or not to persist the resulting dataset
        additional_filters: list of filters to be applied
        additional_casts: list of casts to be applied
        timestamp_var: name of the timestamp variable to convert into date
        kwargs: other parameters such as the name of the timestamp to date conversion field (default: "date")
    
    Returns:     
        Spark dataframe
    
    Example:
        >>> load_dataset(datasource, cols, tz="Europe/Paris", timestamp_var=timestamp_var,
                         additional_filters=[], additional_casts={var_RING_NO: "int"}, persist=False) # for BYDAS tunnel data
    To do:
        Add * to load all columns
        Update parameters docstring
    """

    ## Special parameters
    props = {"date_var": "date"}
    props.update(kwargs)

    def jsoncol_rename(df):
        return df.select([col if '.' not in col else F.col(f"`{col}`").alias(col.split('.')[-1]) for col in df.columns])
    
    ## In case of variable name errors
    missing_variables = [col for col in cols if not any(col in source_col for source_col in spark.sql(f"SELECT * FROM {datasource}").columns)]
    if missing_variables:
        print(f"The following variables were not found in the datasource: {', '.join(missing_variables)}")
        return None
 
    ## Impacts on serialization, depending of the underlying data format
    all_cols = spark.sql(f"SELECT * FROM {datasource}").columns
    table_describe = spark.sql(f"describe extended {datasource}")
    underlying_format = table_describe.filter(F.col("col_name") == "Serde Library").select("data_type").collect()[0][0]
    if underlying_format == 'org.openx.data.jsonserde.JsonSerDe': #json format: variables are backticked
        cols = [[f"`{col}`" for col in all_cols if (col_selected == col) or (f".{col_selected}" in col)][0] for col_selected in cols]
    elif underlying_format == 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe': #csv format
        pass
    elif underlying_format == 'org.apache.hadoop.hive.ql.io.orc.OrcSerde': #orc format
        pass
    elif underlying_format == 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe': #parquet format
        pass
    
    ## Loading of the dataset 
    data = (jsoncol_rename(spark.sql(f"SELECT {', '.join(cols)} FROM {datasource}"))\
            # .dropna() # il faut le passer en paramètre + créer une fonction à passer dans un transform pour suppr col where all nulls avec retour de message
            .dropDuplicates()
            .withColumn(props["date_var"], F.from_unixtime(F.col(timestamp_var) / F.lit(1000.0)).cast("timestamp"))
            .withColumn(props["date_var"], F.from_utc_timestamp(F.col(props["date_var"]), tz))
            .withColumn(timestamp_var, F.col(timestamp_var).cast(LongType())))
    if additional_filters:
        for additional_filter in additional_filters:
            data = data.filter(additional_filter)
    if additional_casts:
        for col, cast_type in additional_casts.items():
            data = data.withColumn(col, F.col(col).cast(cast_type))
    data = data.orderBy(timestamp_var)
    if persist:
        data.persist()
    return data