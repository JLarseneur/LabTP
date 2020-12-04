

## Unstacking of indices
def unstack_indices(df_hpc_source, var_to_consider):
  df_hpc_select = (df_hpc_source
                   .unstack(index=[timestamp_var, date_var, "system_id"], pivot="indice_name", fields=["value"],
                            subpivot_val=main_vars)
                   # Remove outliers
                   .withColumns(var_to_consider,
                                lambda col: F.when(F.col(col) == F.lit(2147483.75), None)
                                             .otherwise(F.col(col)))
                   # Partitionning by system_id
                   .repartition("system_id")
                   .orderBy(timestamp_var))
  return df_hpc_select

## Initialize df_hpc_resample with time index
def create_time_index_v1(start, end, cranes_list):
  time_df = spark.createDataFrame(pd.date_range(start, end,
                                                freq="S")#, tz=tz)
                                  .to_frame(), [date_var]).cache()
  df_hpc_resample = None
  for crane in cranes_list:
    df_hpc_resample_crane = time_df.withColumn("system_id", F.lit(crane))
    if df_hpc_resample:
      df_hpc_resample = (df_hpc_resample
                         .union(df_hpc_resample_crane)
                         .repartition("system_id")
                         .cache())
    else:
      df_hpc_resample = df_hpc_resample_crane
  time_df.unpersist()
  df_hpc_resample_crane.unpersist()
  del time_df, df_hpc_resample_crane
  return df_hpc_resample

to_timestamp_format = lambda date: int(time.mktime(datetime.strptime(date, "%Y-%m-%d %H:%M:%S").timetuple()))
def create_time_index_v2(start, end, cranes_dict, timestamp_var=timestamp_var):
  return (spark
          .range(to_timestamp_format(start), to_timestamp_format(end) + 1, numPartitions=50)
          .rdd.map(lambda row: (datetime.fromtimestamp(row["id"]), *cranes_dict.keys))
          .toDF([timestamp_var, *cranes_dict.values])
          .stack([*cranes_dict.values], stack_cols_dict={"key": "Crane", "value": "system_id"})
          .repartition("system_id"))
#           .orderBy("timestamp"))

#v3
def create_time_index(start, end, cranes_dict, date_var=date_var):
  return (spark
          .range(to_timestamp_format(start), to_timestamp_format(end) + 1, numPartitions=50)
          .select(F.mapPandas("id", lambda series: series.apply(lambda val: pd.Timestamp(val, unit="s")
                                                                              .strftime("%Y-%m-%d %H:%M:%S")),
                              returntype=StringType()).alias(date_var),
                  *[F.lit(k).alias(v) for k, v in cranes_dict.items()])
          .stack([*cranes_dict.values()], stack_cols_dict={"key": "Crane", "value": "system_id"})
          .repartition("system_id")
          .drop("Crane")#en attendant d'avoir intégré les implications du field Crane déjà créé
          .cache())

## Resampling data
def resample(df_hpc_select, df_hpc_resample,
               dead_band_time_dict=dead_band_time_dict,
               date_var=date_var, timestamp_var=timestamp_var):
  ## Process by indice
  for indice in dead_band_time_dict.keys():
    ## Select indice
    df_indice_fill = (df_hpc_select
                      .select("system_id", date_var, indice, timestamp_var)
                      .dropna())
    ## Case no record
    if len(df_indice_fill.head(1)) == 0:
      print(indice, "is empty")
      continue
    ## Unwrap for orientation indice
    if indice == "orientation":
      df_indice_fill = (df_indice_fill
                        .applyPandasFunc(np.unwrap, indice, orderby=date_var, partitionby="system_id", axis=0))
    ## Add time_interval between two records
    fill_limit = 601
    df_indice_fill = (df_indice_fill
                      .withColumn(f"time_interval_{indice}", F.diff(timestamp_var,
                                                                    orderby=date_var, partitionby="system_id") / F.lit(1000))
                      .drop(timestamp_var)
                      # Replace None with fill_limit, so as not to backfill indefinitely beyond the first value
                      .withColumn(f"time_interval_{indice}", F.when(F.col(f"time_interval_{indice}").isNull(),
                                                                    F.lit(fill_limit))
                                                              .otherwise(F.col(f"time_interval_{indice}"))))
    ## Join with df_hpc_resample
    df_hpc_resample = df_hpc_resample.join(df_indice_fill, on=["system_id", date_var], how='left')
    ## Resample data
    fill_mode_lst = [fill_mode for fill_mode in dead_band_time_dict[indice].values() if fill_mode]
    policy_mode_lst = [(filling_policy, fill_mode)
                       for filling_policy, fill_mode in dead_band_time_dict[indice].items() if fill_mode]
    df_hpc_resample = (df_hpc_resample
                       # Forwardfill time_interval
                       .withColumn(f"time_interval_{indice}", F.when(F.col(f"time_interval_{indice}").isNull(),
                                                                     F.first(f"time_interval_{indice}", ignorenulls=True)
                                                                      .over(Window.partitionBy("system_id").orderBy(date_var)
                                                                                  .rowsBetween(Window.currentRow,
                                                                                               fill_limit - 1)))
                                                               .otherwise(F.col(f"time_interval_{indice}")))
                       # Method of filling according to the time interval between two records
                       .transforms((lambda df: df.applyPandasFunc("interpolate", indice,
                                                                  new_col=f"{indice}_fill_{fill_mode}",
                                                                  orderby=date_var, partitionby="system_id",
                                                                  method=fill_mode, limit=fill_limit)
                                    for fill_mode in fill_mode_lst))
                       .withColumn(f"{indice}_fill", F.lit(None))
                       .withColumns(policy_mode_lst,
                                    lambda policy_mode: F.when(eval(f"F.col('time_interval_{indice}') {policy_mode[0]}"),
                                                                             F.col(f"{indice}_fill_{policy_mode[1]}"))
                                                                       .otherwise(F.col(f"{indice}_fill")),
                                    [f"{indice}_fill"] * len(policy_mode_lst))
                       .drop(indice, f"time_interval_{indice}", *[f"{indice}_fill_{fill_mode}"
                                                                  for fill_mode in fill_mode_lst])
                       .cache())
  return df_hpc_resample

## Processing of cranes' states of use
def affect_states(df_hpc_resample):
  working = 3 * 60
  win = Window.orderBy(date_var).partitionBy("system_id")
  mov_indices = [f"{col}_fill" for col in ["lifting_height", "orientation", 'lifting_angle', "distribution", "translation"]
                 if f"{col}_fill" in df_hpc_resample.columns]
  
  df_processing = (df_hpc_resample
                   .repartition("system_id")#, F.year(date_var), F.dayofyear(date_var))
                   .withColumn("wind_off",
                               F.when((F.col("engine_on_fill") == F.lit(1)) &
                                      (F.col("weather_vane_fill") == F.lit(1)), F.lit(1))
                                .otherwise(0))
                   .withColumn("movement",
                               F.when((F.col("engine_on_fill") == F.lit(1)) &
                                      (F.col("wind_off") == F.lit(0)) &
                                      #a movement of one of the 3 segments + translation
                                      (F.greatest(*[(F.diff(col, orderby=date_var,
                                                            partitionby="system_id") != F.lit(0)).astype(IntegerType())
                                                    for col in mov_indices]) == F.lit(1)),
                                      F.lit(1))
                                .otherwise(F.lit(0)))
                   .withColumn("working",
                               F.when((F.col("engine_on_fill") == F.lit(1)) &
                                      (F.col("wind_off") == F.lit(0)) &
                                      #no movement but has moved in the last 3 minutes or is under load
                                      ((F.col("movement") == F.lit(0)) &
                                       ((F.max("movement").over(win.rowsBetween(-working+1, -1)) == F.lit(1)) |
                                       (F.col("load_fill") != F.lit(0)))), F.lit(1))
                                .otherwise(F.lit(0)))
                   .withColumn("stand_by",
                               F.when((F.col("engine_on_fill") == F.lit(1)) &
                                      (F.col("movement") == F.lit(0)) &
                                      (F.col("wind_off") == F.lit(0)) &
                                      (F.col("working") == F.lit(0)), F.lit(1))
                                .otherwise(0))
                   .withColumn("free_slew",
                               F.when(#(F.col("wind_off") == F.lit(0)) &
                                      (F.col("engine_on_fill") == F.lit(0)) &
                                      (F.col("weather_vane_fill") == F.lit(1)), F.lit(1))
                                .otherwise(0))
                   .withColumn("free_slew_off",
                               F.when((F.col("engine_on_fill") == F.lit(0)) &
                                      (F.col("weather_vane_fill") == F.lit(0)), F.lit(1))
                                .otherwise(0))
                   .withColumn("no_data",
                               F.when((F.col("movement") == F.lit(0)) &
                                      (F.col("wind_off") == F.lit(0)) &
                                      (F.col("stand_by") == F.lit(0)) &
                                      (F.col("working") == F.lit(0)) &
                                      (F.col("free_slew") == F.lit(0)) &
                                      (F.col("free_slew_off") == F.lit(0)), F.lit(1))
                                .otherwise(0)))
  return df_processing

## Rename fields and fill gaps at midnight
def format_for_cosmosdb(df_processing):
  sqlContext.sql("set spark.sql.caseSensitive=true")
  rename = {"system_id": "system_id",
            "date": "date",
            "wind_off": "Wind off",
            "movement": "Movement",
            "working": "Working",
            "stand_by": "Stand by",
            "free_slew": "Free slew",
            "free_slew_off": "Free slew off",
            "no_data": "No data"}
  df_state = (df_processing
              #drop _fill columns and rename
              .select([F.col(col).alias(rename[col]) for col in
                       df_processing.select(df_processing.colRegex("`^((?!_fill).)*$`")).columns])
              .withColumn("state", F.coalesce(*[F.when(F.col(col) != F.lit(0), F.lit(col))
                                                for col in list(rename.values())[2:]]))
              .drop(*list(rename.keys())[2:])
#               .transform(lambda df: StringIndexer(inputCol="state", outputCol="state_pass").fit(df).transform(df)) #not whitelisted
              .withColumn("state_pass", F.mapWithDict("state", {v: k+1 for k, v in enumerate(list(rename.values())[2:])},
                                                      returntype=IntegerType()))
              .withColumn("state_pass", F.incrementalCut(F.col("state_pass"), "!= 0", orderby=date_var,
                                                         partitionby="system_id"))
              ## Processing of no data at midnight
              .withColumn("state", F.when((F.max((F.col("state") == F.lit("No data")) &
                                                 (F.lag("state", 1).over(Window.partitionBy("system_id").orderBy(date_var))
                                                  .contains("Free slew"))).over(Window.partitionBy("state_pass"))) &
                                          (F.max((F.col("state") == F.lit("No data")) &
                                                 (F.lag("state", -1).over(Window.partitionBy("system_id").orderBy(date_var))
                                                  .contains("Free slew"))).over(Window.partitionBy("state_pass"))) &
                                          (F.hour(F.min(date_var)
                                                   .over(Window.partitionBy("state_pass"))).isin([23, 0])) &
                                          (F.hour(F.max(date_var)
                                                   .over(Window.partitionBy("state_pass"))) == F.lit(1)),
                                          F.array_except(F.collect_set(F.lag("state", 1).over(Window.partitionBy("system_id")
                                                                                                    .orderBy(date_var)))
                                                          .over(Window.partitionBy("state_pass")), F.array(F.lit("No data")))
                                           .getItem(0))
                                    .otherwise(F.col("state"))))
  return df_state

## Returns a daily aggregated dataset
def daily_aggregate(df_state):
  df_state = (df_state
              .groupBy("system_id", F.year(date_var).alias("Year"), F.dayofyear(date_var).alias("Dayofyear"),
#                        F.hour(date_var).alias("Hours"),
                       F.col("state").alias("State"))
              .agg(F.from_unixtime(F.unix_timestamp(F.min(date_var)), "yyyy-MM-dd HH:mm:ss").alias("Date"),
                   F.month(F.first(date_var)).alias("Month"),
                   F.weekofyear(F.first(date_var)).alias("Week"),
                   F.date_format(F.first(date_var), 'EEEE').alias("Dayofweek"),
                   F.date_format(F.first(date_var), 'MMMM').alias("Monthofyear"),
                   F.dayofmonth(F.first(date_var)).alias("Dayofmonth"),
#                    F.count("state_pass").alias("count_sec"),
                   (F.count("state_pass") / 3600).alias("count_hour"))
              .withColumn("id", F.concat_ws("_", *["system_id", "State", "Year", "Dayofyear"]))#, "Hours"]))
              .withColumn("Crane", F.mapWithDict("system_id", crane_id_dict, returntype=StringType()))
              .withColumn("OnOff", F.when(F.col("State") != "No data",
                                           F.when(F.col("State").isin(["Movement", "Working", "Stand by"]), F.lit("On"))
                                            .otherwise(F.lit("Off")))
                                     .otherwise(F.lit("No data")))
              .orderBy("system_id", "Year", "Dayofyear"))#, "Hours"))
  return df_state
