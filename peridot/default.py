import pyspark
import pyspark.sql.functions as F
from pyspark.sql.window import Window
# from pyspark.sql import DataFrame
from pyspark.sql.types import *
from delta.tables import DeltaTable

import pandas as pd
import numpy as np
import re

try:
    import plotly
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    pd.options.plotting.backend = "plotly"
except:
    print("plotly is not installed on this cluster")


## Get SparkSession to be able to use functions such as spark.sql or spark.sql without needing to set it as a parameter
from pyspark.sql import SparkSession
try:
    ## Spark 3
    spark = SparkSession.getActiveSession()
except:
    ## Spark 2
    spark = SparkSession.builder.getOrCreate()
