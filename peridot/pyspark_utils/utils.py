
def showConf():
    display(spark.createDataFrame(sorted(sc._conf.getAll()), ["Parameters", "Settings"]))





# plotPartitionDistrib
