
def showConf():
    display(sc.parallelize(sorted(sc._conf.getAll())).toDF(["Parameters", "Settings"]))





# plotPartitionDistrib
