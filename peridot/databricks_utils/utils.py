from ..default import *

def load(path, from_type="csv", to="pandas", dbfs_path="/mnt/tunnelab/outputs/"):
    """
    Load as pandas or spark dataframe

    Parameters:
       path: absolute path from 'outputs' exchange directory on dbfs (dbfs_path)
       to: data type to convert into ("pandas", "spark")
       dbfs_path: address of the exchange directory on dbfs 

    Returns:
        Pandas or spark dataframe
    
    Examples:
        >>> load("D168-IRTS_Safeset/df_IRTS_safeset_trend.csv", "pandas")
        >>> display(load("D168-IRTS_Safeset/df_IRTS_safeset_trend.csv", "spark"))
    """

    if to == "pandas":
        return pd.read_csv(f"/dbfs{dbfs_path}{path}", header='infer')
    elif to == "spark":
        return spark.read.format("csv").option("header", "true").load(f"dbfs:{dbfs_path}{path}")
    elif to == "delta":
        pass
    else:
        pass

## généraliser en load_file (prendre l'extension pour gérer le type)

def plot(fig, filename="temp-plot.html", output="dbfs:/mnt/tunnelab/outputs/", save=False):
    """
    Rendering (and saving on dbfs) either a matplotlib, a seaborn or a plotly figure

    Parameters:
       fig: figure to render (and save)
       filename: name to give to the file
       output: directory to be used for backup on dbfs
       save: weither to save or not to save the plot
    
    Returns:     
       
    """

    import plotly, matplotlib, plotly.express, seaborn
    ## Display figure
    if type(fig) == plotly.graph_objs._figure.Figure:
        if save:
            displayHTML(plotly.offline.plot(fig, output_type='file', filename=filename))
        displayHTML(plotly.offline.plot(fig, output_type='div'))
    elif type(fig) == matplotlib.figure.Figure:
        if save:
            fig.savefig(filename)
        display(fig)
    elif type(fig) in [eval(f"seaborn.axisgrid.{grid}") for grid in
                        [grid for grid in dir(seaborn.axisgrid) if grid[-4:] == "Grid"]]:
        if save:
            fig.savefig(filename)
        display(fig.fig)
    ## Save figure to dbfs
    if save:
        dbutils.fs.cp(f"file:/databricks/driver/{filename}", f"{output}/{filename}")
        dbutils.fs.rm(f"file:/databricks/driver/{filename}")


def save_dataframe(self, path="", convert_to="csv", dbfs_path="", **kwargs):
    """
    Save a dataframe

    Parameters:
       path (str): (if necessary, depending on output datatype) absolute path from 'outputs' exchange directory on dbfs (dbfs_path)
       convert_to (str): data type to convert into ("csv", "json", "parquet", "cosmosdb", "delta")
       dbfs_path (str): address of the exchange directory on dbfs 

    Returns:
        Pandas or spark dataframe
    
    Examples:
        >>> pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]}).save(path="test.csv")
        >>> df.save(convert_to="cosmosdb", **{"mode": "append", "options": cosmos_conf})
        >>> df.save(path="D147-HPC_Cranes/HPC_Cranes_write_test/", convert_to="delta",
                    **{"mode": "overwrite", "repartition": "system_id"})
    """

    flat_files = ["csv", "json", "parquet"]
    
    ## Additional parameters
    if convert_to in flat_files:
        dbfs = "/mnt/tunnelab/outputs/"
    elif convert_to == "delta":
        dbfs = "/mnt/tunnelab/delta/"
    else:
        dbfs = ""
    props = {"dbfs_path": dbfs_path or dbfs}
    props.update(kwargs)
    
    if isinstance(self, pd.core.frame.DataFrame):
        if convert_to in flat_files:
            pd.core.frame.DataFrame.convert_to = eval(f"pd.core.frame.DataFrame.to_{convert_to}")
            self.convert_to(f"/dbfs{props['dbfs_path']}{path}")
        else:
            pass #convertir en spark dataframe et transformer le elif suivant en if
    elif isinstance(self, pyspark.sql.dataframe.DataFrame):
        convert_to = "com.microsoft.azure.cosmosdb.spark" if convert_to == "cosmosdb" else convert_to
        coalesce = ".repartition(1)" if convert_to in flat_files+["cosmosdb"] else ""
        repartition = f".partitionBy(\"{props['repartition']}\")" if "repartition" in props.keys() else ""
        mode = f".mode(\"{props['mode']}\")" if "mode" in props.keys() else ""
        options = f".options(**{props['options']})" if "options" in props.keys() else ""
        description = f" COMMENT \'{props['description']}\'" if "description" in props.keys() else ""
        if convert_to in flat_files:
            save_path = f"\"dbfs:{props['dbfs_path']}{path}\""
        elif convert_to == "delta":
            save_path = f"\"{props['dbfs_path']}{path}\""
        else:
            save_path = ""
        
        eval(f"self{coalesce}.write.format(\"{convert_to}\"){mode}{repartition}{options}.save({save_path})")
        
        ## Deletion of the folder structure in the case of json or csv
        if convert_to in ["csv", "json"]:
            files_to_remove = [file.name for file in dbutils.fs.ls(f"dbfs:{props['dbfs_path']}{path}")]
            file_to_copy = [file for file in files_to_remove if re.match(f"part-00000-.*\.{convert_to}", file)][0]
            dbutils.fs.cp(f"dbfs:{props['dbfs_path']}{path}/{file_to_copy}", f"dbfs:{props['dbfs_path']}{file_to_copy}")
            for file_to_remove in files_to_remove:
                dbutils.fs.rm(f"dbfs:{dbfs_path}{path}/{file_to_remove}")
            dbutils.fs.rm(f"dbfs:{props['dbfs_path']}{path}")
            dbutils.fs.mv(f"dbfs:{props['dbfs_path']}{file_to_copy}", f"dbfs:{props['dbfs_path']}{path}")
        ## Creation of the Delta table in case it doesn't exist
        if convert_to == "delta":
            table_name = [elem for elem in path.split("/") if elem][-1]
            spark.sql(f"CREATE TABLE IF NOT EXISTS delta.{table_name} USING DELTA LOCATION {save_path} {description}")
            if "properties" in props.keys():
                spark.sql(f"""ALTER TABLE delta.{table_name} SET TBLPROPERTIES ('{props["properties"][0]}' = '{props["properties"][1]}')""")

pd.core.frame.DataFrame.save = save_dataframe
# DataFrame.save = save_dataframe
