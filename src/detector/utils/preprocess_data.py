"""utils scripts for preprocessing data by using method of the class FeatureTools
N.B pyspark works with Java 8, so it's necessary to have Java 8 installed and using this version."""

import json
import pickle
import logging
import pandas as pd
import warnings
from pathlib import Path
from utils.feature_tools import FeatureTools

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s : %(message)s',
                    datefmt='%d/%m/%Y %H:%M ',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

from pyspark import SparkContext
from pyspark.sql.session import SparkSession
import pyspark.sql.types as tp
from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler
from pyspark.sql import Row, Column
from pyspark.sql.functions import udf


warnings.filterwarnings("ignore")
###################################################
# Spark Configuration
###################################################

sc = SparkContext()

spark = SparkSession\
    .builder\
    .appName('FraudTransaction')\
    .master('local')\
    .getOrCreate()


# define the schema (maybe useful in future)
my_schema = tp.StructType([
    tp.StructField(name='Time', dataType= tp.FloatType(),  nullable= True),
    tp.StructField(name='V1', dataType= tp.FloatType(),  nullable= True),
    tp.StructField(name='V2', dataType= tp.FloatType(),   nullable= True)
    ])


def load_new_training_data(path):
    data = []
    try:
        with open(path, "r") as f:
            for line in f:
                data.append(json.loads(line))
            #df = pd.DataFrame(data)  # so that hasattr(df, columns)
            rdd = sc.parallelize(data)
            return spark.createDataFrame(rdd)
    except Exception as e:
        logger.info(str(e))


def build_train(train_path, new_train_path=None):


    target ='Class'
    #read initial DataFrame
    df = spark.read.format("csv")\
        .options(header='true', inferschema='true')\
        .load(train_path)

 # new train data available?
    if new_train_path:
        logger.info(f'fd.columns: {df.columns}')
        logger.info(f'df.take(3): {df.take(3)}')
        df_tmp = load_new_training_data(new_train_path)
        logger.info(f'df_tmp.columns: {df_tmp.columns}')
        logger.info(f'df_tmp.take(3): {df_tmp.take(3)}')

        #in order to be consistent with df
        df_tmp = df_tmp[df.columns]
        # concatenate for a new DataFrame
        df = df.union(df_tmp)
        #ALERT: save on Disk. This operation could be dangerous for Big Amount of Data
        df.write.csv(train_path)

    # UDF for converting columns type from vector to double type
    #unlist = udf(lambda x: round(float(list(x)[0]), 3), tp.DoubleType())
    preprocessor = FeatureTools()
    logger.info('Preprocessing Data')
    #df = df.withColumn(column, unlist(column))
    dataprocessor = preprocessor.fit(
            df,
            target,
            df.columns,
            StandardScaler(inputCol="features", outputCol='features'+'_scd')
            )




    """Since we use pyspark as preprocessing it is not possible to save only python object that contains spark rdd
    as attribute."""

    return dataprocessor





