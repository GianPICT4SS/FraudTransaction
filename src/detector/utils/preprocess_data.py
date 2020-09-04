"""utils scripts for preprocessing data by using method of the class FeatureTools
N.B pyspark works with Java 8, so it's necessary to have Java 8 installed and using this version."""

import json
import logging
import warnings

from utils.feature_tools import FeatureTools
from utils.config import COL

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s : %(message)s',
                    datefmt='%d/%m/%Y %H:%M ',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import pyspark.sql.types as tp
from pyspark.sql import Row
from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler, MinMaxScaler

from collections import OrderedDict



warnings.filterwarnings("ignore")
###################################################
# Spark Configuration
###################################################
conf = SparkConf().setAppName('FT')
sc = SparkContext(conf=conf)

spark = SparkSession\
    .builder\
    .appName('FraudTransaction')\
    .master('local')\
    .getOrCreate()


# define the schema (maybe useful in future)
my_schema = tp.StructType([
    tp.StructField(name='Time', dataType= tp.DoubleType(),  nullable= True),
    tp.StructField(name='V1', dataType= tp.DoubleType(),  nullable= True),
    tp.StructField(name='V2', dataType= tp.DoubleType(),   nullable= True),
    tp.StructField(name='V3', dataType= tp.DoubleType(),   nullable= True),
    tp.StructField(name='V4', dataType= tp.DoubleType(),   nullable= True),
    tp.StructField(name='V5', dataType= tp.DoubleType(),   nullable= True),
    tp.StructField(name='V6', dataType= tp.DoubleType(),   nullable= True),
    tp.StructField(name='V7', dataType= tp.DoubleType(),   nullable= True),
    tp.StructField(name='V8', dataType= tp.DoubleType(),   nullable= True),
    tp.StructField(name='V9', dataType= tp.DoubleType(),   nullable= True),
    tp.StructField(name='V10', dataType= tp.DoubleType(),   nullable= True),
    tp.StructField(name='V11', dataType= tp.DoubleType(),   nullable= True),
tp.StructField(name='V12', dataType= tp.DoubleType(),   nullable= True),
tp.StructField(name='V13', dataType= tp.DoubleType(),   nullable= True),
tp.StructField(name='V14', dataType= tp.DoubleType(),   nullable= True),
tp.StructField(name='V15', dataType= tp.DoubleType(),   nullable= True),
tp.StructField(name='V16', dataType= tp.DoubleType(),   nullable= True),
tp.StructField(name='V17', dataType= tp.DoubleType(),   nullable= True),
tp.StructField(name='V18', dataType= tp.DoubleType(),   nullable= True),
tp.StructField(name='V19', dataType= tp.DoubleType(),   nullable= True),
tp.StructField(name='V20', dataType= tp.DoubleType(),   nullable= True),
tp.StructField(name='V21', dataType= tp.DoubleType(),   nullable= True),
tp.StructField(name='V22', dataType= tp.DoubleType(),   nullable= True),
tp.StructField(name='V23', dataType= tp.DoubleType(),   nullable= True),
tp.StructField(name='V24', dataType= tp.DoubleType(),   nullable= True),
tp.StructField(name='V25', dataType= tp.DoubleType(),   nullable= True),
tp.StructField(name='V26', dataType= tp.DoubleType(),   nullable= True),
tp.StructField(name='V27', dataType= tp.DoubleType(),   nullable= True),
tp.StructField(name='V28', dataType= tp.DoubleType(),   nullable= True),
tp.StructField(name='Amount', dataType= tp.DoubleType(),   nullable= True),
tp.StructField(name='Class', dataType= tp.IntegerType(),   nullable= True)

    ])



def load_new_training_data(path):


    data = []
    try:
        with open(path, "r") as f:
            for line in f:
                data.append(json.loads(line))
            logger.info(f'TYPE: {type(data[1])}')
            #rdd = sc.parallelize(data).map(lambda x: Row(**OrderedDict(sorted(x.items()))))
            rdd = sc.parallelize(data).map(lambda x: Row(**x))
            #rdd = sc.parallelize(data)
            df = spark.createDataFrame(rdd, my_schema)
            return df
    except Exception as e:
        logger.exception(str(e))
        pass


def build_train(train_path, new_train_path=None):


    target ='Class'

    #read initial DataFrame
    #df = spark.read.format("csv")\
    #    .options(header='true', inferschema='true')\
    #    .load(train_path)
    
    df = spark.read.csv(train_path, header=True, inferSchema=True)


    # new train data available?
    if new_train_path:
        df_tmp = load_new_training_data(new_train_path)
        #in order to be consistent with df
        #  df_tmp = df_tmp.select(df.columns)  NO MORE NECESSARY (We use my_schema for datatype consistent)
        # concatenate for a new DataFrame
        df = df.union(df_tmp)



    preprocessor = FeatureTools()
    logger.info('Preprocessing Data')

    dataprocessor = preprocessor.fit(
            df,
            target,
            df.columns,
            MinMaxScaler(inputCol="features", outputCol='scFeatures')
            )




    """Since we use pyspark as preprocessing it is not possible to save only python object that contains spark rdd
    as attribute."""

    return dataprocessor





