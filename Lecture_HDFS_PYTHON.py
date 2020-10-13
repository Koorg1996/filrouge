#####Script python pour se connecter à HDFS avec hadoop installé en local

#### 1ere méthode 

import os
from pyspark.sql import SparkSession
import pandas as pd
os.environ["HADOOP_USER_NAME"] = "cloudera"
os.environ["PYTHON_VERSION"] = "3.6.9"
sparkSession = SparkSession.builder.appName("pyspark_test").getOrCreate()
df = sparkSession.read.csv('hdfs://quickstart.cloudera:8020/user/cloudera/Alassane/movies_metadata.csv')
df.show()

### 2e méthode mais qui marche avec python2

#import pydoop.hdfs as hd
#with hd.open("hdfs://quickstart.cloudera:8020/user/cloudera/Alassane/movies_metadata.csv") as f:
       #print(f.read())