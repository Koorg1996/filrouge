#from cassandra.cluster import Cluster
#cluster = Cluster()
#session = cluster.connect()


from pyspark import SparkContext

sc = SparkContext()

x = sc.parallelize([1,2,3])

x.collect()