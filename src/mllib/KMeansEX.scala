package mllib

import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}

object KMeansEX {
  def main(args: Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf().setMaster("local[2]").setAppName("KMeansEX"))
    val data = sc.textFile("data/kmeans_data.txt")
    val parseData = data.map(s => {
     val tmp = s.split(' ').map(_.toDouble)
      Vectors.dense(tmp)
    })
    val numberCluster = 2
    val numIterations = 2
    val clusters = KMeans.train(parseData, numberCluster, numIterations)
    // Evaluate clustering by computing Within Set Sum of Squared Errors
    val WSSSE = clusters.computeCost(parseData)
    println("Within Set Sum of Squared Errors = " + WSSSE)

    sc.stop()
  }
}
