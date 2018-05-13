package ml

import org.apache.commons.math3.ml.clustering.evaluation.ClusterEvaluator
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object KMeansEx {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName(s"${this.getClass.getSimpleName}").master("local[2]")
      .getOrCreate()

    // $example on$
    // Loads data.
    val dataset = spark.read.format("libsvm").load("data/sample_kmeans_data.txt").toDF("label", "f_eatures")
    dataset.show(false)
    // Trains a k-means model.
    val kmeans = new KMeans().setK(2).setSeed(1L)
    val param = ParamMap(kmeans.featuresCol -> "f_eatures")
    val model = kmeans.fit(dataset,param)
    val summary = model.summary
    summary.predictions.show(false)
    summary.cluster.show(false)
    // $example off$

    spark.stop()
  }
}
