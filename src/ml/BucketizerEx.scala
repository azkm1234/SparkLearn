package ml

import org.apache.spark.ml.feature.Bucketizer
import org.apache.spark.sql.SparkSession

object BucketizerEx {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .master("local[2]")
      .appName("BucketizerEx")
      .getOrCreate()
    val splits = Array(Double.NegativeInfinity, -.5, -.3, 0, .2, .5, Double.PositiveInfinity)
    val data = Array(-999.9, -0.5, -0.3, 0.0, 0.2, 999.9)

    val dataFrame = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val bucketizer = new Bucketizer()
      .setInputCol("features")
      .setOutputCol("bucketFeatures")
      .setSplits(splits)

    val bucketedData = bucketizer.transform(dataFrame)
    println(s"Bucketizer output with ${bucketizer.getSplits.length-1} buckets")
    bucketedData.show()
    spark.stop()
  }
}
