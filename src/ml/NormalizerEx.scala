package ml

import org.apache.spark.ml.feature.Normalizer
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.ParamPair
import org.apache.spark.sql.SparkSession

object NormalizerEx {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder.master("local[2]")
      .appName("NormalizerExample")
      .getOrCreate()

    // $example on$
    val dataFrame = spark.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 0.5, -1.0)),
      (1, Vectors.dense(2.0, 1.0, 1.0)),
      (2, Vectors.dense(4.0, 10.0, 2.0))
    )).toDF("id", "features")

    // Normalize each Vector using $L^1$ norm.
    val normalizer = new Normalizer()
      .setInputCol("features")
      .setOutputCol("normFeatures")
      .setP(1.0)

    val l1NormData = normalizer.transform(dataFrame)
    println("Normalized using L^1 norm")
    l1NormData.show()

    val map: ParamPair[Double] = normalizer.p -> 2
    val l2NormData = normalizer.transform(dataFrame, map)
    println("Normalized using L^inf norm")
    l2NormData.show()

    // Normalize each Vector using $L^\infty$ norm.
    val tmp: ParamPair[Double] = normalizer.p -> Double.PositiveInfinity
    val lInfNormData = normalizer.transform(dataFrame, tmp)
    println("Normalized using L^inf norm")
    lInfNormData.show()
    // $example off$

    spark.stop()
  }
}
