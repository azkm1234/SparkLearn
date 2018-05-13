package ml

import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

object StandardScalerEx {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder.master("local[2]")
      .appName("StandardScalerExample")
      .getOrCreate()

    test2(spark)
    // $example off$

    spark.stop()

  }
  private def test2(spark : SparkSession) = {
    val spark = SparkSession
      .builder.master("local[2]")
      .appName("NormalizerExample")
      .getOrCreate()

    val dataFrame = spark.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 1.0, -1.0)),
      (1, Vectors.dense(0.0, 1.0, -1.0)),
      (2, Vectors.dense(1.0, 1.0, 2.0))
    )).toDF("id", "features")

    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
//      .setWithMean(true)
    val scaledModel1 = scaler.fit(dataFrame)
    val scaledData = scaledModel1.transform(dataFrame)
    scaledData.show(false)
    println("平均值：　" + scaledModel1.mean + "   方差估计： " + scaledModel1.std)

    val scaledModel2 = scaler.setWithMean(true).fit(dataFrame)
    scaledModel2.transform(dataFrame).show(false)
    println("平均值：　" + scaledModel2.mean + "   方差估计： " + scaledModel2.std)

  }
  private def test1(spark : SparkSession) = {
    val dataFrame = spark.read.format("libsvm").load("data/sample_libsvm_data.txt")
    dataFrame.show(false)
    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(true)
      .setWithMean(false)

    // Compute summary statistics by fitting the StandardScaler.
    val scalerModel = scaler.fit(dataFrame)

    // Normalize each feature to have unit standard deviation.
    val scaledData = scalerModel.transform(dataFrame)
    scaledData.show(false)
  }
}
