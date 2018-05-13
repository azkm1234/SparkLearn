package ml

import org.apache.spark.ml.feature.ElementwiseProduct
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

object ElementwiseProducEx {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("ElementwiseProductEx")
      .master("local[2]")
      .getOrCreate()
    val dataFrame = spark.createDataFrame(Seq(
      ("a", Vectors.dense(1.0, 2.0, 3.0)),
      ("b", Vectors.dense(4.0, 5.0, 6.0)))).toDF("id", "vector")

    val transformingVector = Vectors.dense(0.0, 1.0, 2.0)
    val elementWiser = new ElementwiseProduct()
      .setInputCol("vector")
      .setOutputCol("eVector")
      .setScalingVec(transformingVector)

    val elementWiseData = elementWiser.transform(dataFrame)
    elementWiseData.show(false)
    spark.stop()
  }
}
