package mllib

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{Row, SparkSession}

object eortex {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .master("local[2]")
      .appName("eortex")
      .getOrCreate()
    val training = spark.createDataFrame(Seq(
      (1.0, Vectors.dense(0.0, 1.1, 0.1)),
      (0.0, Vectors.dense(2.0, 1.0, -1.0)),
      (0.0, Vectors.dense(2.0, 1.3, 1.0)),
      (1.0, Vectors.dense(0.0, 1.2, -0.5))
    )).toDF("label", "features")
    val lr = new LogisticRegression()
    println("LogisticRegression parameters \n " + lr.explainParams() + "\n")
    lr.setMaxIter(10)
      .setRegParam(0.01)
    val modle1 = lr.fit(training)
    println("Model 1 was fit using paramters :" + modle1.extractParamMap())

    val paramMap = ParamMap(lr.maxIter -> 30)
      .put(lr.maxIter, 20)
      .put(lr.regParam -> 0.01, lr.threshold -> 0.55)

    val paramMap2 = ParamMap(lr.probabilityCol -> "myProbability")
    val paramMapCombine = paramMap ++ paramMap2

    val model2 = lr.fit(training, paramMapCombine)

    println("Model 2 was fit using parameters: " + model2.parent.extractParamMap)

    val test = spark.createDataFrame(Seq(
      (1.0, Vectors.dense(-1.0, 1.5, 1.3)),
      (0.0, Vectors.dense(3.0, 2.0, -0.1)),
      (1.0, Vectors.dense(0.0, 2.2, -1.5))
    )).toDF("label", "features")

    model2.transform(test)
      .select("features", "label", "myProbability", "prediction")
      .collect()
      .foreach{
        case Row(features: Vector, label: Double, prob: Vector, prediction: Double) =>
          println(s"($features, $label) -> prob=$prob, prediction=$prediction")
      }
    spark.stop()
  }
}
