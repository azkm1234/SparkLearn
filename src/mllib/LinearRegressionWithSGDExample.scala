package mllib

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object LinearRegressionWithSGDExample {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("LinearRegressionWithSGDExample").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val data = sc.textFile("data/lpsa.data")
    val parseData: RDD[LabeledPoint] = data.map{ line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }.cache()

    //Building the model
    val numIterations = 100
    val stepSize = 0.00000001
    val model = LinearRegressionWithSGD.train(parseData, numIterations,stepSize)

    // Evaluate model on training example and compute training error
    val valuesAndPreds: RDD[(Double, Double)] = parseData.map{ lp =>
      val prediction = model.predict(lp.features)
      (lp.label, prediction)
    }

    val MSE = valuesAndPreds.map{ case (v, p) => math.pow((v - p), 2)}.mean()
    println("training Mean Squared Error is " + MSE)
    sc.stop()
  }
}
