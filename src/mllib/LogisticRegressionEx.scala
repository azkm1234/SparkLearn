package mllib

import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

object LogisticRegressionEx {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("LogisticRegressionWithLBFGSExample").setMaster("local[2]")
    val sc = new SparkContext(conf)

    val data = MLUtils.loadLibSVMFile(sc,"data/sample_libsvm_data.txt")
    val splits = data.randomSplit(Array(0.6, 0.4), 11L)
    val training = splits(0).cache()
    val test = splits(1)
    val model = new LogisticRegressionWithLBFGS().setNumClasses(10).run(training)
    test.collect().foreach(println)
    println()
    training.collect().foreach(println)
    val predictionAndLabels = test.map{
      case LabeledPoint(label, features) =>
        val prediction = model.predict(features)
        (prediction, label)
    }
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)
    println(metrics.areaUnderROC())
    sc.stop()
  }
}
