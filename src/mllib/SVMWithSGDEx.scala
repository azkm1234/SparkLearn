package mllib

import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object SVMWithSGDEx {
  def main(args: Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf().setMaster("local[2]").setAppName("svmwithsgdex"))
    val data =  MLUtils.loadLibSVMFile(sc, "data/sample_libsvm_data.txt")
    data.take(1).foreach(println(_))
    val splits: Array[RDD[LabeledPoint]] = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test: RDD[LabeledPoint] = splits(1)
    training.collect().foreach(println)

    val numIterations = 100
    val model = SVMWithSGD.train(training, numIterations)
    model.clearThreshold()
    val scoreAndLabel: RDD[(Double, Double)] = test.map{ point =>
      val score = model.predict(point.features)
      (score, point.label)
    }
    scoreAndLabel.take(10).foreach(println(_))

    val metrics = new BinaryClassificationMetrics(scoreAndLabel)
    val auROC: Double = metrics.areaUnderROC()
    println("Area under ROC :" + auROC)
    sc.stop()
  }
}
