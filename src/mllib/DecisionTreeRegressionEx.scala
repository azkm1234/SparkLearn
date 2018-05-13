package mllib

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object DecisionTreeRegressionEx {
  def main(args: Array[String]): Unit = {
    val spark = new SparkContext(new SparkConf().setMaster("local[2]").setAppName("DecisionTreeRegressionEx"))
    val data = MLUtils.loadLibSVMFile(spark, "data/sample_libsvm_data.txt")
    val splits = data.randomSplit(Array(0.7, 0.3))
    val training: RDD[LabeledPoint] = splits(0).cache()
    val test = splits(1)

    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "variance"
    val maxDepth = 5
    val maxBins = 32
    val model = DecisionTree.trainRegressor(training,categoricalFeaturesInfo,impurity,maxDepth,maxBins)

    val labelAndPreds = test.map{case point:LabeledPoint =>
      val preds = model.predict(point.features)
      (point.label, preds)
    }
    val testMSE = labelAndPreds.map{case (k, v) =>
      math.pow((k - v).toDouble, 2.0)
    }.mean()
    println(s"Test Mean Squared Error is : $testMSE")
    spark.stop()
  }
}
