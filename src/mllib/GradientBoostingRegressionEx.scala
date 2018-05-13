package mllib

import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

object GradientBoostingRegressionEx {
  def main(args: Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf().setMaster("local[2]").setAppName("gradientBootingRegression"))
    val data = MLUtils.loadLibSVMFile(sc, "data/sample_libsvm_data.txt")
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    val boostingStrategy = BoostingStrategy.defaultParams("Regression")
    boostingStrategy.numIterations = 10
    boostingStrategy.treeStrategy.maxDepth = 2
    boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()
    
    println("boostingStragy is : \n" + boostingStrategy)
    val model = GradientBoostedTrees.train(trainingData,boostingStrategy)

    val labelAndPreds = testData.map{point =>
      val preds = model.predict(point.features)
      (point.label, preds)
    }
    val mse = labelAndPreds.map(t =>
      math.pow(t._1 - t._2, 2)
    ).mean()
    println("gradientBoostedTrees's mean square error : " + mse)
    model.trees.foreach(t => println(t.toDebugString))
    sc.stop()
  }
}
