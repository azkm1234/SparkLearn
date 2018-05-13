package mllib

import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

object DecisionTreeClassificationEx {
  def main(args: Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf().setAppName("DecisionTreeClassficationEX").setMaster("local[2]"))
    val data = MLUtils.loadLibSVMFile(sc, "data/sample_libsvm_data.txt")

    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
    val numclass = 2
    val categoricalFeatureInfo = Map[Int, Int]()
    val impurity = "gini"
    val maxDepth = 5
    val maxBin = 32

    val model = DecisionTree.trainClassifier(trainingData, numclass,categoricalFeatureInfo,impurity,maxDepth,maxBin)

    val labelAndPreds = testData.map{point =>
      val preds = model.predict(point.features)
      (point.label, preds)
    }
    val testErr = labelAndPreds.filter(touple => touple._2 != touple._1).count().toDouble / labelAndPreds.count()
    println("testErr is :" + testErr)
    println("Learned Classfication tree model \n" + model.toDebugString)

    sc.stop()
  }
}
