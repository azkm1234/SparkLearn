package mllib

import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

object RandomForestClassificationEx {
  def main(args: Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf().setMaster("local[2]").setAppName("randomForestClassficationEx"))
    val data = MLUtils.loadLibSVMFile(sc, "data/sample_libsvm_data.txt")
    val Array(traindata, testdata) = data.randomSplit(Array(0.7, 0.3))
    val numclass = 2
    val categoricalFeatureInfo = Map[Int, Int]()
    val numTrees = 3
    val impurity = "gini"
    val maxDepth = 4
    val maxBins = 32
    val model = RandomForest.trainClassifier(traindata,numclass,categoricalFeatureInfo,numTrees,"auto",impurity,maxDepth,maxBins)

    val labelAndPred = testdata.map{point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testErr = labelAndPred.filter(t => t._1 != t._2).count().toDouble / labelAndPred.count()
    println(s"testErr is $testErr")
    println("Learned classification forest model:\n" + model.toDebugString)

    sc.stop()
  }
}
