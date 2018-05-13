package mllib

import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

object BinaryClassificationMetricsEx {
  def main(args: Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf().setMaster("local[2]").setAppName("BinaryClassification"))

    val data = MLUtils.loadLibSVMFile(sc, "data/sample_binary_classification_data.txt")

    val Array(training, test) = data.randomSplit(Array(0.7, 0.3))
    training.cache()

    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(2)
      .run(training)
    // Clear the prediction threshold so the model will return probabilities
    model.clearThreshold()

    val predictionAndLabels = test.map{case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)

    val precision = metrics.precisionByThreshold()
    precision.foreach{case(t, p) =>
      println(s"Threadhold: $t, Presition : $p")
    }

    val recall = metrics.recallByThreshold()
      recall.foreach{case (t, r) =>
      println(s"Threadhold: $t, Recall: $r")
    }
    //precision-recall curve
    val PRC = metrics.pr()

    val f1score = metrics.fMeasureByThreshold()
    f1score.foreach{case (t, f) =>
      println(s"Theshold: $t, F-score: $f, Beta = 1")
    }

    val beta = 0.5
    val fScore = metrics.fMeasureByThreshold(beta)
    fScore.foreach { case (t, f) =>
      println(s"Threshold: $t, F-score: $f, Beta = 0.5")
    }

    val auPRC = metrics.areaUnderROC()
    println("Area Under precision-recall curve" + auPRC)

    //compute threshold used in ROC, and PR curve
    val thresholds = precision.map(_._1)

    val roc = metrics.roc()

    val auROC = metrics.areaUnderROC
    println("Area under ROC = " + auROC)


    sc.stop()
  }
}
