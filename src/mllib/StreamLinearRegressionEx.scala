package mllib

import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, StreamingLinearRegressionWithSGD}
import org.apache.spark.streaming.{Seconds, StreamingContext}

object StreamLinearRegressionEx {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local[2]").setAppName("StreamLinearRegressionEx")
    val ssc = new StreamingContext(conf, Seconds(1))
    val path = "data/lpsa.data"

    val trainingData = ssc.textFileStream(path).map(LabeledPoint.parse(_)).cache()
    val testData = ssc.textFileStream(path).map(LabeledPoint.parse)

    val numFeatures = 3
    val model = new StreamingLinearRegressionWithSGD()
      .setInitialWeights(Vectors.zeros(numFeatures))
    model.trainOn(trainingData)
    model.predictOnValues(testData.map{lp => (lp.label, lp.features)}).print()

    ssc.start()
    ssc.awaitTerminationOrTimeout(10*1000)
    ssc.stop()
  }
}
