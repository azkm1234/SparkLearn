package mllib

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.feature.{StandardScaler, StandardScalerModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils


object StandardScalerEx {
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("StandardScalerExample").setMaster("local")
    val sc = new SparkContext(conf)

    // $example on$
    val data = sc.parallelize(Seq(
      new LabeledPoint(0, Vectors.dense(1, 0, 0, 0, 1)),
      new LabeledPoint(1, Vectors.dense(1, 1, 0, 1, 0)),
      new LabeledPoint(1, Vectors.dense(1, 1, 0, 0, 0)),
      new LabeledPoint(0, Vectors.dense(1, 0, 0, 0, 0)),
      new LabeledPoint(1, Vectors.dense(1, 1, 0, 0, 0))))

    val scaler1 = new StandardScaler().fit(data.map(x => x.features))
    val scaler2 = new StandardScaler(withMean = true, withStd = true).fit(data.map(x => x.features))
    // scaler3 is an identical model to scaler2, and will produce identical transformations
    val scaler3 = new StandardScalerModel(scaler2.std, scaler2.mean)

    // data1 will be unit variance.
    val data1 = data.map(x => (x.label, scaler1.transform(x.features)))

    // data2 will be unit variance and zero mean.
    val data2 = data.map(x => (x.label, scaler2.transform(Vectors.dense(x.features.toArray))))
    // $example off$

    println("data1: ")
    data1.foreach(x => println(x))

    println("data2: ")
    data2.foreach(x => println(x))

    sc.stop()
  }
}
