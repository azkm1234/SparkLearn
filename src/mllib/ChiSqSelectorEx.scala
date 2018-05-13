package mllib


import org.apache.spark.mllib.feature.ChiSqSelector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

object ChiSqSelectorEx {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("StandardScalerExample").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val data = MLUtils.loadLibSVMFile(sc, "data/sample_libsvm_data.txt")

    val discretizedData = data.map{ lp => LabeledPoint(lp.label, Vectors.dense(lp.features.toArray.map(x => (x / 16).floor)))}
    val selector = new ChiSqSelector(50)
    val transformer = selector.fit(discretizedData)
    val filteredData = discretizedData.map{lp =>
      LabeledPoint(lp.label, transformer.transform(lp.features))
    }
    println("original data: ")
    discretizedData.foreach(println)

    println("filtered data: ")
    filteredData.foreach(x => println(x))

  }
}
