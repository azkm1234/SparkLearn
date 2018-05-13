package mllib

import org.apache.spark.mllib.feature.ElementwiseProduct
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}

object ElementwiseProductEx {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("StandardScalerExample").setMaster("local")
    val sc = new SparkContext(conf)

    val data = sc.parallelize(Array(Vectors.dense(1.0, 2.0, 3.0), Vectors.dense(4.0, 5.0, 6.0)))
    val transformingVector = Vectors.dense(0.0, 1.0, 2.0)
    val transformer = new ElementwiseProduct(transformingVector)
    val transformedData = transformer.transform(data)
    val transformedData2 = data.map(x => transformer.transform(x))

    println("transformedData1 :")
    transformedData.foreach(println)
    println("transformedData2 : \n")
    transformedData2.foreach(println)

    sc.stop()

  }
}
