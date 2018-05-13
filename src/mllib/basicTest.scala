package mllib


import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd.RDD
object basicTest {

  def main(args: Array[String]): Unit = {
    test1()
  }
  def test1(): Unit = {
    val sc = new SparkContext(new SparkConf().setMaster("local[2]").setAppName("BinaryClassification"))
    val data: RDD[Array[Int]] = sc.parallelize(Seq(
      Array(1, 2, 4, 5),
      Array(2, 4, 6, 8)
    ))
    val array = data.collect()
    array.map(x => 10).foreach(println)
    sc.stop()
  }
}
