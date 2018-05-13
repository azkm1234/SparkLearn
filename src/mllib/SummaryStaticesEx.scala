package mllib

import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.{Matrices, Vectors}
import org.apache.spark.mllib.random.RandomRDDs._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
object SummaryStaticesEx {
  def main(args: Array[String]): Unit = {
    test5
  }
  def test5: Unit = {
    val conf = new SparkConf().setAppName("KernelDensityEstimationExample").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val u = normalRDD(sc, 10000, 10)
    u.collect().take(20).foreach(println)

    sc.stop()

  }
  def test4: Unit = {
    val sc = new SparkContext(new SparkConf().setMaster("local[2]").setAppName("test3"))
    val vec = Vectors.dense(0.1, 0.15, 0.2, 0.2, 0.25)
    val goodnessOfFitTestResult = Statistics.chiSqTest(vec)
    println(s"$goodnessOfFitTestResult\n")


    val mat = Matrices.dense(3, 2, Array(1.0, 3.0, 5.0, 2.0, 4.0, 6.0))

    val independceTestResult = Statistics.chiSqTest(mat)
    println(independceTestResult)

    val obs = sc.parallelize(Seq(
      LabeledPoint(1.0, Vectors.dense(1.0, 0.0, 3.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 2.0, 0.0)),
      LabeledPoint(-1.0, Vectors.dense(-1.0, 0.0, -0.5))
    ))

    val featuresTestResult = Statistics.chiSqTest(obs)
    println(independceTestResult)
    featuresTestResult.zipWithIndex.foreach{
      case (k, v) => {
        println("Column " + (v + 1).toString + ":")
        println(k)
      }
    }
    sc.stop()
  }
  def test3: Unit ={
    val sc = new SparkContext(new SparkConf().setMaster("local[2]").setAppName("test3"))
    val data = sc.parallelize(Seq(
      (1, 'a'), (1, 'b'), (2, 'c'), (2, 'd'), (2, 'e'), (3, 'f')
    ))
    val fraction = Map(1 -> 0.1, 2 -> 0.6, 3 -> 0.3)

    val approxSample = data.sampleByKey(withReplacement = false, fractions = fraction)

    val exactSample = data.sampleByKeyExact(withReplacement = false, fractions = fraction)

    println("approxSample size is " + approxSample.collect().size.toString)
    approxSample.collect().foreach(println)

    println("exactSample its size is " + exactSample.collect().size.toString)
    exactSample.collect().foreach(println)
  }
  def test2: Unit = {
    val spark = SparkSession.builder().appName("ssex")
      .master("local[2]")
      .getOrCreate()
    val seriesX: RDD[Double] = spark.sparkContext.parallelize(Array(1, 2, 3, 4, 5))
    val seriesY: RDD[Double] = spark.sparkContext.parallelize(Array(11, 22, 33, 44, 55))

    val correlation = Statistics.corr(seriesX, seriesY, "pearson")
    println(s"Correlation is : $correlation")
    val data: RDD[linalg.Vector] = spark.sparkContext.parallelize(
      Seq(
        Vectors.dense(1.0, 10.0, 100.0),
        Vectors.dense(2.0, 20.0, 200.0),
        Vectors.dense(5.0, 33.0, 366.0))
    )
    val corrMatrix = Statistics.corr(data, "pearson")
    println(s"Correlation Matrix is  \n$corrMatrix")
    spark.stop()
  }
  def test1: Unit = {
    val spark = SparkSession.builder().appName("ssex")
      .master("local[2]")
      .getOrCreate()
    val observation = spark.sparkContext.parallelize(Seq(
      Vectors.dense(1.0, 10.0, 100.0),
      Vectors.dense(2.0, 20.0, 200.0),
      Vectors.dense(3.0, 30.0, 300.0)
    ))

    val summary: MultivariateStatisticalSummary = Statistics.colStats(observation)
    println(summary.mean)
    println(summary.max)
    println(summary.min)
    println(summary.variance)
    println(summary.numNonzeros)
    println(summary)
    spark.stop()
  }
}
