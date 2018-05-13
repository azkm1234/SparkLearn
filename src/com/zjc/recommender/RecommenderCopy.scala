package com.zjc.recommender


import java.io.PrintWriter

import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SparkSession

object RecommenderCopy {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("RecommenderCopy")
      .master("local[2]")
      .getOrCreate()

    val data = spark.sparkContext.textFile("data/recommend/000001_0").map{_.split("\t")}

    //得到所有的features，也就是后面前面的字符串
    val features = data.flatMap(_.drop(1)(0).split(";").map{_.split(":")(0)}).distinct()

    val dict: collection.Map[String, Long] = features.zipWithIndex().collectAsMap()

    val trainData = data.map{x =>
      val label = x(0) match {
        case "-1" => 0.0111
        case "1" => 1.0
      }
      val index: Array[Int] = x.drop(1)(0).split(";").map(_.split(":")(0)).map{ fe =>
        val index = dict.get(fe) match {
          case Some(n) => n
          case None => 0.0
        }
        index.toInt
      }
      val vector = new SparseVector(dict.size,index,Array.fill(index.length)(1))
      LabeledPoint(label.toDouble, vector)
    }

    val model = LogisticRegressionWithSGD.train(trainData, 10, 0.1)
    val weights = model.weights.toArray

    val map = dict.map(x => (x._2, x._1)).toMap

    val pw = new PrintWriter("data/recommend/test")
    for (i <- 0 until weights.length) {
      val featuresName = map.get(i) match {
        case Some(x) => x
        case None =>""
      }
      val result = featuresName + "\t" + weights(i)
      pw.write(result)
      pw.println()
    }
    pw.flush()
    pw.close()
    spark.stop()
  }
}
