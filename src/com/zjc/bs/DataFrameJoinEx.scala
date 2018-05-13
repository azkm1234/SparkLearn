package com.zjc.bs

import org.apache.spark.sql.SparkSession

object DataFrameJoinEx {
  case class Match(matchId:Int, player1:String, player2:String)
  case class Player(name:String, birthYear:Int)
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local[2]")
      .appName("DataFrameJoinEx")
      .getOrCreate()
    test3(spark)
    spark.stop()
  }
  private def test3 (spark:SparkSession) = {
    val df = spark.createDataFrame(Seq(
      (1, "asd"),
      (1, "ass"),
      (1, "bomb"),
      (1, "pussy"),
      (2, "asd"),
      (3, "ass"),
      (2, "bomb"),
      (5, "pussy")
    )).toDF("num", "name")
    println("*******")
    df.show(false)
    val countDf = df.groupBy("num").count()
    println("*******")
    countDf.show(false)
    countDf.createTempView("count")
    df.createTempView("data")
    println("*******")
    spark.sql("select data.*, count.count from data left join count " +
      "on data.num = count.num").show(false)
    spark.stop()
  }
  private def test2(spark:SparkSession) = {
    val matches = Seq(
      Match(1, "John Wayne", "John Doe"),
      Match(2, "Ive Fish", "San Simon")
    )
    val players = Seq(
      Player("John Wayne", 1986),
      Player("Ive Fish", 1990),
      Player("San Simon", 1974),
      Player("John Doe", 1995)
    )

    val matchesDf = spark.createDataFrame(matches)
    val playersDf = spark.createDataFrame(players)
    matchesDf.createTempView("matches")
    playersDf.createTempView("players")

  }
  private def test1(spark:SparkSession) = {
    import spark.sqlContext.implicits._
    val llist = Seq(("bob", "2015-01-13", 4), ("alice", "2015-04-23",10), ("maria", "2016-06-07", 20))
    val left = llist.toDF("name", "data", "duration")
    val right = Seq(("alice", 100),("bob", 23)).toDF("name","upload")
//    val df = left.join(right,left.col("name") === right.col("name"))
    left.show(false)
    right.show(false)
//    df.show(false)
    left.createTempView("left")
    right.createTempView("right")
    spark.sql("select l.*, r.* from left l left join right r " +
      "on l.name = r.name").show(false)
  }

}
