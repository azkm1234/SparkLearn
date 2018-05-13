package mllib

import java.lang

import org.apache.spark.sql.{Dataset, SparkSession}

object SparkSessionEX {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .master("local[2]")
      .appName("SparkSessionEx")
      .enableHiveSupport()
      .getOrCreate()
    //set new runtime options
    spark.conf.set("spark.sql.shuffle.partition", 6)
    spark.conf.set("spark.executor.memory", "2g")
    //get all setting
    spark.catalog.listDatabases().show()
    spark.catalog.listTables.show(false)
    spark.conf.getAll.foreach(s1 => println(s1))
    val numDs: Dataset[lang.Long] = spark.range(5, 100, 5)

    spark.stop()
  }
}
