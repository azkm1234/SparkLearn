package mllib

import org.apache.spark.sql.{DataFrame, SparkSession}

object SparkTest2 {
  case class Person(name:String, age: Int)
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession
      .builder()
      .master("local[2]")
      .appName("Spark SQL basic example")
      .config("spark.some.config.option", "some-value")
      .getOrCreate()

    import spark.implicits._
    val squaresDF: DataFrame = spark.sparkContext.makeRDD(1 to 5).map(i => (i, i * i)).toDF("value", "square")
//    squaresDF.write.parquet("data/test_table/key=1")
    val cubesDF: DataFrame = spark.sparkContext.makeRDD(6 to 10).map(i => (i, i * i * i)).toDF("value", "cube")
//    cubesDF.write.parquet("data/test_table/key=2")
    val mergedDF: DataFrame = spark.read.option("mergeSchema", "true").parquet("data/test_table")
    mergedDF.show()
    mergedDF.printSchema()
  }
}
