package mllib

import org.apache.spark.sql.SparkSession
object sparkSQLearn {
  case class Person(name:String, age:Int)
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .master("local[2]")
      .appName("spark example")
      .getOrCreate()
    val df = spark.read.json("resources/people.json")
    df.show()
    df.printSchema()
    df.select("name").show()
    import spark.implicits._
    df.select($"name", $"age" + 1).show()
    df.filter($"age" > 1).show()
    df.createOrReplaceTempView("people")
    val sqlDf = spark.sql("select * from people where age > 10")
    sqlDf.show()
    df.createGlobalTempView("people")
    spark.sql("select * from global_temp.people").show()
    spark.newSession()
  }
}

