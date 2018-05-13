package sql

import org.apache.spark.sql.types._
import org.apache.spark.sql.{Dataset, Row, SparkSession, types}
import org.apache.spark.sql.functions._

object SQLSparkEx {

  case class Person(name: String, age: Long)
  case class Location(lat: Double, long: Double)
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("Spark SQL basic example")
      .config("spark.some.config.option", "some-value")
      .master("local[2]")
      .getOrCreate()
//    runBasicDataFrameExample(spark)
//    runDataSetCreationEx(spark)
//    runInferSchemaEx(spark)
//    runProgrammaticSchemaEx(spark)
    runCreateNewCol(spark)
//    runCreateNewCol2(spark)
//    不管是createDataFrame 还是createDataSet，Seq中包含的都是元组，不是数组
    spark.stop()
  }

  private def runCreateNewCol2(spark:SparkSession): Unit ={

    val rdd = spark.sparkContext.parallelize(
      Row(52.23, 21.01, "Warsaw") :: Row(42.30, 9.15, "Corte") :: Nil)
    val schema = StructType(
        StructField("lat", DoubleType, false) ::
        StructField("long", DoubleType, false) ::
        StructField("key", StringType, false) :: Nil)
    val df = spark.createDataFrame(rdd, schema)
    df.show(false)

    val makeLocation = udf((lat: Double, long: Double) => Location(lat, long))

    val dfRes = df.
      withColumn("location", makeLocation(col("lat"), col("long")))
    dfRes.show(false)
    dfRes.printSchema()

//    dfRes.printSchema
  }
  private def runCreateNewCol(spark: SparkSession): Unit = {
    val sqlContext = spark.sqlContext
    import sqlContext.implicits._
    val sentences = spark.createDataset(Seq(
      "This is an example",
      "And this is another example",
      "One_Word"
    )).toDF("sentence")
    sentences.show(false)
    val extractFristWord = udf((sentences: String) => sentences.split(" ").head)
    val extractLastWord = udf((sentences: String) => sentences.split(" ").last)
    sentences
      .withColumn("first_word", extractFristWord(col("sentence")))
      .withColumn("last_word", extractLastWord(col("sentence")))
      .show(false)
  }

  private def runBasicDataFrameExample(spark: SparkSession): Unit = {
    import spark.implicits._
    val df = spark.read.format("json").load("data/people.json")
    df.show(false)
    df.printSchema()
    df.select("name").show(false)
    df.select($"name", $"age" + 1).show(false)
    df.filter($"age" > 21).show(false)
    df.groupBy($"age").count().show(false)

    df.createOrReplaceTempView("people")
    val sqlDF = spark.sql("select * from people")
    sqlDF.show(false)

    df.createGlobalTempView("people")
    spark.sql("select * from global_temp.people").show(false)

  }

  private def runDataSetCreationEx(spark: SparkSession): Unit = {
    import spark.implicits._
    val caseClassDS = Seq(Person("Andy", 32)).toDF()
    caseClassDS.show(false)

    val primitiveDS = Seq(1, 2, 3)
    primitiveDS.map(_ + 1).toDF().show(false)

    val personDS: Dataset[Person] = spark.read.json("data/people.json").as[Person]
    personDS.show(false)
  }

  private def runInferSchemaEx(spark: SparkSession): Unit = {
    import spark.implicits._
    val peopleDF = spark.sparkContext
      .textFile("data/people.txt")
      .map(_.split(","))
      .map(attributes => Person(attributes(0), attributes(1).trim().toLong))
      .toDF()
    peopleDF.show(false)
    peopleDF.createOrReplaceTempView("people")
    val teenagersDF = spark.sql("select name, age from people where age between 13 and 19")
    teenagersDF.show(false)
    teenagersDF.map(teenager => "Name:" + teenager(0) + "\t" + teenager(1)).show(false)
    teenagersDF.map(teenager => "Name:" + teenager.getAs[String]("name")).show(false)

    implicit val mapEncoder = org.apache.spark.sql.Encoders.kryo[Map[String, Any]]
    teenagersDF.map(teenager => teenager.getValuesMap[Any](List("name", "age"))).show(false)
  }

  private def runProgrammaticSchemaEx(spark: SparkSession): Unit = {
    import spark.implicits._
    val peopleRDD = spark.sparkContext.textFile("data/people.txt")
    val schemaString = "name age"
    val fields = schemaString.split(" ")
      .map(fieldName => StructField(fieldName, DataTypes.StringType, nullable = true))
    val rowRDD = peopleRDD.map(_.split(","))
      .map { attributes =>
        Row(attributes(0), attributes(1).trim())
      }
    val peopleDF = spark.createDataFrame(rowRDD, types.StructType(fields))
    peopleDF.show(false)
    peopleDF.createTempView("people")
    val results = spark.sql("select name from people")
    results.printSchema()
    results.map { attributes =>
      "Name:" + attributes(0)
    }.printSchema()
  }
}
