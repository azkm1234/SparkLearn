package mllib

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.stat.ChiSquareTest
import org.apache.spark.sql.{Row, SparkSession}

object sparkTest3 {
  def main(args: Array[String]): Unit = {
    test4()
  }
  def test1() = {
    import org.apache.spark.sql.SQLContext
    import org.apache.spark.{SparkConf, SparkContext}
    val conf = new SparkConf().setAppName("test").setMaster("local")
    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")
    val sqlContext = new SQLContext(sc)
    case class Person(id:Int, age:Int)
    val idAgePerson = sc.parallelize(Array(Person(1, 12), Person(2, 13), Person(3, 15)))
    idAgePerson.filter(_.age > 13).collect().foreach(println)
  }
  def test2() = {
    import org.apache.spark.sql.types.{DataTypes, StructField, StructType}
    import org.apache.spark.sql.{Row, SQLContext}
    import org.apache.spark.{SparkConf, SparkContext}
    val conf = new SparkConf().setAppName("test").setMaster("local")
    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")
    val sqlContext = new SQLContext(sc)
    val idAgeRDDRow = sc.parallelize(Array(Row(1, 30), Row(2, 29), Row(4, 21)))
    val schema: StructType = StructType(Array(StructField("id", DataTypes.IntegerType), StructField("age", DataTypes.IntegerType)))
    val idAgeDS = sqlContext.createDataFrame(idAgeRDDRow, schema)
    idAgeDS.show()
    import sqlContext.implicits._
    Seq(1, 2, 3).toDS().show()
    sqlContext.createDataset(sc.parallelize(Array(1, 2, 3))).show()
  }
  def test3() : Unit = {
    import org.apache.spark.sql.types.{DataTypes, StructField, StructType}
    import org.apache.spark.sql.{Row, SQLContext}
    import org.apache.spark.{SparkConf, SparkContext}
    val conf = new SparkConf().setAppName("test").setMaster("local[2]") // 调试的时候一定不要用local[*]
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    val idAgeRDDRow = sc.parallelize(Array(Row(1, 30), Row(2, 29), Row(4, 21)))

    val schema = StructType(Array(StructField("id", DataTypes.IntegerType), StructField("age", DataTypes.IntegerType)))

    // 在2.0.0-preview中这行代码创建出的DataFrame, 其实是DataSet[Row]
    val idAgeDS = sqlContext.createDataFrame(idAgeRDDRow, schema)
    idAgeDS.show()

    // 目前支持String, Integer, Long等类型直接创建Dataset
    Seq(1, 2, 3).toDS().show()
    sqlContext.createDataset(sc.parallelize(Array(1, 2, 3))).show()
  }
  def test4() : Unit  = {
    val spark = SparkSession.builder()
      .master("local[2]")
      .appName("test4")
      .getOrCreate()
    import spark.implicits._
    val data = Seq(
      (0.0, Vectors.dense(0.5, 10.0)),
      (0.0, Vectors.dense(1.5, 20.0)),
      (1.0, Vectors.dense(1.5, 30.0)),
      (0.0, Vectors.dense(3.5, 30.0)),
      (0.0, Vectors.dense(3.5, 40.0)),
      (1.0, Vectors.dense(3.5, 40.0)),
      (0.0, Vectors.dense(0.5, 10.0)),
      (0.0, Vectors.dense(1.5, 20.0)),
      (1.0, Vectors.dense(1.5, 30.0)),
      (0.0, Vectors.dense(3.5, 30.0)),
      (0.0, Vectors.dense(3.5, 40.0)),
      (1.0, Vectors.dense(3.5, 40.0))
    )
    val df = data.toDF("label", "features")
    val chi: Row = ChiSquareTest.test(df, "features", "label")
      .head()
    println("pValues = " + chi.getAs[Vector](0))
    println("degreesOfFreedom = " + chi.getSeq[Int](1).mkString("[", ",", "]"))
    println("statistics = " + chi.getAs[Vector](2))
    spark.stop()
  }
}
