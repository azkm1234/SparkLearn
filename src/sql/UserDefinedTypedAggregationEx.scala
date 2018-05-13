package sql


import org.apache.spark.sql.expressions.Aggregator
import org.apache.spark.sql.{Encoder, Encoders, SparkSession, TypedColumn}

object UserDefinedTypedAggregationEx {
  case class Employee(name:String, salary:Long)
  case class Average(var sum:Long, var count:Long)
  object MyAverage extends Aggregator[Employee, Average, Double] {
    override def zero: Average = Average(0L,0L)

    override def reduce(b: Average, a: Employee): Average = {
      b.sum += a.salary
      b.count += 1
      b
    }

    override def merge(b1: Average, b2: Average): Average = {
      b1.sum += b2.sum
      b1.count += b2.count
      b1
    }

    override def finish(reduction: Average): Double = {reduction.sum.toDouble / reduction.count.toDouble}

    override def bufferEncoder: Encoder[Average] = Encoders.product

    override def outputEncoder: Encoder[Double] = Encoders.scalaDouble
  }
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local[2]")
      .appName("Spark SQL user-defined Datasets aggregation example")
      .getOrCreate()
    import spark.implicits._
    val ds = spark.read.json("data/employees.json").as[Employee]
    ds.show(false)
    val averageSalary: TypedColumn[Employee, Double] = MyAverage.toColumn.name("average_salary")
    val result = ds.select(averageSalary)
    result.show(false)
    spark.stop()
  }
}
