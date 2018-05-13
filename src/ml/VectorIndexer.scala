package ml

import org.apache.spark.sql.SparkSession

object VectorIndexerEx {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local[2]")
      .appName("RandomForestClassifierEx")
      .getOrCreate()

  }
}
