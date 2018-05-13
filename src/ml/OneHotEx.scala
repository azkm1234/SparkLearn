package ml

import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.sql.SparkSession

object OneHotEx {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("OneHotEncoderExample")
      .master("local[2]")
      .getOrCreate()

    // $example on$
    val df = spark.createDataFrame(Seq(
      (0, "a", "d"),
      (1, "b", "e"),
      (2, "c", "f"),
      (3, "a", "g"),
      (4, "a", "d"),
      (5, "c", "d")
    )).toDF("id", "c1", "c2")

    val indexer = new StringIndexer()
      .setInputCol("c1")
      .setOutputCol("c1Index")
      .fit(df)
    val indexed = indexer.transform(df)

    val encoder = new OneHotEncoder()
      .setInputCol("c1Index")
      .setOutputCol("c1Vec")

    val encoded = encoder.transform(indexed)
    encoded.show()
    // $example off$

    spark.stop()
  }
}
