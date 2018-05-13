package ml

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.SparkSession

object StringIndexerEx {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder.master("local[2]")
      .appName("StringIndexerExample")
      .getOrCreate()

    // $example on$
    val df = spark.createDataFrame(
      Seq((0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c"))
    ).toDF("id", "category")
    val tmp = spark.createDataFrame(Seq(
      (0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c"),(6, "f"),(7, "x")
    )).toDF("id", "category")
    val indexer = new StringIndexer()
      .setInputCol("category")
      .setOutputCol("categoryIndex")

    val indexed = indexer.fit(df).transform(tmp)
    indexed.show()
    // $example off$

    spark.stop()
  }
}
