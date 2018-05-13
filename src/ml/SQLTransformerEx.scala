package ml

import org.apache.spark.ml.feature.SQLTransformer
import org.apache.spark.sql.SparkSession

object SQLTransformerEx {
  def main(args: Array[String]) {
    val spark = SparkSession
      .builder.master("local[2]")
      .appName("SQLTransformerExample")
      .getOrCreate()

    // $example on$
    val df = spark.createDataFrame(
      Seq((0, 1.0, 3.0), (2, 2.0, 5.0))).toDF("id", "v1", "v2")
    df.show(false)
    val sqlTrans = new SQLTransformer().setStatement(
      "SELECT *, (v1 + v2) AS v3, (v1 * v2) AS v4 FROM __THIS__")

    sqlTrans.transform(df).show(false)
    // $example off$

    spark.stop()
  }
}
