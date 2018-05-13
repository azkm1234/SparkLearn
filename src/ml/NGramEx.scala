package ml

import org.apache.spark.ml.feature.NGram
import org.apache.spark.sql.SparkSession

object NGramEx {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("NGramExample").master("local[2]")
      .getOrCreate()

    // $example on$
    val wordDataFrame = spark.createDataFrame(Seq(
      (0, Array("Hi", "I", "heard", "about", "Spark")),
      (1, Array("I", "wish", "Java", "could", "use", "case", "classes")),
      (2, Array("Logistic", "regression", "models", "are", "neat", "clear"))
    )).toDF("id", "words")

    val ngram = new NGram().setN(6).setInputCol("words").setOutputCol("ngrams")
    wordDataFrame.show(false)
    val ngramDataFrame = ngram.transform(wordDataFrame)
    ngramDataFrame.show(false)
    // $example off$

    spark.stop()
  }
}
