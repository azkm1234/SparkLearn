package ml

import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
object Tokenizer {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("TokenizerExample").master("local[2]")
      .getOrCreate()

    val sentenceDataFrame = spark.createDataFrame(Seq(
      (0, "Hi I heard about Spark"),
      (1, "I wish Java could use case classes"),
      (2, "Logistic,regression,models,are,neat")
    )).toDF("id", "sentence")

    val tokenizer: Tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")

    val regexTokenizer: RegexTokenizer = new RegexTokenizer().setInputCol("sentence").setOutputCol("words")
      .setPattern("\\W")

    val countTokens: UserDefinedFunction = udf((words:Seq[String]) => words.length)

    val tokenized = tokenizer.transform(sentenceDataFrame)
    tokenized.show(false)

    tokenized.select("sentence", "words")
      .withColumn("tokens", countTokens(col("words"))).show(false)

    val regexTokenized = regexTokenizer.transform(sentenceDataFrame)
    regexTokenized.show(false)
    regexTokenized.select("sentence", "words")
      .withColumn("tokens", countTokens(col("words"))).show(false)

    spark.stop()
  }
}
