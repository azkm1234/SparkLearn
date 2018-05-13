package ml

import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.SparkSession

object TfidfEx {
  def main(args: Array[String]): Unit = {
    println(Double.PositiveInfinity)
  }
  def test(): Unit = {
    val spark = SparkSession.builder()
      .master("local[2]")
      .appName("tfidfEx")
      .getOrCreate()
    val sentensceData = spark.createDataFrame(Seq(
      (0.0, "Hi I heard about Spark"),
      (0.0, "I wish Java could use case classes"),
      (1.0, "Logistic regression models are neat")
    )).toDF("label", "sentence")
    sentensceData.show()

    val tokenizer = new Tokenizer()
      .setInputCol("sentence")
      .setOutputCol("words")

    val wordsData = tokenizer.transform(sentensceData)
    wordsData.show()

    val hashingTf = new HashingTF().setInputCol("words").setOutputCol("rawfeatures").setNumFeatures(20)
    val featurizedData = hashingTf.transform(wordsData)
    featurizedData.show()
    featurizedData.select("rawfeatures").foreach(println(_))
    val idf = new IDF().setInputCol("rawfeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)

    val rescaleData = idfModel.transform(featurizedData)
    rescaleData.show()
    rescaleData.select("label", "features").foreach(println(_))
    spark.stop()
  }
}
