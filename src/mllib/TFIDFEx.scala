package mllib


import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.mllib.linalg
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object TFIDFEx {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("TFIDFExample").setMaster("local[1]")
    val sc = new SparkContext(conf)
    val documents: RDD[Seq[String]] = sc.textFile("data/kmeans_data.txt").map{_.split(' ').toSeq}
    val hashingTF = new HashingTF()
    documents.foreach(println(_))

    val tf: RDD[linalg.Vector] = hashingTF.transform(documents)

    tf.foreach(println)
    tf.cache()
    val idf = new IDF().fit(tf)
    println(s"idf :  $idf")
    val tfidf: RDD[linalg.Vector] = idf.transform(tf)
    tfidf.foreach(println)

    val idfIgnore = new IDF(minDocFreq = 2).fit(tf)
    val tfidfIgnore: RDD[linalg.Vector] = idfIgnore.transform(tf)
    // $example off$

    println("tfidf: ")
    tfidf.foreach(x => println(x))

    println("tfidfIgnore: ")
    tfidfIgnore.foreach(x => println(x))
    sc.stop()
  }
}
