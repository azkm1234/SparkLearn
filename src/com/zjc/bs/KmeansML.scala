package com.zjc.bs

import java.io.File

import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.{StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.{SparseVector, Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.types.{DataType, DataTypes, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.sql.functions._


object KmeansML {
  val scaledFeatures = "scaledFeatures"
  val numOfClusters = 23
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .master("local[4]")
      .appName("KmeansReader")
      .getOrCreate()
    println("labeledData :")
    val labeledData = prepareTrainingData(spark)
    labeledData.show(5, false)
//    在labeledData当中应该插入正规化

    val scaledData = scaleData(labeledData)

    scaledData.show(5, false)
    println("scaledData :")
    println("data length : " + labeledData.count())

    val model: KMeansModel = trainModel(scaledData, scaledFeatures)
    val predictions = model.transform(scaledData, ParamMap(model.featuresCol -> scaledFeatures))
//    helper(predictions)

    predictions.show(5, false)
    println("predictions : ")
    val cData = convertData(spark, predictions, model, scaledFeatures)
    cData.show(5, false)
    println("cData")
    val aData = assembleData(cData, scaledFeatures,"distanceToCenter","prediction")
      .drop("scaledFeatures", "distanceToCenter", "prediction")

    aData.show(5, false)
    println("aData : ")
    println("aData length : " + aData.count())

    spark.stop()
  }
  private def helper(data:DataFrame) = {
    data.groupBy("label")
      .count().orderBy("count").write.csv("data/countByLabel")
    data.groupBy("prediction").
      count().orderBy("count").write.csv("data/countByPrediction")
  }
  /** features scale into scaledFeatures
    * scaled Features col
    * @param data
    * @return
    */
  private def scaleData(data:DataFrame):DataFrame = {
    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol(scaledFeatures)
    scaler.fit(data)
      .transform(data)
      .drop("features")
  }
  //assemble features and DdistanceToCenter, prediction
  private def assembleData(data:DataFrame, col1:String, col2:String, col3:String):DataFrame = {
    val assember = new VectorAssembler()
      .setInputCols(Array(col1, col2, col3))
      .setOutputCol("assemberFeatures")
    assember.transform(data)
  }

  /**
    * 计算每一个簇的个数并且左连接\n
    * 计算每一调数据到每一个center的距离的平方，\n
    * 并且作为特征值与原来的ScaledFeatures相结合
    * @param spark
    * @param data
    * @param model
    * @return
    */
  private def convertData(spark:SparkSession,data:DataFrame, model: KMeansModel, colName:String):DataFrame = {
    val countData = data.groupBy("prediction").count()
    println("countData : ")
    countData.orderBy("count").show(39, false)
    countData.createTempView("count")
    data.createTempView("data")
    val nData = spark.sql("select d.*, c.count from data d left join count c " +
      "on d.prediction = c.prediction")
//    对对表格的连接理解还不到位
//    val step1 = data.join(countData, data("prediction") === countData("prediction"))
//    println("step1 length : " + step1.count())
    println("countData.length" + countData.count())
    val clusterCenters = model.clusterCenters
    val appendClusterCenter = udf((features:Vector) => {
      val r = clusterCenters.toArray.map{ v1 =>
        Vectors.sqdist(v1, features)
      }
      Vectors.dense(r)
    })
    nData.withColumn("distanceToCenter", appendClusterCenter(col(colName)))
  }

  /**
    * 查看modelPath下是否有model模型了,如果有读取，如果没有训练模型
    * @param data
    * @return
    */
  private def trainModel(data:DataFrame, inputCol:String): KMeansModel = {
    val modelpath = new File("data/model/kmeans1.model")

    if (!modelpath.exists) {
      val kmeans = new KMeans().setK(numOfClusters).setSeed(1L)
      val map = ParamMap(kmeans.featuresCol -> inputCol)
      val model: KMeansModel = kmeans.fit(data, map)
      model.write.overwrite().save("data/model/kmeans1.model")
      return model
    }
    return KMeansModel.load("data/model/kmeans1.model")
  }

  /**
    * 导入name， 创建trainingData，导入col name和type， assemble data,
    * 输出 features 和 label
    * @param spark
    * @return
    */
  private def prepareTrainingData(spark : SparkSession): DataFrame = {
    var flag = false
    val lines: Array[String] = spark.read.textFile("data/kddcup.names.txt").filter(line => {
      flag match {
        case false => {
          flag = true; false
        }
        case true => true
      }
    }).collect()

    val structFields: Array[StructField] = lines.map { x =>
      val tmp = x.split(":").map(_.trim())
      val s_type = tmp(1) match {
        case "continuous." => DataTypes.FloatType
        case "symbolic." => DataTypes.StringType
      }
      StructField(tmp(0), s_type)
    } :+ (StructField("label", DataTypes.StringType))

    import spark.implicits._
    val allData: DataFrame = spark.read.format("csv").schema(StructType(structFields))
      .load("data/kddcup.data.corrected")

    val featuresAssember = new VectorAssembler()
      .setInputCols(Array("duration", "src_bytes", "dst_bytes", "wrong_fragment", "urgent", "hot", "num_failed_logins", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"))
      .setOutputCol("features")

    val indexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("newLabel")


    val featuresData = featuresAssember.transform(allData).select("features","label")
    val labeledData = indexer.fit(featuresData).transform(featuresData)
    labeledData
  }
  def resoveHelper(string: String): DataType = {
    string match {
      case "continuous." => DataTypes.FloatType
      case _ => DataTypes.StringType
    }
  }
}
