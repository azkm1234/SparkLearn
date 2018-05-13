package com.zjc.bs

import java.io.File

import com.zjc.bs.KmeansML.{numOfClusters, scaledFeatures}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{DataTypes, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession}

object KMeansML3 {
  val scaledFeatures = "scaledFeatures"
  val features = "features"
  val label = "label"
  val kmeansPrediction = "kmeansPrediction"
  val distanceToCentersVector = "distanceToCentersVector"
  val assembledFeatures = "assemblerFeatures"
  val indexedLabel = "indexedLabel"
  val rfPrediction = "rfPrediction"
  val predictedLabel = "predictedLabel"
  val pipeline2ModelPath = "data/pipeline2Model"
  val pipeline1ModelPath = "data/pipeline1Model"

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .master("local[4]")
      .appName("KmeansReader")
      .getOrCreate()
    val readData: DataFrame = readDataAndDataSchema(spark)

    val pipeline1ModelFile = new File(pipeline1ModelPath)
    val pipeline1Model = pipeline1ModelFile.exists() match {
      case true => PipelineModel.load(pipeline1ModelPath)
      case false => {
        val p1Model = getPipeLine1(spark, readData).fit(readData)
        p1Model.save(pipeline1ModelPath)
        p1Model
      }
    }
    //    val p1Data = pipeline1Model.transform(readData).select(indexedLabel, scaledFeatures, label)
    //    p1Data.printSchema()
    //    val kMeansModel = trainModel(p1Data, scaledFeatures)
    //    val KmeansedResult = kMeansModel
    //      .setPredictionCol(kmeansPrediction)
    //      .transform(p1Data)
    val pipeline1Result: DataFrame = pipeline1Model.transform(readData)
      .select("indexedLabel", "label", "scaledFeatures", "kmeansPrediction")
    pipeline1Result.printSchema()
    val kMeansModel = pipeline1Model.stages(3).asInstanceOf[KMeansModel]
    val convertedData = convertData(spark, pipeline1Result, kMeansModel, scaledFeatures)
    convertedData.printSchema()

    val Array(trainingData, testData) = convertedData.randomSplit(Array(0.7, 0.3))
    val labels = pipeline1Model.stages(1).asInstanceOf[StringIndexerModel].labels
    val pipeline2ModelFile = new File(pipeline2ModelPath)
    val pipeline2Model = pipeline2ModelFile.exists() match {
      case true => PipelineModel.load(pipeline2ModelPath)
      case false => {
        val p2Model = getPipeline2(spark, labels).fit(trainingData)
        p2Model.save(pipeline2ModelPath)
        p2Model
      }
    }
    val pipe2Result = pipeline2Model.transform(testData)
    pipe2Result.printSchema()

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol(indexedLabel)
      .setPredictionCol(rfPrediction)
      .setMetricName("accuracy")
    val accuracy: Double = evaluator.evaluate(pipe2Result)
    println("Test Error = " + (1.0 - accuracy))
    spark.stop()
  }
  private def predictData(data:DataFrame, spark:SparkSession) = {
    val pipeline1ModelFile = new File(pipeline1ModelPath)
    val pipeline1Model = pipeline1ModelFile.exists() match {
      case true => PipelineModel.load(pipeline1ModelPath)
      case false => {
        val p1Model = getPipeLine1(spark, data).fit(data)
        p1Model.save(pipeline1ModelPath)
        p1Model
      }
    }
    //    val p1Data = pipeline1Model.transform(readData).select(indexedLabel, scaledFeatures, label)
    //    p1Data.printSchema()
    //    val kMeansModel = trainModel(p1Data, scaledFeatures)
    //    val KmeansedResult = kMeansModel
    //      .setPredictionCol(kmeansPrediction)
    //      .transform(p1Data)
    val pipeline1Result: DataFrame = pipeline1Model.transform(data)
      .select("indexedLabel", "label", "scaledFeatures", "kmeansPrediction")
    pipeline1Result.printSchema()
    val kMeansModel = pipeline1Model.stages(3).asInstanceOf[KMeansModel]
    val convertedData = convertData(spark, pipeline1Result, kMeansModel, scaledFeatures)
    convertedData.printSchema()

    val Array(trainingData, testData) = convertedData.randomSplit(Array(0.7, 0.3))
    val labels = pipeline1Model.stages(1).asInstanceOf[StringIndexerModel].labels
    val pipeline2ModelFile = new File(pipeline2ModelPath)
    val pipeline2Model = pipeline2ModelFile.exists() match {
      case true => PipelineModel.load(pipeline2ModelPath)
      case false => {
        val p2Model = getPipeline2(spark, labels).fit(trainingData)
        p2Model.save(pipeline2ModelPath)
        p2Model
      }
    }
    val pipe2Result = pipeline2Model.transform(testData)
  }
  private def getPipeline2(Spark: SparkSession, labels: Array[String]): Pipeline = {
    val assembler = new VectorAssembler()
      .setInputCols(Array(scaledFeatures, distanceToCentersVector, kmeansPrediction))
      .setOutputCol(assembledFeatures)

    val rf = new RandomForestClassifier()
      .setLabelCol(indexedLabel)
      .setFeaturesCol(assembledFeatures)
      .setNumTrees(10)
      .setPredictionCol(rfPrediction)

    val labelConverter: IndexToString = new IndexToString()
      .setLabels(labels)
      .setInputCol(rfPrediction)
      .setOutputCol(predictedLabel)

    val pipeline2 = new Pipeline()
      .setStages(Array(assembler, rf, labelConverter))
    pipeline2
  }

  /**
    * 计算每一调数据到每一个center的距离的平方，
    * 并且作为特征值与原来的ScaledFeatures相结合
    *
    * @param spark
    * @param data
    * @param model
    * @return
    */
  private def convertData(spark: SparkSession, data: DataFrame, model: KMeansModel, colName: String): DataFrame = {
    val clusterCenters = model.clusterCenters
    val appendClusterCenter = udf((features: Vector) => {
      val r = clusterCenters.toArray.map { v1 =>
        Vectors.sqdist(v1, features)
      }
      Vectors.dense(r)
    })
    data.withColumn(distanceToCentersVector, appendClusterCenter(col(colName)))
  }

  /**
    * inputCol Vector -> kmeans -> Model or read Model
    *
    * @param data
    * @param inputCol
    * @return
    */
  /*
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
   */
  /**
    * assembler features -》 labelIndex -》 features scale-》输出model
    *
    * @param spark
    * @param data
    * @return
    */
  private def getPipeLine1(spark: SparkSession, data: DataFrame): Pipeline = {
    val featuresAssembler = new VectorAssembler()
      .setInputCols(Array("duration", "src_bytes", "dst_bytes", "wrong_fragment", "urgent", "hot", "num_failed_logins", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"))
      .setOutputCol(features)

    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol(indexedLabel)
      .fit(data)
    val scaler = new StandardScaler()
      .setInputCol(features)
      .setOutputCol(scaledFeatures)
    val kmeans = new KMeans()
      .setK(numOfClusters)
      .setSeed(1L)
      .setFeaturesCol(scaledFeatures)
      .setPredictionCol(kmeansPrediction)
    new Pipeline().setStages(Array(featuresAssembler, labelIndexer, scaler, kmeans))
  }

  private def readDataAndDataSchema(spark: SparkSession): DataFrame = {

    val texts: Array[String] = spark.read.textFile("data/kddcup.names.txt").collect()
    val lines = texts.slice(1, texts.length)

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
    allData
  }
}
