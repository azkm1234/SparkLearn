package ml

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.sql.{DataFrame, SparkSession}

object RandomForestClassifierEx {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local[2]")
      .appName("RandomForestClassifierEx")
      .getOrCreate()
    val data = spark.read.format("libsvm").load("data/sample_libsvm_data.txt")
    data.printSchema()
    println("data schema")
    // Index labels, adding metadata to the label column.
    // Fit on whole dataset to include all labels in index.
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data)
    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(data)

    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    // Train a RandomForest model.
    val rf = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(10)

    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)
    val s2 = featureIndexer.transform(labelIndexer.transform(trainingData))
//    val s3 = rf.fit(s2).transform(s2)
//    labelConverter.transform(s3)

    val step2 = featureIndexer.transform(labelIndexer.transform(testData))
    step2.printSchema()
    println("step2")
    val step3 = rf.fit(s2).transform(step2)
    step3.printSchema()
    println("step3")
    val result = labelConverter.transform(step3)
    result.printSchema()
    // Chain indexers and forest in a Pipeline.
    val pipeline: Pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

    // Train model. This also runs the indexers.
    val model = pipeline.fit(trainingData)

    // Make predictions.
    testData.show(5, false)
    println("testData : ")
    val predictions = model.transform(testData)
    predictions.show(5, false)
    println("predictions : ")
    // Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(5)

    predictions.printSchema()
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - accuracy))

    val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
//    println("Learned classfication forest model : \n" +  rfModel.toDebugString)
  }
}
