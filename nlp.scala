  
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext 
import org.apache.spark.SparkContext._ 
import org.apache.spark._ 
import org.apache.log4j.{Level, Logger}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer, StopWordsRemover, IDFModel}
import org.apache.spark.rdd.RDD 
import org.apache.spark.sql.Row
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import spark.implicits._

//val awsAccessKeyId = "ASIA3ZPJMDVE7Q44I5MD"
//val awsSecretAccessKey = "GQKVe/qKKC022+Sp4RFwk6UcWsN3Uedj6w1BF/0"
Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
//spark.sparkContext.hadoopConfiguration.set("fs.s3a.awsAccessKeyId", awsAccessKeyId)
//spark.sparkContext.hadoopConfiguration.set("fs.s3a.awsSecretAccessKey", awsSecretAccessKey)
//spark.sparkContext.hadoopConfiguration.set("fs.s3a.impl", "org.apache.hadoop.fs.s3native.NativeS3FileSystem")
val sourcefile = "s3a://mynlpdata/"
//val sourcefile = "/home/ec2-user/data"
//val sourcefile = "/home/ec2-user/data/amazon_reviews_us_Toys_v1_00.tsv.gz"
val df = spark.read.options(Map("inferSchema"->"true","delimiter"->"\t","header"->"true")).csv(sourcefile).where("review_body is not null")

df.printSchema()
val clean = df.select("review_body","star_rating").withColumnRenamed("star_rating", "label")
clean.printSchema
val cleanDF = clean.selectExpr("cast (label as int) label","review_body")
//val cleanDF = x.na.replace("label", Map(1->0, 2->0, 3->0, 4->1, 5->1))
//cleanDF.show(5)
/*
val splits = cleanDF.randomSplit(Array(0.8, 0.2), seed = 11L)
val trainingData = splits(0)
val testData = splits(1)

trainingData.coalesce(32).persist(StorageLevel.MEMORY_ONLY)
trainingData.cache
*/
val tokenizer = new Tokenizer().setInputCol("review_body").setOutputCol("words")
val wordsData = tokenizer.transform(cleanDF)
val remover = new StopWordsRemover().setInputCol("words").setOutputCol("words_clean").setStopWords(StopWordsRemover.loadDefaultStopWords("english"))
val dat = remover.transform(wordsData)
val hashingTF = new HashingTF().setInputCol("words_clean").setOutputCol("rawFeatures").setNumFeatures(100000)
val featurizedData = hashingTF.transform(dat)

val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
val idfModel = idf.fit(featurizedData)
val rescaledData = idfModel.transform(featurizedData)
//rescaledData.select("star_rating", "features").show()
        
//val data = rescaledData.select("star_rating", "features")
//val list = List("label" , "features")
//val completeData = data.toDF(list:_*)
/*
val mlr = new LogisticRegression().setMaxIter(10)
        
val pipeline = new Pipeline().setStages(Array(tokenizer, remover, hashingTF, idf, mlr))
val paramGrid = new ParamGridBuilder().addGrid(hashingTF.numFeatures, Array(100, 10000, 100000)).addGrid(mlr.regParam, Array(0.1, 0.01, 0.001, 1e-10, 1e-5)).build()
        
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy").setLabelCol("label").setPredictionCol("predictions")

val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(2)
        
val cvModel = cv.fit(trainingData)

cv.transform(trainingData).select()
*/
val mlr = new LogisticRegression()//.setMaxIter(10).setRegParam(0.1).setElasticNetParam(0.8)
val data = rescaledData.select("features", "label")
//val data = d.na.replace("label", Map(1->0, 2->0, 3->0, 4->1, 5->1))
//val tr = data.selectExpr("cast (label as int) label","features")
val splits = data.randomSplit(Array(0.8, 0.2), seed = 11L)
val trainingData = splits(0)
val testData = splits(1)
val mlrModel = mlr.fit(trainingData)
println(s"Coefficients: ${mlrModel.coefficientMatrix} Intercept: ${mlrModel.interceptVector}")
        
val trainingSummary = mlrModel.summary
val objectiveHistory = trainingSummary.objectiveHistory
println("objectiveHistory:")
objectiveHistory.foreach(loss => println(loss))
        
println("False positive rate by label:")
trainingSummary.falsePositiveRateByLabel.zipWithIndex.foreach { case (rate, label) =>
    println(s"label $label: $rate")
}
println("True positive rate by label:")
trainingSummary.truePositiveRateByLabel.zipWithIndex.foreach { case (rate, label) =>
    println(s"label $label: $rate")
}
println("Precision by label:")
trainingSummary.precisionByLabel.zipWithIndex.foreach { case (prec, label) =>
    println(s"label $label: $prec")
}
println("Recall by label:")
trainingSummary.recallByLabel.zipWithIndex.foreach { case (rec, label) =>
    println(s"label $label: $rec")
}
println("F-measure by label:")
trainingSummary.fMeasureByLabel.zipWithIndex.foreach { case (f, label) =>
    println(s"label $label: $f")
}
val accuracy = trainingSummary.accuracy
val falsePositiveRate = trainingSummary.weightedFalsePositiveRate
val truePositiveRate = trainingSummary.weightedTruePositiveRate
val fMeasure = trainingSummary.weightedFMeasure
val precision = trainingSummary.weightedPrecision
val recall = trainingSummary.weightedRecall
println(s"Accuracy: $accuracy\nFPR: $falsePositiveRate\nTPR: $truePositiveRate\n" +s"F-measure: $fMeasure\nPrecision: $precision\nRecall: $recall")


import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils

val predictions = mlrModel.transform(testData)
val cleanPreds = predictions.select("prediction", "label")
val cl = cleanPreds.selectExpr("cast (label as double) label","prediction")
val predictionAndLabels =cl.rdd.map{case Row(prediction: Double, label: Double) => (prediction, label)}

val metrics = new MulticlassMetrics(predictionAndLabels)
val accuracy = metrics.accuracy
println("Summary Statistics")
println(s"Accuracy = $accuracy")

// Precision by label
val labels = metrics.labels
labels.foreach { l =>
  println(s"Precision($l) = " + metrics.precision(l))
}

// Recall by label
labels.foreach { l =>
  println(s"Recall($l) = " + metrics.recall(l))
}

// False positive rate by label
labels.foreach { l =>
  println(s"FPR($l) = " + metrics.falsePositiveRate(l))
}

// F-measure by label
labels.foreach { l =>
  println(s"F1-Score($l) = " + metrics.fMeasure(l))
}

// Weighted stats
println(s"Weighted precision: ${metrics.weightedPrecision}")
println(s"Weighted recall: ${metrics.weightedRecall}")
println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")
println(s"Weighted true positive rate: ${metrics.weightedTruePositiveRate}")

data.createOrReplaceTempView("table")
spark.sql("select count(label) from pap where label=1").show
spark.sql("select count(label) from pap where label=2").show
spark.sql("select count(label) from pap where label=3").show
spark.sql("select count(label) from pap where label=4").show
spark.sql("select count(label) from pap where label=5").show
spark.sql("select avg(label) from pap").show