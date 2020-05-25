import org.apache.spark.SparkConf
import org.apache.spark.SparkContext 
import org.apache.spark.SparkContext._ 
import org.apache.spark._ 
import org.apache.log4j.{Level, Logger}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.rdd.RDD 
import org.apache.spark.sql.Row
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.optimization.{LBFGS, LogisticGradient, SquaredL2Updater}
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator



object NlpApp{
    def main(args : Array[String]){
        var logger = Logger.getLogger(this.getClass())
        if(args.length < 3){
            logger.error("=> wrong parameters")
            System.err.println("Usage: <source file><number of partitions><max number of iterations>")
            System.exit(1)
        }
        Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
        Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
        val sourcefile = args(0)
        val numOfPartitions = args(1).toInt
        val spark = SparkSession
                .builder()
                .appName("Spark SQL basic example")
                .config("spark.some.config.option", "some-value")
                .getOrCreate()
        import spark.implicits._
        val df = spark.read.options(Map("inferSchema"->"true","delimiter"->"\t","header"->"true")).csv(sourcefile).where("review_body is not null")
        df.printSchema()
        val cleanDF = df.select("review_body","star_rating").withColumnRenamed("star_rating", "label")
        cleanDF.show(5)
        
        val splits = cleanDF.randomSplit(Array(0.8, 0.2), seed = 11L)
        val trainingData = splits(0);
        val testData = splits(1);
        trainingData.coalesce(numOfPartitions).persist(StorageLevel.MEMORY_ONLY)
        trainingData.cache
        val tokenizer = new Tokenizer().setInputCol("review_body").setOutputCol("words")
        //val wordsData = tokenizer.transform(cleanDF)
        val remover = new StopWordsRemover()
                    .setInputCol("words")
                    .setOutputCol("words_clean")
                    .setStopWords(StopWordsRemover.loadDefaultStopWords("english"))
        val hashingTF = new HashingTF().setInputCol("words_clean").setOutputCol("rawFeatures")
        //val featurizedData = hashingTF.transform(wordsData)

        val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
        //val idfModel = idf.fit(featurizedData)

        //val rescaledData = idfModel.transform(featurizedData)
        //rescaledData.select("star_rating", "features").show()
        
        //val data = rescaledData.select("star_rating", "features")
        //val list = List("label" , "features")
        //val completeData = data.toDF(list:_*)
        
        val mlr = new LogisticRegression().setMaxIter(10)
        
        val pipeline = new Pipeline()
            .setStages(Array(tokenizer, remover, hashingTF, idf, mlr))
        val paramGrid = new ParamGridBuilder()
            .addGrid(hashingTF.numFeatures, Array(100, 10000, 100000))
            .addGrid(lr.regParam, Array(0.1, 0.01, 0.001, 1e-10, 1e-5))
            .build()
        
        val evaluator = new MulticlassClassificationEvaluator()
            .setMetricName( "accuracy")
            .setLabelCol("label")
            .setPredictionCol("predictions")

        val cv = new CrossValidator()
            .setEstimator(pipeline)
            .setEvaluator(evaluator)
            .setEstimatorParamMaps(paramGrid)
            .setNumFolds(3)  // Use 3+ in practice
            .setParallelism(2)
        
        val cvModel = cv.fit(trainingData)
        /*
        cv.transform(trainingData)
            .select()
        */
        println(s"Coefficients: ${cvModel.coefficientMatrix} Intercept: ${cvModel.interceptVector}")
        
        val trainingSummary = cvModel.summary
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
        println(s"Accuracy: $accuracy\nFPR: $falsePositiveRate\nTPR: $truePositiveRate\n" +
            s"F-measure: $fMeasure\nPrecision: $precision\nRecall: $recall")
        
    }
}
