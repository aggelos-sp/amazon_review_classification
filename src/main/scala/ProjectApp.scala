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
            //val conf = new SparkConf().setAppName("NLP APP")
            //val sc = new SparkContext(conf)
            //val sqlContext = new org.apache.spark.sql.SQLContext(sc)
            val spark = SparkSession
                    .builder()
                    .appName("Spark SQL basic example")
                    .config("spark.some.config.option", "some-value")
                    .getOrCreate()
            import spark.implicits._
            //val df = sqlContext.read.format("csv").option("header", "true").load(sourcefile)
            //val df = spark.read.options(Map("inferSchema"->"true","delimiter"->"\t","header"->"true")).csv(sourcefile)
            val df = spark.read.options(Map("inferSchema"->"true","delimiter"->"\t","header"->"true")).csv("/home/aggelosspiliotis/Documents/hy543/amazon_review_classification/src/main/resources").where("review_body is not null")
            df.printSchema()
            //df.show(false)
            //df.na.drop().show(false)
            val cleanDF = df.select("review_body","star_rating")
            cleanDF.show(5)
            val tokenizer = new Tokenizer().setInputCol("review_body").setOutputCol("words")
            val wordsData = tokenizer.transform(cleanDF)

            val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)
            val featurizedData = hashingTF.transform(wordsData)

            val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
            val idfModel = idf.fit(featurizedData)

            val rescaledData = idfModel.transform(featurizedData)
            rescaledData.select("star_rating", "features").show()
        
            val data = rescaledData.select("star_rating", "features")
            val list = List("label" , "features")
            val completeData = data.toDF(list:_*)
            val splits = completeData.randomSplit(Array(0.7, 0.3), seed = 11L)
            val trainingData = splits(0);
            val testData = splits(1);
        
            val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
            val lrModel = lr.fit(trainingData)
            println(s"Coefficients: ${lrModel.coefficientMatrix} Intercept: ${lrModel.interceptVector}")
        
            val trainingSummary = lrModel.summary
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
