/**
 *
 * @author spaggelos
 * 
 */


import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext 
import org.apache.spark.SparkContext._ 
import org.apache.spark._ 
import org.apache.log4j.{Level, Logger}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.optimization.{LBFGS, LogisticGradient, SquaredL2Updater}

object MyLbfgs{
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

    val conf = new SparkConf().setAppName("MyLbfgs")
    val sc = new SparkContext(conf)

    val inp = sc.textFile(sourcefile).coalesce(numOfPartitions).persist(StorageLevel.MEMORY_ONLY)
    val data = inp.map(LabeledPoint.parse)

    val numFeatures = data.take(1)(0).features.size
    //split data to training and test
    val splits = data.randomSplit(Array(0.7, 0.3), seed = 11L)
    val training = splits(0).map(x => (x.label, MLUtils.appendBias(x.features))).cache()
    val test = splits(1)

    val numCorrections = 10
    val convergenceTol = 1e-4
    val maxNumIterations = args(2).toInt
    val regParam = 0.1
    val initialWeightsWithIntercept = Vectors.dense(new Array[Double](numFeatures + 1))

    val (weightsWithIntercept, loss) = LBFGS.runLBFGS(
      training,
      new LogisticGradient(),
      new SquaredL2Updater(),
      numCorrections,
      convergenceTol,
      maxNumIterations,
      regParam,
      initialWeightsWithIntercept)

    val model = new LogisticRegressionModel(
      Vectors.dense(weightsWithIntercept.toArray.slice(0, weightsWithIntercept.size - 1)),
      weightsWithIntercept(weightsWithIntercept.size - 1))

    // Clear the default threshold.
    model.clearThreshold()

    // Compute raw scores on the test set.
    val scoreAndLabels = test.map { point =>
        val score = model.predict(point.features)
        (score, point.label)
      }

    // Get evaluation metrics.
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()

    println("Loss of each step in training process")
    loss.foreach(println)
    println(s"Area under ROC = $auROC")
  }
}

