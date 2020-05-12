import org.apache.spark.SparkConf
import org.apache.spark.SparkContext 
import org.apache.spark.SparkContext._ 
import org.apache.spark._ 
import org.apache.log4j.{Level, Logger}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}


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
        val sqlContext = SparkSession
                    .builder()
                    .appName("Spark SQL basic example")
                    .config("spark.some.config.option", "some-value")
                    .getOrCreate()
        //val df = sqlContext.read.format("csv").option("header", "true").load(sourcefile)
        val df = sqlContext.read.options(Map("inferSchema"->"true","delimiter"->"\t","header"->"true")).csv(sourcefile)
        df.printSchema()
        df.show(false)
        df.na.drop().show(false)
        val cleanDF = df.select("review_body","star_rating")
        cleanDF.show(5)
        val tokenizer = new Tokenizer().setInputCol("review_body").setOutputCol("words")
        val wordsData = tokenizer.transform(cleanDF)

        val hashingTF = new HashingTF()
                            .setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)

        val featurizedData = hashingTF.transform(wordsData)

        val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
        val idfModel = idf.fit(featurizedData)

        val rescaledData = idfModel.transform(featurizedData)
        rescaledData.select("label", "features").show()
        
        
    }
}
