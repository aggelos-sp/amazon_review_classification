from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import CountVectorizer
from pyspark.sql.types import StringType
from pyspark.sql.functions import array

data = spark.read.load("/home/eva/Documents/CS-543/project/amazon_review_classification/src/main/resources/amazon_reviews_us_Musical_Instruments_v1_00.tsv", format="csv", sep="\t", inferSchema="true", header="true")
data.printSchema()
data.show()
data.na.drop().show()

cleanData = data.select("review_body","star_rating")
newClean = cleanData.withColumn("review_body",array(cleanData["review_body"]))

cv = CountVectorizer(inputCol="review_body",outputCol="features", vocabSize=20)
model = cv.fit(newClean)
result = model.transform(newClean)

# tokenizer = Tokenizer(inputCol="review_body", outputCol="words")
# wordsData = tokenizer.transform(cleanData)
# newClean = wordsData.withColumn("review_body",array(wordsData["review_body"]))


# hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
# featurizedData = hashingTF.transform(newClean)
# alternatively, CountVectorizer can also be used to get term frequency vectors

# idf = IDF(inputCol="rawFeatures", outputCol="features")
# idfModel = idf.fit(featurizedData)
# rescaledData = idfModel.transform(featurizedData)

# rescaledData.select("label", "features").show()