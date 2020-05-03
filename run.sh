spark-submit --class NlpApp \
	./target/mylbfgs-1.0.jar `#location of the jar`\
	./src/main/resources/ `#input data`\
	5  `#number of partitions`\
	20 `#number of iterations`

#/home1/public/spaggelos/mysmallData
#/archive/users/spaggelos/logistic_big_data/part-00000
#hdfs://sith0-hadoop:9000/user/spaggelos/logistic_regression_data_18g_100p