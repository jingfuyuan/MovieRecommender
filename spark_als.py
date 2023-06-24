from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession, Row
import pandas as pd

spark = SparkSession.Builder().appName("moveirating").getOrCreate()

# set DATA_FILE to path of input file
DATA_FILE = "s3://aws-logs-335220673881-us-east-1/small_rating.csv"
OUTPUT_DIR = "s3://aws-logs-335220673881-us-east-1/"

# read the data into DataFrame
ratings = spark.read.csv(DATA_FILE, header=True, inferSchema=True)

# split the train test dataset
train, test = ratings.randomSplit([0.8, 0.2])

# Build the recommendation model using ALS on the training data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
als = ALS(rank=10, regParam=0.01, userCol="userId", itemCol="movieId", \
    ratingCol="rating", coldStartStrategy="drop")

model = als.fit(train)

# evaluate the train and test error
train_predict = model.transform(train)
evaluator = RegressionEvaluator(metricName="mse", labelCol="rating", predictionCol="prediction")
print("MSE on the train dataset is {}".format(evaluator.evaluate(train_predict)))

test_predict = model.transform(test)
print("MSE on the test dataset is {}".format(evaluator.evaluate(test_predict)))

# make movie recommendations for all the users
rec_no = 5
recommend = model.recommendForAllUsers(rec_no)
rc1 = recommend.select(["userId"] + \
    [recommend.recommendations[i][f] for i in range(rec_no) for f in ["movieId", "rating"]])
# save recommendations to a csv file
rc1.toPandas().to_csv(OUTPUT_DIR + "recommendForAllUsers.csv")