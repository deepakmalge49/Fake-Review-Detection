import os
import json
import socketserver
if not hasattr(socketserver, "UnixStreamServer"):
    socketserver.UnixStreamServer = socketserver.TCPServer
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, max as spark_max, window, desc
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

print("Starting Big Data Processor (PySpark)...")

# 1. Initialize Spark Session Local Mode (Simulates Big Data Environment)
spark = SparkSession.builder \
    .appName("FakeReviewDetection") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# 2. Load Data from HDFS/Local System
dataset_path = os.path.join("dataset", "ecommerce_reviews.csv")
if not os.path.exists(dataset_path):
    print(f"Error: Dataset not found at {dataset_path}. Please run data_generator.py first.")
    spark.stop()
    exit(1)

print("Loading data into PySpark DataFrame...")
df = spark.read.csv(dataset_path, header=True, inferSchema=True)
df = df.dropna()

# Cast label to double as required by MLlib
df = df.withColumn("label", col("label").cast("double"))

counts = df.groupBy("label").count().collect()
total_reviews = sum([row['count'] for row in counts])
fake_reviews = next((row['count'] for row in counts if row['label'] == 1.0), 0)
genuine_reviews = next((row['count'] for row in counts if row['label'] == 0.0), 0)

print(f"Total Loaded Reviews: {total_reviews}")

# 3. NLP & Machine Learning Pipeline
print("Configuring NLP & ML Pipeline...")

# Regular Expression Tokenizer
tokenizer = RegexTokenizer(inputCol="review_text", outputCol="words", pattern="\\W")

# Stop Words Remover
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

# Hashing TF
hashingTF = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=10000)

# IDF (Inverse Document Frequency)
idf = IDF(inputCol="rawFeatures", outputCol="features")

# Logistic Regression Model for Classification
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)

# Build Pipeline
pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])

# Split Data into Train and Test
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Train the Model
print("Training Logistic Regression Model on Big Data...")
model = pipeline.fit(train_data)

# Predict on Test Data to evaluate
print("Evaluating Model...")
predictions = model.transform(test_data)

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Predict on Full Dataset to find predictions for dashboard logging
full_predictions = model.transform(df)

# Rename prediction column to predicted_label
final_df = full_predictions.select("review_id", "user_id", "product_id", "rating", "timestamp", "ip_address", "label", "prediction")

# 4. Big Data Analysis: Suspicious User Detection
print("Running Suspicious User Analysis...")
# Users with more than 3 fake reviews
suspicious_users_df = final_df.filter(col("prediction") == 1.0) \
    .groupBy("user_id") \
    .agg(count("review_id").alias("fake_review_count")) \
    .filter(col("fake_review_count") > 3) \
    .orderBy(desc("fake_review_count"))

# 5. Big Data Analysis: Rating Manipulation Alerts
print("Running Rating Manipulation Analysis...")
# Products with a high volume of fake reviews (e.g. > 10 fake reviews)
manipulated_products_df = final_df.filter(col("prediction") == 1.0) \
    .groupBy("product_id") \
    .agg(count("review_id").alias("fake_reviews_on_product"),
         avg("rating").alias("average_fake_rating")) \
    .filter(col("fake_reviews_on_product") > 10) \
    .orderBy(desc("fake_reviews_on_product"))

# 6. Save Processed Results to Disk for Streamlit Dashboard
print("Saving results for Dashboard visualization...")
os.makedirs("processed_data", exist_ok=True)

# Convert small aggregated data to Pandas for easy loading in UI
suspicious_users_pd = suspicious_users_df.toPandas()
suspicious_users_pd.to_csv(os.path.join("processed_data", "suspicious_users.csv"), index=False)

manipulated_products_pd = manipulated_products_df.toPandas()
manipulated_products_pd.to_csv(os.path.join("processed_data", "manipulated_products.csv"), index=False)

# Save a sample of the processed predictions for the "Live Data Stream" UI
recent_predictions_df = final_df.orderBy(desc("timestamp")).limit(100)
recent_predictions_pd = recent_predictions_df.toPandas()
recent_predictions_pd.to_csv(os.path.join("processed_data", "recent_predictions.csv"), index=False)

# Calculate confusion matrix on test_data
confusion_matrix = predictions.groupBy("label", "prediction").count().collect()
cm_data = [{"Actual": int(row["label"]), "Predicted": int(row["prediction"]), "Count": row["count"]} for row in confusion_matrix]

# Save metrics JSON
metrics = {
    "total_reviews": total_reviews,
    "genuine_reviews": genuine_reviews,
    "fake_reviews": fake_reviews,
    "model_accuracy": accuracy,
    "confusion_matrix": cm_data
}

with open(os.path.join("processed_data", "metrics.json"), "w") as f:
    json.dump(metrics, f)

print("Big Data Processing Complete! Data is ready for Streamlit Dashboard.")
spark.stop()
