# AI-Based Fake Review & Rating Manipulation Detection System Using Big Data

## 🎓 College Level Project - Big Data & Machine Learning

### 1. Problem Statement
With the exponential growth of e-commerce platforms, fake reviews and rating manipulations have become a significant problem. Malicious sellers often use bots or hire individuals to post fake positive reviews for their products or fake negative reviews for competitors' products. This project aims to build an automated, horizontally scalable Big Data system capable of processing large volumes of review data to detect and flag suspicious users, fake reviews, and manipulated product ratings using AI/ML and Natural Language Processing (NLP).

### 2. Objectives
- **Data Generation**: Create a massive sample dataset reproducing real-world fake review patterns (e.g., bot IPs, repetitive text, rating extremes).
- **Big Data Processing**: Utilize **Apache Spark (PySpark)** to process the large dataset efficiently.
- **Machine Learning**: Implement NLP (Tokenization, TF-IDF) and a `Logistic Regression` classifier via Spark MLlib to categorize reviews into `Genuine` vs `Fake`.
- **Suspicious Behavior Detection**: Use Spark SQL/Aggregations to identify IP addresses and users that demonstrate spam-like behavior.
- **Dashboard**: Provide a seamless, professional UI using `Streamlit` to visualize the metrics and manipulation alerts.

### 3. System Architecture
1. **Data Source**: Python Faker synthesizes ~10,000+ review entries into simulated HDFS/Local storage (`ecommerce_reviews.csv`).
2. **PySpark Engine**: The `big_data_processor.py` initializes a local SparkSession. It cleans data, applies NLP feature extraction, and trains the ML classifier.
3. **Analytics Engine**: Analyzes Spark DataFrame to hunt for users performing bot-like tasks and products exhibiting abnormal spikes in fake reviews. Output aggregated metrics to disk.
4. **Presentation Layer**: Streamlit reads the processed metrics and visualizes them using DataFrames, Pie Charts, and Scatter Plots.

### 4. Technology Stack
- **Language**: Python 3.8+
- **Big Data Engine**: Apache Spark (PySpark)
- **Machine Learning**: PySpark MLlib (Logistic Regression, TF-IDF)
- **Frontend / Dashboard**: Streamlit, Plotly
- **Data Handling**: Pandas, Numpy, Faker

---

### 5. Setup Instructions (Local Execution)

Follow these simple steps to run the complete project on any machine.

#### Step 5.1: Install Dependencies
Ensure you have Python installed. Open your terminal/command prompt and install the required modules:
```bash
pip install -r requirements.txt
```

*(Note: PySpark requires Java to be installed on your system. If you face WinUtils errors on Windows, PySpark local mode usually still completes the computation successfully despite warnings.)*

---

### 6. Run Commands for Viva / Execution

#### Command 1: Generate Big Dataset
This will synthesize 10,000 records of genuine and fake reviews.
```bash
py data_generator.py
```
*(Wait unill you see "File Saved At: dataset/ecommerce_reviews.csv")*

#### Command 2: Execute PySpark Big Data Processor
This executes the ML pipeline and Suspicious Behavior detection rules across the massive dataset.
```bash
py big_data_processor.py
```
*(You will see the Model Accuracy logged in the console. Results will be saved to the `processed_data` folder).*

#### Command 3: Launch Dashboard
Start the Streamlit application to visualize the insights.
```bash
py -m streamlit run app.py
```
*(This will automatically open your default browser with the interactive dashboard).*

---

### 7. Future Enhancements
- Integrations with **Apache Kafka** for real-time review stream processing.
- Deploying PySpark jobs onto a real Amazon EMR or Databricks cluster for petabyte-scale data.
- Applying deep learning models (like pre-trained HuggingFace BERT Transformers) for higher-accuracy semantic analysis.
