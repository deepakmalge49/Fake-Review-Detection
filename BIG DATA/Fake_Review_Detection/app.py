import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import plotly.graph_objects as go

# Configure Streamlit Page
st.set_page_config(
    page_title="Big Data Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI styling
st.markdown("""
    <style>
        .main { background-color: #f4f6f9; }
        .metric-card { 
            background-color: white; 
            padding: 20px; 
            border-radius: 10px; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
            text-align: center;
        }
        h1, h2, h3 { color: #2c3e50; }
        .stAlert { border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# Helper function to load data safely
def load_data():
    base_dir = "processed_data"
    
    metrics_file = os.path.join(base_dir, "metrics.json")
    suspicious_users_file = os.path.join(base_dir, "suspicious_users.csv")
    manipulated_products_file = os.path.join(base_dir, "manipulated_products.csv")

    try:
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
        
        suspicious_users_df = pd.read_csv(suspicious_users_file)
        manipulated_products_df = pd.read_csv(manipulated_products_file)
        
        recent_predictions_file = os.path.join(base_dir, "recent_predictions.csv")
        if os.path.exists(recent_predictions_file):
            recent_predictions_df = pd.read_csv(recent_predictions_file)
        else:
            recent_predictions_df = None
            
        dataset_df = pd.read_csv(os.path.join("dataset", "ecommerce_reviews.csv"))
        
        # Load a sample for preview
        return metrics, suspicious_users_df, manipulated_products_df, dataset_df, recent_predictions_df
    except Exception as e:
        return None, None, None, None, None

# App Layout & Header
st.title("🛡️ AI-Based Fake Review & Rating Manipulation Detection System")
st.markdown("**Powered by Apache Spark, Machine Learning (NLP), and Python**")
st.markdown("---")

# Load Data
metrics, suspicious_users_df, manipulated_products_df, dataset_df, recent_predictions_df = load_data()

if metrics is None:
    st.error("Processed data not found! Please ensure you have run `data_generator.py` followed by `big_data_processor.py`.")
else:
    # Top Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
            <div class='metric-card'>
                <h3 style='margin:0; font-size:1.2rem;'>Total Reviews Analyzed</h3>
                <h1 style='color:#3498db; margin:0;'>{metrics['total_reviews']:,}</h1>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class='metric-card'>
                <h3 style='margin:0; font-size:1.2rem;'>Genuine Reviews</h3>
                <h1 style='color:#2ecc71; margin:0;'>{metrics['genuine_reviews']:,}</h1>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class='metric-card'>
                <h3 style='margin:0; font-size:1.2rem;'>Fake Reviews Blocked</h3>
                <h1 style='color:#e74c3c; margin:0;'>{metrics['fake_reviews']:,}</h1>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div class='metric-card'>
                <h3 style='margin:0; font-size:1.2rem;'>ML Model Accuracy</h3>
                <h1 style='color:#f39c12; margin:0;'>{metrics['model_accuracy']*100:.1f}%</h1>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Core Visualizations
    st.subheader("📊 Big Data Metrics Overview")
    row1_c1, row1_c2 = st.columns(2)

    with row1_c1:
        # Pie chart of review distribution
        labels = ['Genuine', 'Fake']
        values = [metrics['genuine_reviews'], metrics['fake_reviews']]
        fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4, marker_colors=['#2ecc71', '#e74c3c'])])
        fig_pie.update_layout(title_text="Review Authenticity Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)

    with row1_c2:
        if "confusion_matrix" in metrics:
            cm_data = metrics["confusion_matrix"]
            cm_df = pd.DataFrame(cm_data)
            cm_df["Actual"] = cm_df["Actual"].map({0: 'Genuine', 1: 'Fake'})
            cm_df["Predicted"] = cm_df["Predicted"].map({0: 'Genuine', 1: 'Fake'})
            # Create a pivot table for the heatmap
            cm_pivot = cm_df.pivot(index="Actual", columns="Predicted", values="Count").fillna(0)
            
            fig_cm = px.imshow(cm_pivot, labels=dict(x="Predicted", y="Actual", color="Count"), 
                               x=cm_pivot.columns, y=cm_pivot.index, color_continuous_scale='Blues', text_auto=True)
            fig_cm.update_layout(title_text="ML Model Confusion Matrix")
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            # Fallback to Rating Distribution
            rating_counts = dataset_df['rating'].value_counts().sort_index().reset_index()
            rating_counts.columns = ['Rating', 'Count']
            fig_bar = px.bar(rating_counts, x='Rating', y='Count', title='Overall Rating Distribution', 
                             labels={'Count': 'Number of Reviews'}, color='Count', color_continuous_scale='Blues')
            st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # Alerts Section
    st.subheader("🚨 Fraud & Manipulation Alerts")
    
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 👤 Suspicious Users Detected")
        if not suspicious_users_df.empty:
            st.warning(f"Detected {len(suspicious_users_df)} users exhibiting extreme bot/spam behavior.")
            # Format display
            display_users = suspicious_users_df.copy()
            display_users.columns = ['User ID', 'Total Fake Reviews Detected']
            st.dataframe(display_users.head(15), use_container_width=True)
        else:
            st.success("No highly suspicious users detected.")

    with col_b:
        st.markdown("### 📦 Rating Manipulation on Products")
        if not manipulated_products_df.empty:
            st.error(f"Detected {len(manipulated_products_df)} products with artificial rating manipulation.")
            
            # Format display
            display_products = manipulated_products_df.copy()
            display_products.columns = ['Product ID', 'Fake Reviews On Product', 'Average Fake Rating']
            display_products['Average Fake Rating'] = display_products['Average Fake Rating'].round(2)
            st.dataframe(display_products.head(15), use_container_width=True)
            
            # Scatter Plot for manipulated products
            fig_scatter = px.scatter(manipulated_products_df, x="fake_reviews_on_product", y="average_fake_rating",
                                     size="fake_reviews_on_product", color="average_fake_rating",
                                     hover_name="product_id", title="Manipulated Products Analysis")
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.success("No products flag for severe manipulation.")
            
    st.markdown("---")
    st.subheader("🔍 Expected vs Predicted Results (Live Data Stream)")
    if recent_predictions_df is not None and not recent_predictions_df.empty:
        # Display the real PySpark predictions
        display_df = recent_predictions_df[['review_id', 'user_id', 'product_id', 'rating', 'label', 'prediction']].copy()
        display_df['Actual'] = display_df['label'].map({0.0: 'Genuine', 1.0: 'Fake'})
        display_df['Predicted'] = display_df['prediction'].map({0.0: 'Genuine', 1.0: 'Fake'})
        display_df['Status'] = display_df.apply(lambda r: '✅ Correct' if r['Actual'] == r['Predicted'] else '❌ Incorrect', axis=1)
        
        display_df = display_df[['review_id', 'user_id', 'product_id', 'rating', 'Actual', 'Predicted', 'Status']]
        st.dataframe(display_df.head(20), use_container_width=True)
    else:
        # Fallback to dataset preview
        st.dataframe(dataset_df[['review_id', 'user_id', 'product_id', 'rating', 'review_text', 'label']].head(10), use_container_width=True)
