import pandas as pd
import random
from faker import Faker
import datetime
import os

print("Starting Data Generation...")

fake = Faker()

# Configuration
NUM_GENUINE_REVIEWS = 8000
NUM_FAKE_REVIEWS = 2000
TOTAL_REVIEWS = NUM_GENUINE_REVIEWS + NUM_FAKE_REVIEWS
PRODUCT_IDS = [f"PROD_{i:04d}" for i in range(1, 151)]  # 150 products

# Sample positive and negative phrases for fake reviews to make them repetitive
FAKE_POS_PHRASES = [
    "Best product ever completely changed my life!!",
    "Amazing quality will buy again!! 10/10",
    "Perfect product, exactly what I needed!!!",
    "I love this it works so well!!!",
    "Highly recommended beyond expectations!!!"
]

FAKE_NEG_PHRASES = [
    "Worst piece of garbage do not buy!!",
    "Terrible fake product completely broken!!",
    "Customer service is terrible do not recommend!!",
    "Scam product money wasted!!",
    "Warning do not purchase this item!!"
]

data = []

# 1. Generate Genuine Reviews
print(f"Generating {NUM_GENUINE_REVIEWS} genuine reviews...")
for _ in range(NUM_GENUINE_REVIEWS):
    user_id = f"USER_{random.randint(1000, 5000):04d}"
    product_id = random.choice(PRODUCT_IDS)
    rating = random.choices([1, 2, 3, 4, 5], weights=[10, 10, 20, 30, 30])[0]  # Normal distribution leaning to 4-5
    
    # 20-100 words of text
    review_text = fake.paragraph(nb_sentences=random.randint(2, 5))
    timestamp = fake.date_time_between(start_date="-1y", end_date="now").strftime("%Y-%m-%d %H:%M:%S")
    ip_address = fake.ipv4()
    
    data.append({
        "review_id": fake.uuid4(),
        "user_id": user_id,
        "product_id": product_id,
        "rating": rating,
        "review_text": review_text,
        "timestamp": timestamp,
        "ip_address": ip_address,
        "label": 0  # 0 indicates Genuine
    })

# 2. Generate Fake Reviews
print(f"Generating {NUM_FAKE_REVIEWS} fake reviews (Spam/Manipulative behavior)...")
# We'll create distinct "bot" users and IPs that mass-post reviews
BOT_USERS = [f"USER_BOT_{i:03d}" for i in range(1, 51)] # 50 bots
BOT_IPS = [fake.ipv4() for _ in range(20)] # 20 IPs shared among bots

for i in range(NUM_FAKE_REVIEWS):
    user_id = random.choice(BOT_USERS)
    product_id = random.choice(PRODUCT_IDS)
    
    # Fake reviews are generally extreme 1 or 5
    rating = random.choice([1, 5])
    
    if rating == 5:
        review_text = random.choice(FAKE_POS_PHRASES)
    else:
        review_text = random.choice(FAKE_NEG_PHRASES)
        
    # Generate timestamp but cluster them to simulate mass posting
    base_time = datetime.datetime.now() - datetime.timedelta(days=random.randint(1, 30))
    # Fake bots post very close to base_time
    offset = datetime.timedelta(minutes=random.randint(1, 120))
    timestamp = (base_time + offset).strftime("%Y-%m-%d %H:%M:%S")
    
    ip_address = random.choice(BOT_IPS)
    
    data.append({
        "review_id": fake.uuid4(),
        "user_id": user_id,
        "product_id": product_id,
        "rating": rating,
        "review_text": review_text,
        "timestamp": timestamp,
        "ip_address": ip_address,
        "label": 1  # 1 indicates Fake
    })

# Shuffle the data
print("Shuffling dataset...")
random.shuffle(data)

# Save to CSV
df = pd.DataFrame(data)
os.makedirs('dataset', exist_ok=True)
csv_path = os.path.join('dataset', 'ecommerce_reviews.csv')
df.to_csv(csv_path, index=False)

print(f"\nDataset generation complete!")
print(f"Total Records: {len(df)}")
print(f"File Saved At: {csv_path}")
print(df['label'].value_counts().rename({0: 'Genuine', 1: 'Fake'}))
