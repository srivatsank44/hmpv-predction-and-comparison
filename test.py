import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Fix for Windows CPU detection issu
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load dataset
file_path = "respiratory_sex_20250226.csv"  # Change this to your file path
df = pd.read_csv(file_path)
# Convert 'WeekBeginning' to datetime
df['WeekBeginning'] = pd.to_datetime(df['WeekBeginning'], format='%Y%m%d')
# Select top 5 viruses
top_viruses = df['Pathogen'].value_counts().head(5).index
df_top_viruses = df[df['Pathogen'].isin(top_viruses)]
# Group data by Pathogen and sum the total cases
virus_counts = df_top_viruses.groupby('Pathogen')['RatePer100000'].sum().reset_index()
# 🔹 Bar Chart: Distribution of Virus Cases with Variations
plt.figure(figsize=(12, 6))
sns.barplot(data=virus_counts, x="Pathogen", y="RatePer100000", palette="viridis", edgecolor="black")
plt.title("Total Cases of Top 5 Viruses")
plt.xlabel("Virus Type")
plt.ylabel("Total Rate per 100,000")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
# Convert 'Pathogen' into a binary target variable (1 = hMPV, 0 = other)
df['Target_hMPV'] = (df['Pathogen'] == "Human metapneumovirus").astype(int)
# Encode categorical 'Sex' column
sex_encoder = LabelEncoder()
df['Sex_encoded'] = sex_encoder.fit_transform(df['Sex'])
# Select features and target
# Convert datetime to numerical features
df['Week'] = df['WeekBeginning'].dt.isocalendar().week  # Extract week number
df['Year'] = df['WeekBeginning'].dt.year  # Extract year
model = lgb.LGBMClassifier(n_estimators=50, random_state=42, n_jobs=1)
# Select features and target (Remove original 'WeekBeginning' column)
X = df[['Week', 'Year', 'WeekEnding', 'Sex_encoded', 'RatePer100000']]
y = df['Target_hMPV']
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train LightGBM classifier
model = lgb.LGBMClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)
# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)
