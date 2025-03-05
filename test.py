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
# Convert 'WeekBeginning' to datetime for time-based analysis
df['WeekBeginning'] = pd.to_datetime(df['WeekBeginning'], format='%Y%m%d')
# Select the top recurring viruses
top_viruses = df['Pathogen'].value_counts().head(5).index
df_top_viruses = df[df['Pathogen'].isin(top_viruses)]
# 🔹 1. Box Plot: Identifying Variations & Outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_top_viruses, x="Pathogen", y="RatePer100000", hue="Pathogen", palette="coolwarm")
plt.legend(title="Virus", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("Variation in Virus Cases (Box Plot)")
plt.xlabel("Virus Type")
plt.ylabel("Rate per 100,000")
plt.grid()
plt.show()
# 🔹 2. Scatter Plot: Highlighting Unusual Variations Over Time
plt.figure(figsize=(14, 7))
sns.boxplot(data=df_top_viruses, x="Pathogen", y="RatePer100000", hue=None, palette="coolwarm", legend=False)
plt.title("Unusual Virus Case Variations Over Time")
plt.xlabel("Time (Weeks)")
plt.ylabel("Rate per 100,000")
plt.xticks(rotation=45)
plt.legend(title="Virus", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.show()
virus_counts = df['Pathogen'].value_counts()
# Display top recurring viruses
print("Top Recurring Viruses:\n", virus_counts.head(10))
# Analyze trends over time
df['WeekBeginning'] = pd.to_datetime(df['WeekBeginning'], format='%Y%m%d')  # Convert to datetime
virus_trends = df.groupby(['WeekBeginning', 'Pathogen']).size().unstack().fillna(0)
# Plot trends for the most common viruses
top_viruses = virus_counts.head(5).index  # Select top 5 viruses
virus_trends[top_viruses].plot(figsize=(12, 6), linewidth=2)
plt.title("Virus Recurrence Trends Over Time")
plt.xlabel("Week")
plt.ylabel("Number of Cases")
plt.legend(title="Virus")
plt.grid()
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
