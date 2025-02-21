import pandas as pd

# -------------------- Load Datasets --------------------
diabetes_df = pd.read_csv("data/diabetes.csv")
heart_df = pd.read_csv("data/heart.csv")
parkinsons_df = pd.read_csv("data/parkinsons.csv")

# -------------------- Data Cleaning --------------------
# ✅ Diabetes Dataset
print("\nDiabetes Dataset Before Cleaning:")
print(diabetes_df.isnull().sum())  # Check for missing values

# Fill missing values (if any) with column means
diabetes_df.fillna(diabetes_df.mean(), inplace=True)

# ✅ Heart Disease Dataset
print("\nHeart Disease Dataset Before Cleaning:")
print(heart_df.isnull().sum())  # Check for missing values

# Convert categorical "Sex" column (M → 1, F → 0)
if "Sex" in heart_df.columns:
    heart_df["Sex"] = heart_df["Sex"].map({"M": 1, "F": 0})

# ✅ Parkinson’s Disease Dataset
print("\nParkinson’s Dataset Before Cleaning:")
print(parkinsons_df.isnull().sum())  # Check for missing values

# Drop unnecessary columns like "name"
if "name" in parkinsons_df.columns:
    parkinsons_df.drop(columns=["name"], inplace=True)

# -------------------- Save Cleaned Data --------------------
diabetes_df.to_csv("data/cleaned_diabetes.csv", index=False)
heart_df.to_csv("data/cleaned_heart.csv", index=False)
parkinsons_df.to_csv("data/cleaned_parkinsons.csv", index=False)

print("\n✅ Data cleaning completed and saved successfully!")
