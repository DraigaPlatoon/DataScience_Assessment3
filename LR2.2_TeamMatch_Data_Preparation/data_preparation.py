import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Check current working directory
print("Working directory:", os.getcwd())
print("Files present:", os.listdir())
print("-" * 50)

# Load the raw data
df = pd.read_csv("lr2_2_raw_data.csv")
print("Data loaded. Shape:", df.shape)

# Remove completely empty rows
df = df.dropna(how="all")
print("After dropping empty rows:", df.shape)

# Convert columns to proper types
int_cols = ["Match_ID", "Goals", "Team_FIFA_Rank", "Opponent_FIFA_Rank",
            "Is_Host", "Is_Knockout", "Team_Squad_Value", "Days_Rest"]

for col in int_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

float_cols = ["Team_Avg_Goals_Scored", "Team_Avg_Goals_Conceded"]
for col in float_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nFirst few rows:")
print(df.head())

# Save cleaned version
df.to_csv("lr2_2_cleaned.csv", index=False)
print("\nCleaned data saved as lr2_2_cleaned.csv")

# ----------------------------------------
# Basic statistics
# ----------------------------------------
print("\n" + "="*50)
print("BASIC STATISTICS")
print("="*50)

print("\nGoals:")
print(df["Goals"].describe())

print("\nTeam FIFA Rank:")
print(df["Team_FIFA_Rank"].describe())

print("\nPre-match average goals scored:")
print(df["Team_Avg_Goals_Scored"].describe())

print("\nDays of rest:")
print(df["Days_Rest"].describe())

print("\nHost status counts:")
print(df["Is_Host"].value_counts())

print("\nStage counts (0=Group, 1=Knockout):")
print(df["Is_Knockout"].value_counts())

# ----------------------------------------
# Graphs
# ----------------------------------------
print("\nCreating graphs...")

# 1. Goals distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Goals"], bins=8, kde=True, color="steelblue", edgecolor="black")
plt.title("Distribution of Goals Scored")
plt.xlabel("Goals")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("01_goals_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

# 2. Goals vs FIFA Rank
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Team_FIFA_Rank", y="Goals",
                hue="Is_Knockout", palette="Set1", s=60)
plt.title("Goals vs Team FIFA Rank")
plt.xlabel("FIFA Rank (lower is stronger)")
plt.ylabel("Goals Scored")
plt.legend(title="Knockout")
plt.tight_layout()
plt.savefig("02_goals_vs_rank.png", dpi=300, bbox_inches="tight")
plt.close()

# 3. Boxplot - Group vs Knockout
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Is_Knockout", y="Goals", hue="Is_Knockout",
            palette="Set2", legend=False)
plt.title("Goals: Group Stage vs Knockout")
plt.xlabel("0 = Group Stage   |   1 = Knockout")
plt.ylabel("Goals")
plt.tight_layout()
plt.savefig("03_goals_by_stage.png", dpi=300, bbox_inches="tight")
plt.close()

# 4. Average goals - Host vs Non-host
plt.figure(figsize=(7, 5))
avg_host = df.groupby("Is_Host")["Goals"].mean().reset_index()
sns.barplot(data=avg_host, x="Is_Host", y="Goals", hue="Is_Host",
            palette="pastel", legend=False)
plt.title("Average Goals by Host Status")
plt.xlabel("0 = Non-host   |   1 = Host")
plt.ylabel("Average Goals")
plt.tight_layout()
plt.savefig("04_avg_goals_host.png", dpi=300, bbox_inches="tight")
plt.close()

# 5. Extra graph - Correlation heatmap of key variables
plt.figure(figsize=(8, 6))
corr_cols = ["Goals", "Team_FIFA_Rank", "Opponent_FIFA_Rank",
             "Team_Avg_Goals_Scored", "Team_Avg_Goals_Conceded", "Days_Rest"]
corr = df[corr_cols].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Correlation between Key Variables")
plt.tight_layout()
plt.savefig("05_correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

# 6. Extra graph - Goals vs Pre-match Average Goals
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Team_Avg_Goals_Scored", y="Goals",
                hue="Is_Knockout", palette="Dark2", s=60)
plt.title("Actual Goals vs Pre-match Average Goals Scored")
plt.xlabel("Pre-match Avg Goals Scored")
plt.ylabel("Actual Goals in Match")
plt.legend(title="Knockout")
plt.tight_layout()
plt.savefig("06_actual_vs_prematch_goals.png", dpi=300, bbox_inches="tight")
plt.close()

print("All graphs saved successfully.")
print("Files created:")
print("  - lr2_2_cleaned.csv")
print("  - 01_goals_distribution.png")
print("  - 02_goals_vs_rank.png")
print("  - 03_goals_by_stage.png")
print("  - 04_avg_goals_host.png")
print("  - 05_correlation_heatmap.png")
print("  - 06_actual_vs_prematch_goals.png")