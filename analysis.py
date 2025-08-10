import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the CSV
file_path = "data/monthly_report.csv"
df = pd.read_csv(file_path)

# 2. Ensure columns exist
required_cols = ["Vertical", "Month", "NPS %", "Course Completion %", "Placement Count",
                 "Registrations", "Average Mentor Rating"]

missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in CSV: {missing_cols}")

# 3. Normalize NPS, Course Completion, Placement Count for Vertical Performance Health
df["Norm NPS"] = df["NPS %"] / df["NPS %"].max() * 100
df["Norm Completion"] = df["Course Completion %"] / df["Course Completion %"].max() * 100
df["Norm Placement"] = df["Placement Count"] / df["Placement Count"].max() * 100

# 4. Calculate Vertical Performance Health
df["Vertical Performance Health"] = df[["Norm NPS", "Norm Completion", "Norm Placement"]].mean(axis=1)

# 5. Calculate extra metrics (not included in health score)
df["Registration to Placement %"] = (df["Placement Count"] / df["Registrations"]) * 100

# 6. Categorize health zones
def health_zone(score):
    if score >= 75:
        return "Healthy"
    elif score >= 50:
        return "Watch"
    else:
        return "Red"

df["Health Zone"] = df["Vertical Performance Health"].apply(health_zone)

# 7. Save processed file
df.to_csv("data/vertical_performance_results.csv", index=False)

# 8. Plot trends for each vertical
plt.figure(figsize=(10,6))
sns.lineplot(data=df, x="Month", y="Vertical Performance Health", hue="Vertical", marker="o")
plt.xticks(rotation=45)
plt.title("Vertical Performance Health Trend")
plt.tight_layout()
plt.savefig("data/vertical_performance_trend.png")

# 9. Summary table
summary = df.groupby("Vertical")[["NPS %", "Course Completion %", "Placement Count",
                                  "Vertical Performance Health", "Registration to Placement %",
                                  "Average Mentor Rating"]].mean().reset_index()

print(summary)
