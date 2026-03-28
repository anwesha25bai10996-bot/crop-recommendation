import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Dataset ────────────────────────────────────────────────────────────────────
profiles = {
    "Rice":       ([80,120],[40,60],[35,55],[20,28],[80,90],[5.5,7.0],[150,300]),
    "Maize":      ([60,110],[35,55],[35,55],[18,27],[55,75],[5.5,7.5],[60,110]),
    "Chickpea":   ([20,50],[45,95],[65,95],[15,24],[15,30],[6.0,9.0],[60,100]),
    "Cotton":     ([100,140],[35,55],[15,25],[23,28],[75,85],[6.0,7.0],[80,120]),
    "Jute":       ([60,80],[40,60],[35,55],[23,28],[70,80],[6.0,7.0],[150,200]),
    "Mango":      ([15,25],[10,20],[25,45],[25,38],[45,55],[4.5,7.0],[90,110]),
    "Banana":     ([80,120],[60,90],[40,60],[25,35],[70,90],[5.5,7.0],[100,200]),
    "Coffee":     ([0,20],[25,35],[25,35],[22,28],[75,85],[6.0,7.0],[150,200]),
    "Watermelon": ([80,120],[10,20],[40,60],[24,30],[80,90],[6.0,7.0],[40,60]),
    "Grapes":     ([15,25],[10,20],[200,240],[8,18],[80,82],[5.5,6.5],[60,70]),
    "Apple":      ([0,20],[120,145],[195,210],[20,24],[90,95],[5.5,6.5],[100,125]),
    "Orange":     ([0,20],[5,15],[5,15],[10,20],[90,95],[6.0,7.5],[100,120]),
    "Coconut":    ([0,20],[0,10],[30,45],[25,32],[90,95],[5.0,8.0],[150,200]),
    "Lentil":     ([18,30],[60,90],[60,90],[15,24],[60,70],[6.0,8.0],[35,55]),
    "Blackgram":  ([30,60],[40,80],[30,70],[25,35],[60,80],[6.0,7.5],[60,100]),
}

np.random.seed(42)
rows = []
for crop, p in profiles.items():
    for _ in range(80):
        rows.append({
            "N":           np.random.uniform(*p[0]),
            "P":           np.random.uniform(*p[1]),
            "K":           np.random.uniform(*p[2]),
            "temperature": np.random.uniform(*p[3]),
            "humidity":    np.random.uniform(*p[4]),
            "ph":          np.random.uniform(*p[5]),
            "rainfall":    np.random.uniform(*p[6]),
            "label":       crop
        })

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)

# ── Train ──────────────────────────────────────────────────────────────────────
features = ["N","P","K","temperature","humidity","ph","rainfall"]
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df["label"], test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(max_depth=10, random_state=42)
clf.fit(X_train, y_train)
acc = accuracy_score(y_test, clf.predict(X_test))
print(f"Model trained! Accuracy: {acc*100:.1f}%")

# ── Charts ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Crop Recommendation System", fontsize=13, fontweight="bold")
counts = df["label"].value_counts()
axes[0].barh(counts.index, counts.values, color="#1D9E75")
axes[0].set_title("Crops in Dataset")
axes[0].set_xlabel("Samples")
axes[0].invert_yaxis()
axes[1].bar(features, clf.feature_importances_, color="#534AB7")
axes[1].set_title(f"Feature Importance (Accuracy: {acc*100:.1f}%)")
axes[1].set_ylabel("Importance")
axes[1].tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig("crop_analysis.png")
plt.close()
print("Chart saved as crop_analysis.png")

# ── Input validation ───────────────────────────────────────────────────────────
def get_input(prompt, lo, hi):
    while True:
        try:
            val = float(input(f"  {prompt} ({lo}-{hi}): "))
            if lo <= val <= hi:
                return val
            else:
                print(f"  Invalid! Value must be between {lo} and {hi}. Try again.")
        except ValueError:
            print("  Please enter a valid number.")

# ── Recommend ──────────────────────────────────────────────────────────────────
print("\n" + "="*45)
print("   CROP RECOMMENDATION SYSTEM")
print("="*45)
print("  Enter your soil and climate details:\n")

N           = get_input("Nitrogen (N)",   0,  200)
P           = get_input("Phosphorous (P)",0,  200)
K           = get_input("Potassium (K)",  0,  300)
temperature = get_input("Temperature C",  0,   45)
humidity    = get_input("Humidity %",     0,  100)
ph          = get_input("Soil pH",        0,   14)
rainfall    = get_input("Rainfall mm",    0,  400)

inp = pd.DataFrame([{"N":N,"P":P,"K":K,"temperature":temperature,
                      "humidity":humidity,"ph":ph,"rainfall":rainfall}])
proba = clf.predict_proba(inp)[0]
top3  = np.argsort(proba)[-3:][::-1]

print("\n" + "="*45)
print("  RESULTS")
print("="*45)
for i, idx in enumerate(top3, 1):
    bar = "█" * int(proba[idx] * 30)
    print(f"  #{i} {clf.classes_[idx]:<15} {proba[idx]*100:5.1f}%  {bar}")
print(f"\n  Best crop for your conditions: {clf.classes_[top3[0]]}")
print("="*45)
print("\n  Note: Consult a local agronomist for field confirmation.")


