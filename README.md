# Crop Recommendation System 🌾

A machine learning tool that recommends the best crop to grow based on soil and climate conditions.

## What It Does
A farmer enters 7 values about their soil and climate — the system recommends the top 3 best crops to grow with confidence percentages.

## How to Run

**Option 1 — Command Line (Recommended)**
1. Install dependencies: `pip install pandas numpy scikit-learn matplotlib`
2. Run: `python crop_recommendation.py`
3. Enter your soil details when prompted

**Option 2 — Google Colab**
1. Open [Google Colab](https://colab.research.google.com)
2. Upload `crop_recommendation.py`
3. Run the file
4. Note: Change input values directly in the code when using Colab

## Inputs
| Input | Range |
|---|---|
| Nitrogen (N) | 0 – 200 |
| Phosphorous (P) | 0 – 200 |
| Potassium (K) | 0 – 300 |
| Temperature (°C) | 0 – 45 |
| Humidity (%) | 0 – 100 |
| Soil pH | 0 – 14 |
| Rainfall (mm) | 0 – 400 |

## Example Output
```
#1 Rice            87.3%  ██████████████████████████
#2 Jute             9.1%  ███
#3 Blackgram        3.6%  █
✅ Best crop: Rice
```

## Concepts Applied
| Concept | Usage |
|---|---|
| Classification | Decision Tree predicts best crop |
| Supervised Learning | Trained on labeled soil/climate data |
| Probability | Confidence scores using predict_proba() |
| Input Validation | Rejects out-of-range values |

## Author
Anwesha Dhote — B.Tech CSE AI ML 25BAI10996
Course: Fundamentals of AI and ML, 2026
