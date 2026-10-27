# 🎮 Game Completion Time Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org)

An interactive Machine Learning and Statistical Inference web application predicting video game completion times across three distinct playstyles (**Main Story**, **Main + Sides**, and **100% Completionist**), trained on **HowLongToBeat** dataset records.

---

## 🌟 Key Features

### 1. 🔮 Multi-Output Time Prediction
* **Game Search & Comparison:** Instant lookup of 7,000+ games with side-by-side comparison between **Predicted** and **Actual (HLTB)** hours and prediction error ($\Delta$).
* **Custom Game Builder:** Configure hypothetical games with custom genres, platforms, release year, and developer experience to simulate completion times.
* **Lab Curriculum ML Models:** Select between **Ridge Regression**, **Decision Trees**, **K-Nearest Neighbors (KNN)**, **Random Forest Regressor**, and **Support Vector Regressor (SVR)**.

### 2. ⚖️ "Completionist Grind Ratio" & Pacing Analysis *(New)*
* Calculates the **Grind Ratio** $\left(\frac{\text{Completionist Time}}{\text{Main Story Time}}\right)$ to quantify game bloat and filler content.
* Automatic **Pacing Classification**:
  * 🎯 **Tight & Story-Driven** ($< 1.6\times$): Minimal side distractions; focused narrative.
  * ⚖️ **Balanced Adventure** ($1.6\times - 2.5\times$): Organic mix of story and rewarding side quests.
  * 🗺️ **Expansive World** ($2.5\times - 4.0\times$): Sprawling exploration and massive side quest volume.
  * ⚔️ **Completionist Grindfest** ($\ge 4.0\times$): Heavy collectible hunts and repetitive endgame grind.
* **Playstyle Breakdown Bar:** Visual percentage breakdown of time spent on Story vs. Side Content vs. 100% Collectible Grind.

### 3. ⚔️ Side-by-Side Game Versus Mode *(New)*
* Pit two games head-to-head in a comparative matchup.
* **Quick Presets:** Instant comparison for iconic matchups (*The Witcher 3 vs. Cyberpunk 2077*, *Elden Ring vs. Dark Souls*, *Skyrim vs. Fallout 4*, etc.).
* **Radar & Grouped Bar Charts:** Compare relative playstyle profiles, narrative length, side quest density, and completionist overhead.
* **Automated Matchup Verdict:** Narrative synthesis explaining which game demands more story time vs. which has more side bloat.

### 4. 📅 Backlog Completion Planner *(New)*
* Turn unplayed backlog games into an actionable schedule.
* **Custom Play Habits:** Configure daily playtime or split between weekdays and weekends, plus rest days.
* **Live Burn-Down Curve:** Visual Plotly area chart showing remaining hours declining to zero over calendar days.
* **Deadline Checker:** Set target deadlines (vacations, upcoming releases) to see if you are on track or how many hours/day are needed to finish on time.
* **Milestone Roadmap:** Projected calendar dates for 25%, 50% (halfway point), 75%, and 100% completion.

### 5. 📊 Exploratory Data Analysis & Statistical Inference
* **Central Limit Theorem (CLT) Demo:** Simulates sampling distributions with varying sample sizes ($N=10, 30, 100$), comparing sample standard error with theoretical $\frac{\sigma}{\sqrt{n}}$.
* **Hypothesis Testing:** Two-sample Welch's $t$-test evaluating whether modern games ($\ge 2015$) are statistically longer than classic games.
* **Confidence Intervals & Correlation:** Pearson & Spearman correlation with release year ($\alpha = 0.05$) and 95% Confidence Intervals for median completion times.

### 6. 🔍 Interactive Dataset Explorer
* Filter and search across 7,000+ cleaned games by title, genre, and release year.
* Leaderboards for **Longest Games**, **Shortest Games**, **Highest Grind Ratios (Most Bloat)**, and **Lowest Grind Ratios (Most Focused)**.

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/krrishparmar/Game-Completion-Time-Predictor.git
   cd Game-Completion-Time-Predictor
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

---

## 📦 Project Structure

```
├── app.py                               # Main Streamlit web application
├── game_completion_time_regression.ipynb # Notebook for regression experiments & statistical tests
├── hltb_dataset.csv                     # HowLongToBeat dataset
├── requirements.txt                     # Project dependencies
└── README.md                            # Documentation
```

---

## 📝 License
Built for educational and research purposes using public data from [HowLongToBeat](https://howlongtobeat.com/).