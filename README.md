# 🏦 Bank Marketing Dataset – EDA & Tree Classifier

## 🧠 Data Science Task 3 – SkillCraft Technology

This project is part of my Data Science journey under **SkillCraft Technology**. The focus of this task is to perform **Exploratory Data Analysis (EDA)** and apply a **Decision Tree Classifier** on the **Bank Marketing Dataset** to uncover customer behavior patterns and predict term deposit subscriptions.

---

## 🎯 Objective

To analyze the telemarketing campaign data of a Portuguese bank and:
- Uncover patterns in customer demographics and campaign effectiveness
- Visualize key categorical and numerical features
- Train a **Decision Tree Classifier** to predict the likelihood of subscription (`y`)

---

## 📂 Dataset Details

- **Dataset Name**: `bank.csv`
- **Source**: [UCI Machine Learning Repository – Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- **Target Variable**: `y` (Yes/No – whether the client subscribed)
- **Features**:
  - Demographic: `age`, `job`, `marital`, `education`
  - Financial: `balance`, `default`, `housing`, `loan`
  - Campaign: `contact`, `duration`, `campaign`, `pdays`, `previous`, `poutcome`

---

## 🛠️ Tools & Libraries Used

- **Python**
- **Pandas**, **NumPy** – Data handling
- **Matplotlib**, **Seaborn** – Visualizations
- **Scikit-learn** – Machine Learning (DecisionTreeClassifier)


---

## 📊 Analysis Performed

- Data inspection and cleaning
- Value counts and distributions of categorical features
- Correlation heatmaps and feature importance
- Visual analysis using barplots, histograms, and pie charts

---

## 🌳 Decision Tree Classifier

- Applied preprocessing (label encoding, feature selection)
- Split the data into training and test sets
- Trained a **DecisionTreeClassifier**
- Evaluated using:
  - **Accuracy**
  - **Confusion Matrix**
  - **Classification Report**
