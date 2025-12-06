# Employee-Sentiment-Analysis# 

## Overview

This project performs a complete **Employee Sentiment Analysis** using NLP techniques, time‑based sentiment tracking, employee ranking, and flight‑risk detection. It includes preprocessing, sentiment extraction using **TextBlob and BERT**, extensive EDA, and basic predictive modeling.

---

## Project Workflow

### **1. Data Loading & Initial Exploration**

* Read CSV data into a pandas DataFrame.
* Inspected dataset structure using `info()` and `describe()`.
* Identified the number of unique employees.
* Renamed columns for clarity.

---

## **2. Text Preprocessing**

A custom `clean_msg()` function was used to clean text. Steps included:

* Lowercasing text
* Removing URLs, mentions, hashtags
* Removing emojis and non‑ASCII characters
* Removing numbers and special symbols
* Removing "AM/PM"
* Removing unwanted spaces

Output stored in a new column: **clean_body**

---

## **3. Sentiment Extraction**

### **Using TextBlob:**

* Computed **Polarity** (−1 to +1)
* Computed **Subjectivity** (0 to 1)
* Classified sentiment into **Positive / Neutral / Negative**

### **Using BERT (DistilBERT SST‑2):**

* Used HuggingFace pipeline for deep learning‑based sentiment scoring.
* Converted model output into numerical polarity.
* Repeated monthly feature aggregation.

---

## **4. Time‑Based Aggregation**

* Added `year_month` column.
* Created pivot table of **monthly average polarity per employee**.
* Used heatmaps and bar charts to visualize sentiment patterns.

---

## **5. EDA & Visualizations**

Key plots included:

* Heatmap of employee × month average polarity
* Monthly average sentiment trend chart
* Correlation heatmaps for numeric features

### **Insights:**

* Some months show more negative or positive sentiment.
* Many employees have neutral or stable sentiment.
* Message length, frequency, and word count show **weak correlation** with sentiment.
* Numeric features alone cannot predict sentiment effectively.

---

## **6. Employee Ranking**

For each month:

* Sorted employees by monthly polarity.
* Extracted **Top 3 Positive** and **Bottom 3 Negative** performers.
* Created a consolidated table of all rankings.

Insight:

* Certain employees consistently appear in top or bottom sentiment groups.

---

## **7. Flight Risk Detection**

A rule‑based system:

* Flag an employee if they have **≥4 negative messages in any 30‑day rolling window**.
* Used date‑sorted negative message groups.
* Built `flight_risk_df` summarizing negative message windows.

**Result:** All 10 employees were flagged at some point.

---

## **8. Predictive Modeling Setup**

Created numeric features:

* Average polarity
* Average message length
* Average word count
* Message frequency

Two correlation heatmaps (TextBlob & BERT) showed:

* **Very weak relationships** → numeric‑only models are ineffective.
* Suggests the need for advanced NLP models (Transformers).

---

## **Conclusion**

* Sentiment varies widely between employees and across months.
* Text‑derived numeric features have **weak correlation** with sentiment.
* Linear regression or numeric‑based models perform poorly.
* BERT improves sentiment detection but still does not correlate strongly with numeric features.
* Flight‑risk detection reveals periods of concentrated negativity.
* Deep NLP models + temporal behavior analysis provide the most reliable insights.

---

## **Tech Stack**

* **Python**, **Pandas**, **NumPy**
* **NLTK**, **TextBlob**
* **HuggingFace Transformers (DistilBERT)**
* **Matplotlib**, **Seaborn**, **Plotly**
* **WordCloud**, **Regex**

