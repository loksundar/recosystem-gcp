# 🛍️ Product Recommendation System with Sentiment-Enhanced Ranking

A hybrid recommendation system that combines **collaborative filtering** with **sentiment analysis** to provide top product recommendations based on user ratings and positive feedback. Deployed with a web interface at [loksundar.com](https://loksundar.com), this system showcases a full-stack ML project from data processing to user-facing prediction.

---

## 👤 Target Users
- **Data Scientists** – Design recommendation and sentiment scoring pipelines.
- **ML Engineers** – Serve pickled models, deploy Flask APIs.
- **GenAI Engineers** – Replace sentiment model with LLM classifiers.
- **Data Engineers** – Handle user/item matrix creation, TF-IDF pipelines.

---

## 🎯 Problem Statement

Traditional recommender systems may suggest products highly rated by users, but not necessarily *liked*. This system improves personalization by factoring in **review sentiment**, ensuring that top-rated products also align with user satisfaction.

---

## 🔧 Tech Stack

| Component         | Tools/Frameworks                              |
|------------------|------------------------------------------------|
| Backend           | Python, Flask                                 |
| Model Persistence | Pickle (Sentiment Classifier, Vectorizer, Matrix) |
| Recommendation    | User-Item Matrix (Collaborative Filtering)    |
| Sentiment Engine  | TF-IDF + ML Classifier (Logistic Regression)  |
| UI/UX             | HTML, Jinja2 Templates                        |
| Hosting           | Custom deployment at [loksundar.com](https://loksundar.com) |

---

## 🔁 How It Works

### Prediction 1 – Name-Based Recommendation
1. User enters a known username.
2. System retrieves the top 20 product predictions based on user-item scores.
3. Sentiment model filters top 5 based on highest positive feedback %.

### Prediction 2 – Rating-Based Cold Start
1. User rates 7 seed products.
2. System updates a test user in the matrix and fills with random values.
3. Sentiment scoring determines final top 5 suggestions.

---

## 📁 Project Structure

```bash
├── app.py # Flask web app
├── templates/
│ ├── index.html # UI for name-based recs
│ └── index2.html # UI for cold-start recs
├── user_final_rating.pkl # User-item prediction matrix
├── sent_df.pkl # Product review data
├── Tfidf_vectorizer.pkl # Vectorizer for review text
├── Finalized_Model.pkl # Sentiment classification model
├── mapping.pkl # Product ID to name mapping
└── static/ # (Optional) CSS or JS assets
```
---

---

## ✅ Sample Output

### User: `Test123`
Top 5 recommendation Products according to your preference are:

Dermalogica Special Cleansing Gel

Tim Holtz Retractable Craft Pick

Alberto VO5 Salon Series Shampoo

Fiskars Classic Stick Rotary Cutter

Cantu Coconut Milk Shine Hold Mist

---

## 🚀 Running Locally

```bash
# 1. Clone the repository
git clone https://github.com/loksundar/product-recommendation-app.git
cd product-recommendation-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Flask app
python app.py

```
---

## 🚀 Running Locally

```bash
# 1. Clone the repository
git clone https://github.com/loksundar/product-recommendation-app.git
cd product-recommendation-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Flask app
python app.py
```
Access it at: http://127.0.0.1:8080
## 🧠 Future Enhancements
Replace ML sentiment classifier with LLM-based sentiment analysis.

Add user feedback loop to improve recommendations over time.

Integrate with Firebase/Auth0 for real-time user identity management.

Expand product catalog via e-commerce APIs.

