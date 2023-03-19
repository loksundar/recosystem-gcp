from flask import Flask, render_template, request
import pickle
from memory_profiler import profile
import pandas as pd
import numpy as np
import random
import os
import gc

# contains one ML model and only one recommendation system that we have obtained from the

app = Flask(__name__)
@profile
def predict1(user_name=""):
    list_data = []
    text_info = ""
    path = "./"
    user_final_rating = pd.read_pickle(path+"user_final_rating.pkl")
    if user_name not in user_final_rating.index:
        text_info = "Invalid user! please enter valid user name."
        del user_final_rating
        gc.collect()
        return text_info,[]
    else:
        top20_recommended_products = list(user_final_rating.loc[user_name].sort_values(ascending=False)[0:20].index)
        text_info = "Top 5 Recommended products for \"" + user_name +  "\""
        del user_final_rating
        gc.collect()
        return predict(top20_recommended_products,text_info)

def predict2(ls):
    path = "./"
    user_final_rating = pd.read_pickle(path+"user_final_rating.pkl")
    u_name = "Test123"
    ids = ["Barielle Nail Rebuilding Protein","Cantu Coconut Milk Shine Hold Mist - 8oz","Fiskars174 Classic Stick Rotary Cutter (45 Mm)","Dermalogica Special Cleansing Gel, 8.4oz","Voortman Sugar Free Fudge Chocolate Chip Cookies","Tim Holtz Retractable Craft Pick-Red 6x.5","Alberto VO5 Salon Series Smooth Plus Sleek Shampoo"]
    ls = [1 if i==min(ls) else (i-min(ls))*5/(max(ls)-min(ls)) for i in ls]
    user_final_rating.loc[u_name,ids] = ls
    ids = random.choices(range(0,50),k=20)
    user_final_rating.loc[u_name].iloc[ids] = [random.randint(1,5) for i in range(20)]
    user_final_rating.fillna(0,inplace=True)
    top20_recommended_products = list(user_final_rating.loc[u_name].sort_values(ascending=False)[0:20].index)
    text_info = "Top 5 recommendation Products according to your preference are "
    del user_final_rating
    gc.collect()
    return predict(top20_recommended_products,text_info)

def predict(top20_recommended_products,text_info):
    path = "./"
    sent_df = pd.read_pickle(path+"sent_df.pkl")
    df_top20_products = sent_df[sent_df.id.isin(top20_recommended_products)]
    # For these 20 products, get their user reviews and pass them through TF-IDF vectorizer to convert the data into suitable format for modeling
    del sent_df
    gc.collect()
    with open(path+"Tfidf_vectorizer.pkl", 'rb') as file:
        tfidf = pickle.load(file)
    X = tfidf.transform(df_top20_products["reviews"].values.astype(str))
    del tfidf
    gc.collect()
    # Use the best sentiment model to predict the sentiment for these user reviews
    with open(path+"Finalized_Model.pkl", 'rb') as file:
        model = pickle.load(file)
    df_top20_products['predicted_sentiment'] = model.predict(X)
    del model
    del X
    gc.collect()
    # Create a new dataframe "pred_df" to store the count of positive user sentiments
    df_top20_products = df_top20_products[["id","predicted_sentiment"]]
    mapping = pd.read_pickle(path+"mapping.pkl")
    df_top20_products.drop_duplicates(inplace=True)
    df_top20_products = pd.merge(mapping,df_top20_products,on="id",how = "inner")
    del mapping
    gc.collect()
    df_top20_products =  df_top20_products[["name","predicted_sentiment"]]
    pred_df = df_top20_products.groupby(by='name').sum()
    # Create a column to measure the total sentiment count
    pred_df['total_count'] = df_top20_products.groupby(by='name')['predicted_sentiment'].count()
    del df_top20_products
    gc.collect()
    # Create a column that measures the % of positive user sentiment for each product review
    pred_df['post_percentage'] = np.round(pred_df['predicted_sentiment']/pred_df['total_count']*100,2)
    # Return top 5 recommended products to the user
    result = pred_df.sort_values(by='post_percentage', ascending=False)[:5]
    del pred_df
    gc.collect()
    list_data = list(result.index)
        
    return text_info, list_data

# This is the Flask interface file to connect the backend ML models with the frontend HTML code

@app.route('/', methods=['POST', 'GET'])
@profile
def index():
    if request.method == 'POST':
        user_name = request.form['uname']
        text_info = "Please enter the user name."
        if len(user_name)<1:
            return render_template('index.html', info=text_info, data=[])
        else:
            text_info, data_list = predict1(user_name = user_name)
            return render_template('index.html', info=text_info, data=data_list)  
    else:
        return render_template('index.html')
@app.route('/index2', methods=['POST', 'GET'])
@profile
def index2():
    if request.method == 'POST':
        data_list = []
        r1 = int(request.form['r1'])
        r2 = int(request.form['r2'])
        r3 = int(request.form['r3'])
        r4 = int(request.form['r4'])
        r5 = int(request.form['r5'])
        r6 = int(request.form['r6'])
        r7 = int(request.form['r7'])
        ls = [r1,r2,r3,r4,r5,r6,r7]
        text_info, data_list = predict2(ls)
        return render_template('index2.html', info=text_info, data=data_list)  
    else:
        return render_template('index2.html')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))