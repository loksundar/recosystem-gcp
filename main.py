from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import os
# contains one ML model and only one recommendation system that we have obtained from the

app = Flask(__name__)

def predict(user_name,ls):
    '''
    Predicting the top recommended products using best ML models
    '''
    list_data = []
    text_info = ""
    
    # load all input files
    path = "./"
    model_path = path+'Finalized_Model'+".pkl"
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    sent_df = pd.read_pickle(path+"sent_df.pkl")

    ratings_df = pd.read_pickle(path+"ratings_df.pkl")

    mapping = pd.read_pickle(path+"mapping.pkl")

    user_final_rating = pd.read_pickle(path+"user_final_rating.pkl")

    pkl_filename = path+"Tfidf_vectorizer.pkl"
    with open(pkl_filename, 'rb') as file:
        tfidf = pickle.load(file)
    
    if user_name not in user_final_rating.index:
        temp = user_final_rating.copy()
        u_name = "Test123"
        ids = ["Barielle Nail Rebuilding Protein","Cantu Coconut Milk Shine Hold Mist - 8oz","Fiskars174 Classic Stick Rotary Cutter (45 Mm)","Dermalogica Special Cleansing Gel, 8.4oz","Voortman Sugar Free Fudge Chocolate Chip Cookies","Tim Holtz Retractable Craft Pick-Red 6x.5","Alberto VO5 Salon Series Smooth Plus Sleek Shampoo"]
        ls = [1 if i==min(ls) else (i-min(ls))*5/(max(ls)-min(ls)) for i in ls]
        temp.loc[u_name,ids] = ls
        temp.fillna(0,inplace=True)
        top20_recommended_products = list(temp.loc[u_name].sort_values(ascending=False)[0:20].index)
        text_info = "Given user name doent exists, Top 5 Recommended products according to your Preferences "
    else:
        # Get top 20 recommended products from the best recommendation model
        top20_recommended_products = list(user_final_rating.loc[user_name].sort_values(ascending=False)[0:20].index)
        text_info = "Top 5 Recommended products for \"" + user_name +  "\""
        # Get only the recommended products from the prepared dataframe "df_sent"
    df_top20_products = sent_df[sent_df.id.isin(top20_recommended_products)]
    # For these 20 products, get their user reviews and pass them through TF-IDF vectorizer to convert the data into suitable format for modeling
    X = tfidf.transform(df_top20_products["reviews"].values.astype(str))
    # Use the best sentiment model to predict the sentiment for these user reviews
    df_top20_products['predicted_sentiment'] = model.predict(X)
    # Create a new dataframe "pred_df" to store the count of positive user sentiments
    df_top20_products = df_top20_products[["id","predicted_sentiment"]]
    df_top20_products = pd.merge(mapping,df_top20_products,on="id",how = "inner")
    df_top20_products =  df_top20_products[["name","predicted_sentiment"]]
    pred_df = df_top20_products.groupby(by='name').sum()
    # Create a column to measure the total sentiment count
    pred_df['total_count'] = df_top20_products.groupby(by='name')['predicted_sentiment'].count()
    # Create a column that measures the % of positive user sentiment for each product review
    pred_df['post_percentage'] = np.round(pred_df['predicted_sentiment']/pred_df['total_count']*100,2)
    # Return top 5 recommended products to the user
    result = pred_df.sort_values(by='post_percentage', ascending=False)[:5]
    list_data = list(result.index)
        
    return text_info, list_data

# This is the Flask interface file to connect the backend ML models with the frontend HTML code
@app.route('/', methods=['POST', 'GET'])
def get_recommendations():

    if request.method == 'POST':
        user_name = request.form['uname']
        data_list = []
        title=['Product']
        text_info = "Invalid user! please enter valid user name."
        r1 = int(request.form['r1'])
        r2 = int(request.form['r2'])
        r3 = int(request.form['r3'])
        r4 = int(request.form['r4'])
        r5 = int(request.form['r5'])
        r6 = int(request.form['r6'])
        r7 = int(request.form['r7'])
        ls = [r1,r2,r3,r4,r5,r6,r7]
        if len(user_name) > 0:
            text_info, data_list = predict(user_name,[0,0,0,0,0,0,0]) 
        else:
            text_info, data_list = predict(user_name,ls)
        return render_template('index.html', info=text_info, data=data_list)  
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))