import streamlit as st
import streamlit.components.v1 as stc

#load libraries

#importing the basic libraries
import numpy as np
import pandas as pd
import ast



 #importing the regex library
import re

#importing the nltk library
import nltk


#lemmatizer
from nltk.stem import WordNetLemmatizer

#  nltk stopwords
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords

#defining the stop words
ENGLISH_STOP_WORDS = stopwords.words('english')



#library for word embedding
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity


#funkSVD libraries
from surprise import Dataset
from surprise.reader import Reader
from surprise.prediction_algorithms.matrix_factorization import SVD as FunkSVD
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV


#load data

@st.cache_data
def load_data():


    recipes = pd.read_csv('../Docs/Datasets/recipes_streamlit.csv')
    return recipes

@st.cache_data
def load_reviews():

    ratings= pd.read_csv('../Docs/Datasets/reviews.csv')
    ratings_ = ratings.sort_values(by=['AuthorId', 'RecipeId'])

    return ratings_





#parser ingredients input.

@st.cache_data
def parser(input_keys):

    remove_= { 'oil', 'salt', 'pepper'}

    #defining measuring units

    measuring_words= ['ml', 'mL', 'milliliter', 'millilitre', 'cc' , 'cubic centimeter', 'l', 'L', 'liter', 'litre', 'dl', 'dL', 'deciliter', 'decilitre', 'teaspoon', 't' , 'tsp.','tablespoon' , 'T', 'tbl', 'tbs', 'tbsp', 'fluid ounce', 'fl oz',  'gill', 'cup',  'c', 'pint', 'p', 'pt', 'fl pt','quart', 'q', 'qt', 'fl qt', 'gallon' , 'g' , 'gal' , 'g', 'milligram', 'milligramme', 'g' , 'gram' , 'gramme', 'kg','kilogram', 'kilogramme', 'pound', 'lb', 'ounce', 'oz', 'mm', 'milimeter', 'millimetre', 'cm' , 'centimeter', 'centimetre', 'm' , 'meter','metre', 'inch', 'in', 'yard', '°C' , 'degree celsius','°F' ,'Farenheit', 'tsp']

    ingredients_cleaned = []
    lemmatizer = WordNetLemmatizer()
    
    # for ingredients_list in input_keys:
        
    ingredients_words = re.split(',', input_keys)

    
    for ingredient in ingredients_words:
        items= ingredient.split()
        items = [lemmatizer.lemmatize(word.lower()) for word in items
                    if word.isalpha() and 
                    word.lower() not in ENGLISH_STOP_WORDS and 
                    word.lower() not in measuring_words and 
                    word.lower() not in remove_]
        
        if items:
            ingredients_cleaned.append(' '.join(items))
        #ingredients_cleaned.append(' '.join(items))
    

    return ingredients_cleaned




#parser ingredients
@st.cache_data
def parser_ing(input_keys):

    # Defining measuring units
    remove_ = {'oil', 'salt', 'pepper'}


    # Defining measuring units
    measuring_words = ['ml', 'mL', 'milliliter', 'millilitre', 'cc', 'cubic centimeter', 'l', 'L', 'liter', 'litre', 'dl',
                   'dL', 'deciliter', 'decilitre', 'teaspoon', 't', 'tsp.', 'tablespoon', 'T', 'tbl', 'tbs', 'tbsp',
                   'fluid ounce', 'fl oz', 'gill', 'cup', 'c', 'pint', 'p', 'pt', 'fl pt', 'quart', 'q', 'qt', 'fl qt',
                   'gallon', 'g', 'gal', 'g', 'milligram', 'milligramme', 'g', 'gram', 'gramme', 'kg', 'kilogram',
                   'kilogramme', 'pound', 'lb', 'ounce', 'oz', 'mm', 'milimeter', 'millimetre', 'cm', 'centimeter',
                   'centimetre', 'm', 'meter', 'metre', 'inch', 'in', 'yard', '°C', 'degree celsius', '°F', 'Farenheit', 'tsp']


    cleaned_ingredients_all_recipes = []


    for each_ingredient_list in input_keys:
        comma_list = re.split(',', each_ingredient_list)  # splitting the ingredients by commas


        cleaned_ingredients = []  # new list to store cleaned ingredients for a single recipe


        lemmatizer = WordNetLemmatizer()  # lemmatize


        for each_word_set in comma_list:
            items = [word.lower() for word in re.findall(r'\b\w+\b', each_word_set)]  # Extract individual words and convert to lowercase


            items = [word for word in items if word.isalpha()]  # filtering only letters


            


            items = [lemmatizer.lemmatize(word) for word in items]  # lemmatizing


            items = [word for word in items if word not in ENGLISH_STOP_WORDS]  # removing stop words


            items = [word for word in items if word not in measuring_words]  # removing measuring words


            items = [word for word in items if word not in remove_]


            if items:
                cleaned_ingredients.extend(items)


        cleaned_ingredients_all_recipes.append(cleaned_ingredients)

    return(cleaned_ingredients_all_recipes)





#parser
@st.cache_data
def model():
    
    
    phrases_model= Word2Vec.load('../Models/Word2Vec/phrases_model_new(op1).bin')
        
    return phrases_model #return the list


@st.cache_data
def load_full_train_set():
    import pickle


    ratings_= load_reviews()



    my_dataset = Dataset.load_from_df(ratings_[['AuthorId', 'RecipeId', 'Rating']], Reader(rating_scale=(1, 5)))
    full_train_set= my_dataset.build_full_trainset()

    return full_train_set

def svd_model():

    full_train_set= load_full_train_set()
    final_model= FunkSVD(n_factors=15,
                           n_epochs=6,
                           lr_all=0.005,    # learning rate for each epoch
                           biased=False,  # forces the algorithm to store all latent information in the matrices
                           verbose=0)
    final_model.fit(full_train_set)

    return final_model



#parser recommender
@st.cache_data
def recommender(user_input):

    print("before paser : ",user_input)
    print("type : ",type(user_input))

    #user_input = ','.join(map(str, user_input)) #converting to str, mapping and joining by the ','
    
    user_input.split(',')
    print('after split:', user_input)

    user_input_2=parser(user_input)
    
    print("after paser : ",user_input_2)
    print("type : ",type(user_input_2))



    recipes= load_data()
    
    phrases_model= model()
    print('model')
    
    ingredients= recipes['RecipeIngredientParts']
    print('before parser:', ingredients[0])

    

    
    
    ingredients_cleaned = parser_ing(ingredients)
    #ingredients_cleaned=ingredients.apply(parser_ing)
    #ingredients_cleaned = recipes['ingredients_cleaned']

    print(ingredients_cleaned[0])
    print(type(ingredients_cleaned))


# mean Embeddings for User Input
    user_mean_vector =np.mean([phrases_model.wv[ingredient] for ingredient in user_input_2 if ingredient in phrases_model.wv] or [np.zeros(300)], axis=0)



# check if user_mean_vector contains NaN values
    if np.isnan(user_mean_vector).any():
        print("User input vectors contain NaN values.")

 
    else:

    # mean ingredient vectors for each recipe
        recipe_vectors = [np.mean([phrases_model.wv[sub_ingredient] for sub_ingredient in ingredient if sub_ingredient in phrases_model.wv] or [np.zeros(300)], axis=0) for ingredient in ingredients_cleaned]
        


    # Check if any recipe vector contains NaN values
        if not recipe_vectors:
            print('nothing obtained for recipe vectors')

        else:
        
        # cosine similarity between user mean vector and recipe mean vectors
            cosine_similarities = cosine_similarity([user_mean_vector], recipe_vectors)


        # indices of top N most similar recipes
            top_recipes = np.argsort(cosine_similarities[0])[::-1][:5]



            recommendation= pd.DataFrame( columns=['RecipeId', 'Name','Ingredients', 'Category','Calories', 'Time', 'Score'] ) #dataframe with columns

    
            for i in top_recipes: #defining the data for each recommendation

            
                Recipe_id= recipes['RecipeId'].iloc[i]
                Name = recipes['Name'].iloc[i]
                Calories = '{:.0f}'.format(recipes['Calories'].iloc[i])
                Time= recipes['TotalTime'].iloc[i]
                Category = recipes['RecipeCategory'].iloc[i]
                Ingredients = recipes['RecipeIngredientParts'].iloc[i]
                Score= '{:.0f}'.format(cosine_similarities[0][i])
                #instructions= recipes['RecipeInstructions'].iloc[i]
                recommendation = pd.concat([recommendation,pd.DataFrame(
                                    {'RecipeId' : [Recipe_id],
                                     'Name': [Name], 
                                    'Ingredients': [Ingredients], 
                                    'Category' : [Category],
                                    'Calories': [Calories], 
                                     'Time': [Time],
                                     'Score': [Score]})], 
                                        ignore_index=True)
                
    return recommendation  



# Function to initialize user_ratings_dict

@st.cache_data
def hybrid_recommender(user_id, user_input_ingredients):
        # Initialize user_ratings_dict
    
    def initialize_user_ratings_dict(ratings_):
        user_ratings_dict = {}
        for index, row in ratings_.iterrows():
            user_id = row['AuthorId']
            recipe_id = row['RecipeId']
            rating = row['Rating']
            
            if user_id not in user_ratings_dict:
                user_ratings_dict[user_id] = {}
            user_ratings_dict[user_id][recipe_id] = rating
        return user_ratings_dict
    
    ratings_=load_reviews()
    
    user_ratings_dict = initialize_user_ratings_dict(ratings_)




        # Content-Based Filtering
    
    def content_based_filtering(user_input_ingredients):
        recommended_recipes= recommender(user_input_ingredients)
        return recommended_recipes
    
    content_based_recommendations =content_based_filtering(user_input_ingredients)


    def collaborative_filtering(user_id, recommended_recipes, user_ratings_dict):

        

        full_train_set= load_full_train_set()
        #user_id= full_train_set.to_raw_uid(int(user_id))
        final_model= svd_model()


        # Get ratings for recommended recipes
        recommended_ratings = []
        for RecipeID in recommended_recipes['RecipeId']:
            if RecipeID not in user_ratings_dict.get(user_id, {}):
                predicted_rating = '{:.0f}'.format(final_model.predict(user_id, RecipeID).est)
                recommended_ratings.append((RecipeID, predicted_rating))
        
        # Sort recommended recipes by predicted rating
        recommended_ratings.sort(key=lambda x: x[1], reverse=True)

        recommended_ratings_df = pd.DataFrame(recommended_ratings, columns=['RecipeId', 'PredictedRating'])




        final_recommendations= pd.merge(recommended_recipes,recommended_ratings_df, on='RecipeId')
        
        return final_recommendations




        # Collaborative Filtering (SVD) for predicted ratings
    collaborative_based_recommendations = collaborative_filtering(user_id, content_based_recommendations, user_ratings_dict)




    return collaborative_based_recommendations






#recommending logic

import streamlit as st
import requests
from streamlit_lottie import st_lottie

left_column, right_column = st.columns(2)

with left_column:

    st.title('Flavor Fuze')
    st.markdown('#### Elevate your Culinary experience with Flavor Fuze - where every dish becomes an adventure')
    st.caption('By Anupa Jayakody')

with right_column:

    lottiefile = requests.get('https://lottie.host/dc1b8d65-0ec5-4ca6-9959-c35c8842809b/picTTTEeVt.json', verify=False)
    st_lottie(lottiefile.json(), height=300, width=300)


#st.divider()
st.subheader('Search by Ingredients')



#user input

user_id_input= st.text_input('###### Enter your login id')
user_ingredients= st.text_input('###### Do you have some ingredients and you want to search the recipes that you can make from them??')
 
if st.button('Lets ing-in !!'):


    loading_image= st_lottie((requests.get('https://lottie.host/9e4a2516-2bd2-4f3a-97dc-e13a1c8b17d3/WsU810K1aZ.json',verify=False).json()), height=100, width=100)

    full_train_set= load_full_train_set()

    
    
    #user_inner_id= full_train_set.to_inner_uid(str(user_id_input))
    user_id= full_train_set.to_raw_uid(int(user_id_input))
    
    
    recommend_recipes= hybrid_recommender(user_id, user_ingredients)
    
    if recommend_recipes is not None and not recommend_recipes.empty:
        
        st.table(recommend_recipes)
    
    else:

        st.write('I am sorry, I couldnt find you a good recipe today')
