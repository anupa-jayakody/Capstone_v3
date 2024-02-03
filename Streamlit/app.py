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




#load data

@st.cache_data
def load_data():


    #recipefile = requests.get('https://pythonstuff.blob.core.windows.net/capstone-files/recipes_streamlit.csv')
    #recipes = pd.read_csv(recipefile)
    recipes = pd.read_csv('C:/Users/e312995/OneDrive - WESCO DISTRIBUTION/Documents/PERSONAL/BRAINSTATION/CAPSTONE/Datasets/recipes_streamlit.csv')
    return recipes


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
    
    # op1bin = requests.get('https://pythonstuff.blob.core.windows.net/capstone-files/phrases_model_new(op1).bin')
    # phrases_model= Word2Vec.load(op1bin)
    phrases_model= Word2Vec.load('C:/Users/e312995/OneDrive - WESCO DISTRIBUTION/Documents/PERSONAL/BRAINSTATION/CAPSTONE-GIT/CAPSTONE_NEW/Models/phrases_model_new(op1).bin')
        
    return phrases_model #return the list


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
    print(ingredients_cleaned[0])


# mean Embeddings for User Input
    user_mean_vector =np.mean([phrases_model.wv[ingredient] for ingredient in user_input_2 if ingredient in phrases_model.wv] or [np.zeros(200)], axis=0)



# check if user_mean_vector contains NaN values
    if np.isnan(user_mean_vector).any():
        print("User input vectors contain NaN values.")

 
    else:

    # mean ingredient vectors for each recipe
        recipe_vectors = [np.mean([phrases_model.wv[sub_ingredient] for sub_ingredient in ingredient if sub_ingredient in phrases_model.wv] or [np.zeros(200)], axis=0) for ingredient in ingredients_cleaned]





    # Check if any recipe vector contains NaN values
        if not recipe_vectors:
            print('nothing obtained for recipe vectors')

        else:
        
        # cosine similarity between user mean vector and recipe mean vectors
            cosine_similarities = cosine_similarity([user_mean_vector], recipe_vectors)


        # indices of top N most similar recipes
            top_recipes = np.argsort(cosine_similarities[0])[::-1][:5]



            recommendation= pd.DataFrame( columns=['Name','Ingredients', 'Category','Calories', 'Time', 'Score'] ) #dataframe with columns

    
            for i in top_recipes: #defining the data for each recommendation

            
                Name = recipes['Name'].iloc[i]
                Calories = '{:.0f}'.format(recipes['Calories'].iloc[i])
                Time= recipes['TotalTime'].iloc[i]
                Category = recipes['RecipeCategory'].iloc[i]
                Ingredients = recipes['RecipeIngredientParts'].iloc[i]
                Score= '{:.0f}'.format(cosine_similarities[0][i])
                #instructions= recipes['RecipeInstructions'].iloc[i]
                recommendation = pd.concat([recommendation,pd.DataFrame(
                                    {'Name': [Name], 
                                    'Ingredients': [Ingredients], 
                                    'Category' : [Category],
                                    'Calories': [Calories], 
                                     'Time': [Time],
                                     'Score': [Score]})], 
                                        ignore_index=True)
                
    return recommendation  




    
    

#recommending logic

import streamlit as st
import requests
from streamlit_lottie import st_lottie

left_column, right_column = st.columns(2)

with left_column:

    st.title('Flavor Fuze')
    st.markdown('#### Elevate your Culinary experience with Flavor Fuze - where every dish becomes and adventure')
    st.caption('By Anupa Jayakody')

with right_column:

    lottiefile = requests.get('https://lottie.host/dc1b8d65-0ec5-4ca6-9959-c35c8842809b/picTTTEeVt.json', verify=False)
    st_lottie(lottiefile.json(), height=300, width=300)


st.divider()
st.subheader('Search by Ingredients')



#user input

user_ingredients= st.text_input('Do you have some ingredients and you want to search the recipes that you can make them??')

if st.button('Lets ing-in !!'):


    loading_image= st_lottie((requests.get('https://lottie.host/9e4a2516-2bd2-4f3a-97dc-e13a1c8b17d3/WsU810K1aZ.json',verify=False).json()), height=100, width=100)


    recommend_recipes= recommender(user_ingredients)
    
    if recommend_recipes is not None and not recommend_recipes.empty:
        
        st.table(recommend_recipes)
    
    else:

        st.write('no recommendations today')






### KEYWORD MODEL
        

#load data
@st.cache_data
def load_model_text():

    # op2bin = requests.get('https://pythonstuff.blob.core.windows.net/capstone-files/phrases_model_new(op2).bin')
    # phrases_model= Word2Vec.load(op2bin)
    phrases_model= Word2Vec.load('C:/Users/e312995/OneDrive - WESCO DISTRIBUTION/Documents/PERSONAL/BRAINSTATION//CAPSTONE/Models/phrases_model_new(op2).bin')
        
    return phrases_model #return the list




#parser text input

@st.cache_data
def parser_text(input_keys): #function

    # Defining measuring units
    remove_ = {'oil', 'salt', 'pepper'}


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


            # removing measuring words


            items = [word for word in items if word not in remove_]


            if items:
                cleaned_ingredients.extend(items)


        cleaned_ingredients_all_recipes.append(cleaned_ingredients)

    return(cleaned_ingredients_all_recipes)





#parser user text 

@st.cache_data
def parser_user_text(input_keys):


#defining measuring units

    measurment_url= 'https://en.wikibooks.org/wiki/Cookbook:Units_of_measurement' #data source


    measuring_words= ['ml', 'mL', 'milliliter', 'millilitre', 'cc' , 'cubic centimeter', 'l', 'L', 'liter', 'litre', 'dl', 'dL', 'deciliter', 'decilitre', 'teaspoon', 't' , 'tsp.',
'tablespoon' , 'T', 'tbl', 'tbs', 'tbsp', 'fluid ounce', 'fl oz',  'gill', 'cup',  'c', 'pint', 'p', 'pt', 'fl pt',
'quart', 'q', 'qt', 'fl qt', 'gallon' , 'g' , 'gal' , 'g', 'milligram', 'milligramme', 'g' , 'gram' , 'gramme', 'kg',
'kilogram', 'kilogramme', 'pound', 'lb', 'ounce', 'oz', 'mm', 'millimeter', 'millimetre', 'cm' , 'centimeter', 'centimetre', 'm' , 'meter',
'metre', 'inch', 'in', 'yard', '°C' , 'degree celsius','°F' ,'Farenheit', 'tsp']
    
    cleaned_ingredients_all_recipes = []


    #for each_ingredient_list in input_keys:
    comma_list = re.split(',', input_keys)
    
    #comma_list = re.split(' ', each_ingredient_list)   # splitting the ingredients by commas


    cleaned_ingredients = []  # new list to store cleaned ingredients for all  recipes


    lemmatizer = WordNetLemmatizer()  # lemmatize


    for each_word_set in comma_list:
        items = [word.lower() for word in re.findall(r'\b\w+\b', each_word_set)]  # extract individual words and convert to lowercase
        print('lower:', items)


        items = [word for word in items if word.isalpha()]  # filtering only letters
        print('alpha list:', comma_list)


        items = [lemmatizer.lemmatize(word) for word in items]  # lemmatizing


        items = [word for word in items if word not in ENGLISH_STOP_WORDS]  # removing stop words


        if items:
            cleaned_ingredients.extend(items)
            print('one before last:',cleaned_ingredients )


    #cleaned_ingredients_all_recipes.append(cleaned_ingredients)

    return(cleaned_ingredients)





#recommender

@st.cache_data
def recommender_text(text_input):

    recipes= load_data()
    
    phrases_model= load_model_text()
    print('model')

    text_input.split(',')
    print('after split:', text_input)
    

    user_input= parser_user_text(text_input)
    print('after parser user text:', user_input )
    

    # mean Embeddings for User Input
    user_mean_vector =np.mean([phrases_model.wv[user_text] for user_text in user_input if user_text in phrases_model.wv] or [np.zeros(200)], axis=0)



    #text data

    text_data= recipes['text_data']
    

    text_data_cleaned= parser_text(text_data)
    

    
    print('after parser:', text_data_cleaned[0])
    




    # check if user_mean_vector contains NaN values
    if np.isnan(user_mean_vector).any():
        print("User input vectors contain NaN values.")


    else:

        # mean ingredient vectors for each recipe
        text_vectors = [np.mean([phrases_model.wv[sub_text] for sub_text in text if sub_text in phrases_model.wv] or [np.zeros(200)], axis=0) for text in text_data_cleaned]





        # Check if any recipe vector contains NaN values
        if not text_vectors:
            print('nothing obtained for recipe vectors')

        else:
            
            # cosine similarity between user mean vector and recipe mean vectors
            cosine_similarities_text= cosine_similarity([user_mean_vector], text_vectors)


            # indices of top N most similar recipes
            top_recipes = np.argsort(cosine_similarities_text[0])[::-1][:5]



            recommendation= pd.DataFrame( columns=['Name','Ingredients', 'Category', 'Keywords', 'Calories', 'Time', 'Score'] ) #dataframe with columns

    
            for i in top_recipes: #defining the data for each recommendation

            
                Name = recipes['Name'].iloc[i]
                Calories = '{:.0f}'.format(recipes['Calories'].iloc[i])
                Time= recipes['TotalTime'].iloc[i]
                Ingredients = recipes['RecipeIngredientParts'].iloc[i]
                Category = recipes['RecipeCategory'].iloc[i]
                Keywords = recipes['Keywords'].iloc[i]
                Score= '{:.0f}'.format(cosine_similarities_text[0][i])

                recommendation = pd.concat([recommendation,pd.DataFrame({
                                    'Name': [Name], 
                                    'Ingredients': [Ingredients], 
                                    'Category' : [Category],
                                    'Keywords' : [Keywords], 
                                    'Calories': [Calories], 
                                    'Time' : [Time],
                                    'Score': [Score]
                                    })], 
                                        ignore_index=True)




    return recommendation

       

#




#recommending logic

st.divider()
st.subheader('Search by Keywords')

#user input

keyword_input= st.text_input('Do you want to search by a keyword today??')

if st.button('Lets key-in !!'):

    loading_image= st_lottie((requests.get('https://lottie.host/9e4a2516-2bd2-4f3a-97dc-e13a1c8b17d3/WsU810K1aZ.json',verify=False).json()), height=100, width=100 )

    recommend_recipes= recommender_text(keyword_input)
    
    if recommend_recipes is not None and not recommend_recipes.empty:

        
        st.table(recommend_recipes)
        st.empty()

    else:

        st.write('I am sorry, no good recommendations today')
