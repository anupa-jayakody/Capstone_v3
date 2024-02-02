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


    recipes = pd.read_csv('C:/Users/e312995/OneDrive - WESCO DISTRIBUTION/Documents/PERSONAL/BRAINSTATION/CAPSTONE/recipe_sample_streamlit.csv')
    return recipes


#parser ingredients input

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





#parser model
@st.cache_data
def model():
    
    phrases_model= Word2Vec.load('C:/Users/e312995/OneDrive - WESCO DISTRIBUTION/Documents/PERSONAL/BRAINSTATION/CAPSTONE/phrases_model_new(op1).bin')
        
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



            recommendation= pd.DataFrame( columns=['title','ingredients', 'category','calories', 'time', 'score'] ) #dataframe with columns

    
            for i in top_recipes: #defining the data for each recommendation

            
                title = recipes['Name'].iloc[i]
                calories = recipes['Calories'].iloc[i]
                time= recipes['TotalTime'].iloc[i]
                category = recipes['RecipeCategory'].iloc[i],
                ingredients = recipes['RecipeIngredientParts'].iloc[i]
                score= '{:.0f}'.format(cosine_similarities[0][i])
                recommendation = recommendation.append(
                                {'title': title, 
                                'ingredients': ingredients,
                                'category' : category, 
                                'calories' : calories, 
                                'time' : time, 
                                'score': score}, 
                                     ignore_index=True)
                
    return recommendation  




    
    

#recommending logic

st.title('Recipe recommender')
st.write('by anupa')

st.divider()

#user input

user_ingredients= st.text_input(' Are you hungry and do you want to find a quick delicious recipe that you can make at home?? If so, lets go !!')

if st.button('Lets go !!'):

    #recipes= load_data('C:/Users/e312995/OneDrive - WESCO DISTRIBUTION/Documents/PERSONAL/BRAINSTATION/CAPSTONE/Datasets/recipe_sample_3.csv')

    loading_image= st.image("C:/Users/e312995/OneDrive - WESCO DISTRIBUTION/Documents/PERSONAL/BRAINSTATION/CAPSTONE/Streamlit/loading.gif")


    recommend_recipes= recommender(user_ingredients)
    
    if recommend_recipes is not None and not recommend_recipes.empty:
        
        st.table(recommend_recipes)
    
    else:

        st.write('no recommendations today')






### KEYWORD MODEL
        

#load data
@st.cache_data
def load_model_text():

    phrases_model= Word2Vec.load('C:/Users/e312995/OneDrive - WESCO DISTRIBUTION/Documents/PERSONAL/BRAINSTATION/CAPSTONE/phrases_model_new(op2).bin')
        
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
    print('before parser:', text_data)
    text_data_cleaned= parser_text(text_data)

    #text_data_cleaned= recipes['TextDataCleaned']

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

            
                title = recipes['Name'].iloc[i]
                calories = recipes['Calories'].iloc[i]
                time= recipes['TotalTime'].iloc[i]
                ingredients = recipes['RecipeIngredientParts'].iloc[i],
                category = recipes['RecipeCategory'].iloc[i],
                keywords = recipes['Keywords'].iloc[i]
                score= '{:.0f}'.format(cosine_similarities_text[0][i])
                recommendation = recommendation.append(
                                {'Name': title, 
                                'Ingredients': ingredients, 
                                'Category': category,
                                'Keywords': keywords,
                                'Calories' : calories, 
                                'Time' : time, 
                                'Score': score}, 
                                     ignore_index=True)




    return recommendation

       






#recommending logic

st.divider()

#user input

keyword_input= st.text_input(' Do you want to search by a keyword today?? If so, lets go !!')

if st.button('Lets key !!'):

    loading_image= st.image("C:/Users/e312995/OneDrive - WESCO DISTRIBUTION/Documents/PERSONAL/BRAINSTATION/CAPSTONE/Streamlit/loading.gif")

    recommend_recipes= recommender_text(keyword_input)
    
    if recommend_recipes is not None and not recommend_recipes.empty:
        
        st.table(recommend_recipes)
    
    else:

        st.write('no recommendations today')

