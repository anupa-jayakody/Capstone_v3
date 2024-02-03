# Readme

**Project Overview**


The end goal of the project is to satisfy your hunger buds by giving  a recipe within a few seconds to match your needs.  
The scope of the project will be to develop a recipe recommender system that will help the user choose a recipe that they could try on their own. 



**Problem Area**

With the vast majority of the people being non-vegans, most of the recipe content found on different channels/sites is not vegan-friendly. Hence vegans usually struggle to find recipes tailored to their liking and often end up needing to find substitutes to prepare food on their own with no proper following, which is not a convenient solution when they need a quick preparation.

More importantly, with everyone seeking easy getaway solutions, people need a quick solution to a recipe when they plan to prepare food. 

So the project will focus on delivering 2 solutions;  

1. Recommending  a recipe  (Vegan data can be filtered out as well)
    - Based on the ingredients you have on hand or you need to be included in the recipe 
    - Based on keywords such as the cuisine, type of recipe etc. 
    - Based on both keywords & ingredients combined
2. Predict the rating for the recipe 

**My original scope was to only give vegan recommendations, but with the dataset changing,  I was able to widen the scope for generic plus vegan recipes. 



**Data Science Approach**


**Solution 1 - Content-Based Filtering**

Option 1- Search by Ingredients

1. User will input a n number of ingredients that they have on hand right now or the ingredients that they specifically want to be included in the recipe. 
2. The solution will check the database of recipes and match the available recipe  ingredients with the input ingredients.
3. It will then recommend a specific number of recipes to the user based on the similarity between the input ingredients and the database recipe ingredients using an appropriate similarity matrix.
4. The user then selects the recipe that they like and follows the steps to prepare it and/or refers to the web link to get more details


Option 2- Search by Keywords

1. The user will input a keyword to search for in the recipe in the recipe. 
2. The solution will check the database of recipes and match the available recipe  keywords/category/name with the input key.
3. It will then recommend a specific number of recipes to the user based on the similarity between the input ingredients and the database recipe ingredients using an appropriate similarity matrix.
4. The user then selects the recipe that they like and follows the steps to prepare it and/or refers to the web link to get more details

Option 3- Search by Ingredients & Keywords

**Solution 2- Collaborative Based Filtering**

Predict the recipe rating. This will enable the repeating users to be recommended with a recipe based on the item-item similarity (rating). In item-item filtering, we say that if two items are similar, and a user ranked one of those items, that user's rating of the other item will be similar.



**Impact of the solution**

- Enable any individual to find a recipe within 1-2  mins to their liking, to suit their ingredient preferences or what they have on hand making it convenient to find a recipe matching their need at that point. 

- Provide generic recommendations to which the individual might not have any ingredient readily available. 

By giving a recommendation considering the above, he will not need to read through the entire recipe first to decide whether it will be a good try for him at that point as the recommendations will be given based on his preferences. 



**Dataset**

The capstone scope of the project is to develop the solution based on a dataset of ~520000 recipes and ~1401982 reviews extracted from food.com.(Kaggle) and  So the solution developed will be from this dataset.

[But I wish to extend this to a live extracting of recipes so that the recipe database will be much wider. I also plan to add an approximate cost for each recipe based on the market cost for each ingredient so that the user can refine their selection based on the budget criteria as well.] 

The dataset includes below columns: 

- RecipeId 
- Name 
- AuthorId 
- AuthorName 
- CookTime 
- PrepTime 
- TotalTime 
- DatePublished 
- Description 
- Images 
- RecipeCategory 
- Keywords 
- RecipeIngredientQuantities 
- RecipeIngredientParts 
- AggregatedRating 
- ReviewCount 
- Calories 
- FatContent 
- SaturatedFatContent 
- CholesterolContent 
- SodiumContent 
- CarbohydrateContent 
- FiberContent 
- SugarContent 
- ProteinContent 
- RecipeServings 
- RecipeYield 
- RecipeInstructions 


**Data Cleaning**

Below are the processes followed to conduct the preliminary analysis of the dataset and to do an EDA.  

Dataset checking  

- Summary and types of data 
- Count
- Duplicates
- Null/missing values 


**EDA**
    
Since the dataset has many text columns, text data and recipe attributes relevant to the scope of the project was analysed.

- Most common ingredients by frequency of usage in recipes
- Most common recipe types by the title words
- Length of recipes by ingredient count
- Length of recipes by title count
- Correlation between length of titles and ingredients 
- Recipe rating distribution
- Correlation of different nutrition contents
- Distribution of recipes by nutritional contents 
- Vegan recipes distribution


**Pre-Processing**

When analysing data, it was visible that the ingredients column has many words that are not real ingredients. Since this is the column we will use for our modeling, I removed those unnecessary words for the analysis. Techniques used here are; 

- Tokenization
- Lemmatization
- Removing stop words
- Removing measuring units
- Removing symbols/characters/numbers
- Lower casing

In Sprint 1, I separated each ingredient to individual tokens and analysed them, however, it was visible that some ingredients need to be maintained in the original form such as  red pepper,  and bell pepper. Therefore, I improved my pre-processing steps to incorporate this. Furthermore, one function was created by combining all the steps. 

A similar process was used to preprocess the Title and RecipeInstructions columns too.



**Modeling**


**Solution 1 - Content-Based Filtering**

I used the below word embedding models for the recommendation working
- TFIDF vectorisation
- Word2Vec vectorisation


Below are the steps used

- Use Word2Vec, TFIDF vectorization for model comparison
- Calculating mean embedding for each ingredient 
- Calculating the mean embedding for the input ingredient
- Map the embeddings of different words to check if it is sensible
- Enter random keywords to get the most similar ingredients 
- Calculating the cosine similarity between the input and the recipe
- Sorting the results by the scores 
- Select the best model
- Finetune the hyperparameters

I selected the Word2Vec model here for my next steps. Both Pre-processing and Modeling steps were done for both search options (Ingredient and Keywords) 


**Solution 2- Collaborative Based Filtering**

FunkSVD was used to get the predicted ratings for each recipe
Actual vs Predicted was compared with the highest FCP of 0.64 for the test data




**Productization**

Web application & mobile application



**Next steps**

- Speed up the recommendations (Currently it takes ~1min for app processing)
- Give a URL to directly connect with the source recipe for the user to follow along easily
- Convert the data source to a dynamic API pulling data from the web
- Create a mobile application



