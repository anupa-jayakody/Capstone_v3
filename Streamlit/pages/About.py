import streamlit as streamlit

streamlit.set_page_config(
    page_title="Home",
    #page_icon="👋",
    page_icon= ":fork_and_knife:"
)

streamlit.write("# Welcome to Flavor Fuze :curry:")

streamlit.caption('By Anupa Jayakody')



streamlit.markdown(
        """
        Flavor Fuze revolutionizes cooking with personalized recipes tailored to your tastes and pantry inventory. Explore culinary creativity effortlessly, transforming each meal into an exciting adventure in your kitchen. 
        
        #### Experience the joy of cooking like never before with Flavor Fuze!!


        


        ## How did Flavor Fuze cook?? :green_apple:

        - First I cleaned data using NLP data processing methods.
        - Then I tried different models and evaluated the results.
        - Then I picked NLP Word2Vec and FunkSVD models for the final solution.
        - Next I developed the Recommender System.
        - Finally I developed the customer application using Streamlit.

        


        ## What Flavor Fuze offers you?? :pizza:

        - A recipe recommendation based on different ingredients or keywords.
        - A recipe recommendation based on your past rated recipes.
    """
    )
