#!/usr/bin/env python
# coding: utf-8

# In[9]:


import streamlit as st
import pickle
import pandas as pd


# In[10]:


st.sidebar.header('User Input Parameters')


# In[11]:


data = pd.read_csv("Cleaned_data.csv")


# In[17]:


similarity = pickle.load(open("similarity.pkl", "rb"))


# In[12]:


st.title("Product Recommendation System")


# In[13]:


product_list = data['Product_id'].unique()
selected_product = st.selectbox("Select a product", product_list)


# In[14]:


def recommend(product):
    index = data[data['Product_id'] == product].index[0]
    distances = similarity[index]
    products_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_products = []
    for i in products_list:
        recommended_products.append(data.iloc[i[0]].Product_id)
    return recommended_products


# In[15]:


if st.button("Recommend"):
    recommendations = recommend(selected_product)
    st.subheader("Recommended Products:")
    for item in recommendations:
        st.write(item)