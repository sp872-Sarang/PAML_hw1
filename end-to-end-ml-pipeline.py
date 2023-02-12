import numpy as np                      # pip install numpy
import pandas as pd                     # pip install pandas
import matplotlib.pyplot as plt         # pip install matplotlib
from sklearn import preprocessing, svm  #  pip install -U scikit-learn 
from sklearn.model_selection import train_test_split
import streamlit as st                  # pip install streamlit

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Homework 1 - End-to-End ML Pipeline")

#############################################

st.markdown("Welcome to the Practical Applications in Machine Learning (PAML) Course! You will build a series of end-to-end ML pipelines, working with various data types and formats, and will need to engineer your system to support training, testing, and deploying ML models.")

st.markdown("""**Homework 1:** The goal of this assignment is to build your first end-to-end Machine Learning (ML) pipeline using public datasets and by creating your own datasets. The learning outcomes for this assignment are:

- Build first end-to-end ML pipeline in Jupyter Notebook, focusing primarily on data curation (collection and handling) and preprocessing techniques.
- Develop web application that walks users through steps of the pipeline 
- Collect small dataset and perform data labeling study (in-class activity during week on ‘Dataset Acquisition and Curation’).
""")

st.markdown("""You will build a web application that helps Machine Learning Practitioners through an end-to-end ML pipeline.

    1. Explore and visualize the data to gain insights.
    2. Preprocess and prepare the data for machine learning algorithms.
    3. Select a model and train it. Fine-tune your model.
    4. Test Model.  Evaluate models using standard metrics.
    5. Deploy your application (present your solution). Launch, monitor, and maintain your system.
""")

st.markdown(""" This assignment is contains two parts:

Part I - End-to-End ML Pipeline: Many ML projects are NOT used in production or NOT easily used by others including ML engineers interested in exploring prior ML models, testing their models on new datasets, and helping users explore ML models. To address this challenge, the goal of this assignment is to implement a front- and back-end ML project, focusing on the dataset exploration and preprocessing steps. The hope is that the homework assignments help students showcase ML skills in building end-to-end ML pipelines and deploying ML web applications for users which we build on in future assignments. 

Part II - Dataset Curation (In-Class Activity): It is often challenging to collect datasets when none exists in your problem domain. Thus, it is important to understand how to curate new datasets and explore existing methodologies for data collection. Part II of HW1 focuses on how to collect datasets, annotate the data, and evaluate the annotations in preparation for ML tasks.

HW1 serves as an outline for the remaining assignments in the course, building end-to-end ML pipelines and deploying useful web applications using those models. This assignment in particular focuses on the data exploration and preprocess. 
""")

st.markdown(""" Future homework assignments will engage students in more an in depth implementation and discussion of ethical implications of four applications.
    
    1. Regression for Predicting Housing Prices
    2. Clustering for Document Retrieval
    3. Classification for Product Recommendation
    4. Deep Learning for Image Search
""")

st.markdown("Click **Explore Dataset** to get started.")