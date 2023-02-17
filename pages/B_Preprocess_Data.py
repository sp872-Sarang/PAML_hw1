import numpy as np                      # pip install numpy
import pandas as pd                     # pip install pandas
import matplotlib.pyplot as plt        # pip install matplotlib
from sklearn.model_selection import train_test_split
import streamlit as st                  # pip install streamlit

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Homework 1 - End-to-End ML Pipeline")

#############################################

st.markdown('# Preprocess Dataset')

#############################################

# Checkpoint 1
def restore_dataset():
    """
    Input: 
    Output: 
    """
    df=None
    if 'house_df' not in st.session_state:
        data = st.file_uploader('Upload a Dataset', type=['csv', 'txt'])
        if data:
            df = pd.read_csv(data)
            st.session_state['house_df'] = df
        else:
            return None
    else:
        df = st.session_state['house_df']
    return df

# Checkpoint 3
def remove_features(X,removed_features):
    """
    Input: 
    Output: 
    """
    x = X.drop(removed_features, axis=1)
    return x

# Checkpoint 4
def impute_dataset(X, impute_method):
    """
    Input: 
    Output: 
    """
    if impute_method == 'Zero':
        X = X.fillna(0)
    elif impute_method == 'Mean':
        X = X.fillna(X.mean())
    elif impute_method == 'Median':
        X = X.fillna(X.median())
    else:
        st.markdown('Invalid impute method. Please select a valid impute method')

    return X

# Checkpoint 5
def compute_descriptive_stats(X, stats_feature_select, stats_select):
    """

    Input: 
    X - pandas dataframe
    stats_feature_select - feature to compute stats on
    stats_select - method of stats to compute

    Output: 
    output_str - string of stats
    out_dict - dictionary of stats

    """
    for stat in stats_select:
        if stat == 'Mean':
            mean = X[stats_feature_select].mean().item()
        elif stat == 'Median':
            median = X[stats_feature_select].median().item()
        elif stat == 'Max':
            max = X[stats_feature_select].max().item()
        elif stat == 'Min':
            min = X[stats_feature_select].min().item()
        else:
            st.markdown('Invalid stats method. Please select a valid stats method')
            return None
    if type(stats_feature_select) == list:
        stats_feature_select = stats_feature_select[0]
    output_str = stats_feature_select + ' mean: {0:.2f} | median: {1:.2f} | max: {2:.2f} | min: {3:.2f}'.format(mean, median, max, min)

    out_dict = {
        'mean': round(mean,2),
        'median': round(median,2),
        'max': round(max,2),
        'min': round(min,2)
    }

    return output_str, out_dict

# Checkpoint 6
def split_dataset(X, number):
    """
    Input: 
    Output: 
    """
    train, test = train_test_split(X, test_size=number/100, random_state=42)
    return train, test

# Restore Dataset
df = restore_dataset()

if df is not None:

    st.markdown('View initial data with missing values or invalid inputs')

    # Display original dataframe
    st.write(df)

    # Show summary of missing values including the 1) number of categories with missing values, average number of missing values per category, and Total number of missing values
    st.markdown('Number of categories with missing values: {0:.2f}'.format(df.isna().any(axis=0).sum()))

    st.markdown('Total number of missing values : {0:.2f}'.format(df.isna().sum().sum()))
    ############################################# MAIN BODY #############################################

    #numeric_columns = list(df.select_dtypes(['float','int']).columns)

    # Provide the option to select multiple feature to remove using Streamlit multiselect
    st.markdown('### Remove features')
    st.markdown('Select features to remove from the dataset')

    # Remove irrelevant/useless features. Collect a user's preferences on one or more features to remove using the Streamlit multiselect function. 
    columns_to_remove = st.multiselect("Select columns to remove", df.columns)

    # Call remove_features function to remove features
    df_filtered = remove_features(df, columns_to_remove)

    # Display updated dataframe
    st.write(df_filtered)

    # Clean dataset
    st.markdown('### Impute data')
    st.markdown('Transform missing values to 0, mean, or median')

    # Use selectbox to provide impute options {'Zero', 'Mean', 'Median'}
    impute_method = st.selectbox('Select impute method', ['Zero', 'Mean', 'Median'])

    # Call impute_dataset function to resolve data handling/cleaning problems by calling impute_dataset
    df_imputed = impute_dataset(df_filtered, impute_method)
    
    # Display updated dataframe
    st.write(df_imputed)
    
    # Descriptive Statistics 
    st.markdown('### Summary of Descriptive Statistics')

    # Provide option to select multiple feature to show descriptive statistics using Streamit multiselect

    numeric_columns = list(df.select_dtypes(['float','int']).columns)
    selected_columns = st.multiselect("Select columns to summarize", numeric_columns, default=numeric_columns)

    # Provide option to select multiple descriptive statistics to show using Streamit multiselect

    stats = ["Mean", "Median", "Max", "Min"]

    # Display a multiselect widget for the user to select which statistics to show
    selected_stats = st.multiselect("Select statistics to show", stats, default=stats)
 
    # Compute Descriptive Statistics including mean, median, min, max
    for f in selected_columns:
        output_str, out_dict = compute_descriptive_stats(df_imputed, f, selected_stats)
        st.markdown(output_str)
        
    # Display updated dataframe
    st.markdown('### Result of the imputed dataframe')
    st.write(df_imputed)

    # Split train/test
    st.markdown('### Enter the percentage of test data to use for training the model')

    # Compute the percentage of test and training data
    test_size = st.number_input('Enter the percentage of test data in %:', min_value=1, max_value=100, value=20, step=1)

    # Call split_dataset function to split the dataset into train and test
    train, test = split_dataset(df_imputed, test_size)
    
    # Print dataset split result
    st.markdown('Train dataset size: {0:.2f} | Test dataset size: {1:.2f}'.format(train.shape[0], test.shape[0]))

    # Save state of train and test split using streamlit session state
    st.session_state['train'] = train
    st.session_state['test'] = test

    

