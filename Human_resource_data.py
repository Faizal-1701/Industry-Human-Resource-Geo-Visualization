import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import plotly.express as px
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import Counter

# Read the CSV files
data=pd.read_csv("D:/New folder/Data Science Materials/HR data/CSV FILES/IndustrialHumanResource.csv", encoding='cp1252')
def load_data():
   return data

## streamlit parts
st.set_page_config(page_title= "Industrial Human Resource Data | Mohammed Faizal N", page_icon='chart_with_upwards_trend',layout='wide',initial_sidebar_state= "expanded")
def setting_bg(background_color):
    st.markdown(f""" <style>.stApp {{
                        background-color:{background_color};
                       }}
                       </style>""", unsafe_allow_html=True)
setting_bg("#1d2a40")

# # Sidebar for navigation
with st.sidebar:
    st.title(":rainbow[Industrial Human Resource Geo-Visualization]")
    st.markdown("""<b><h2 style='text-color:yellow;'>:green[Technologies:]</h2>
                :violet[<ol><li>EDA</li>
                    <li>Visualization</li> 
                    <li>NLP</li></ol>]
                <h2 style = 'text-color:yellow;'>:green[Domain:]<br> <br>:violet[Resource Management]</h2></b>""", unsafe_allow_html=True)
    
selected = option_menu("Main Menu", ["Data Exploration & Cleaning", "Statistical Metrics", "Feature Engineering ","Data Visualization"], 
                       icons=['brush', 'calculator', 'gear', 'graph-up'], default_index=0,
                    styles={"nav-link": {"font-size": "20px", "text-align": "center", "margin": "-2px", "--hover-color": "gray"},
                    "nav-link-selected": {"background-color": "gray"}}, orientation="horizontal")
## data cleaning
def cleaning_1(data):
    cl_columns = data[['ï»¿State Code','District Code', 'India/States','Division','Group','Class']]
    for cl in cl_columns:
        data[cl] = data[cl].str.replace("`",'').str.replace('ï»¿','')
    data['State Code'] = data['ï»¿State Code'].str.replace('ï»¿','')
    data.drop('ï»¿State Code', axis =1, inplace = True)
    data.drop_duplicates()    
    return data

if selected == "Data Exploration & Cleaning":
    def main(data):
            # Display the first few rows of the DataFrame
            st.markdown("## :green[Data Cleaning]")    
            st.markdown("## :rainbow[Head of the Dataset Before cleaning]")
            st.write(data.head())
            print(data.info())
            cleaning_1(data)
            st.markdown("## :rainbow[Head of the Dataset After cleaning]")
            st.dataframe(data.head())
            st.write(data.shape)
            st.markdown("### :rainbow[Null Values in Each Columns]")
            st.write(data.isnull().sum())
            st.markdown("### :green[There is No-Null Values]")
            st.markdown("## :rainbow[Descriptive Statistics]")
            st.write(data.describe().T)

            # Normalize data (Example: Min-Max scaling)
            scaler = MinMaxScaler()
            data[['Main Workers - Total -  Persons']] = scaler.fit_transform(data[['Main Workers - Total -  Persons']])
            
            # Split data
            X = data.drop('Main Workers - Total -  Persons', axis=1)
            y = data['Main Workers - Total -  Persons']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                # Visualize distribution
            st.markdown("## :rainbow[Categorical Data Consistency]")
            categorical_columns = ['State Code', 'District Code', 'India/States', 'Division', 'Group', 'Class', 'NIC Name']
            for col in categorical_columns:
                if col in data:
                    st.write(f":green[Unique values in {col}:]")
                    st.write(data[col].unique())

                    # Show effect of One-Hot Encoding
            if st.checkbox(" :green[Show One-Hot Encoded Data]"):
                # Convert categorical data to numerical (Example: One-hot encoding)
                data = pd.get_dummies(data, columns=['State Code'])
                st.write("### :rainbow[One-Hot Encoded Data]") 
                st.write(data)
                # Group the data by industry division and sum up the number of main workers for each division
                industry_division_totals = data.groupby('Division')['Main Workers - Total -  Persons'].sum().sort_values(ascending=False)
                fig, ax = plt.subplots()
                industry_division_totals.plot(kind='bar', color='skyblue', ax=ax)
                ax.set_title('Total Main Workers by Industry Division')
                ax.set_xlabel('Industry Division')
                ax.set_ylabel('Number of Main Workers')
                ax.legend(['Main Workers'])
                plt.xticks(rotation=90)
                plt.tight_layout()
                st.pyplot(fig)

    if __name__ == "__main__":
        main(data)

elif selected == "Statistical Metrics":
    st.markdown("## :green[Statistical Metrics]")
    st.markdown("### :rainbow[Mean Values]")
    numeric_data = data.select_dtypes(include=['number'])
    mean_values = numeric_data.mean()
    mean_values
    st.markdown("### :rainbow[Mean Values of each Columns]")
    fig, ax = plt.subplots()
    sns.barplot(x=mean_values.index, y=mean_values.values, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_ylabel("Mean Value")
    st.pyplot(fig)

    st.markdown("### :rainbow[Median Values]")
    median_values = data.median(numeric_only=True)
    median_values
    st.markdown("### :rainbow[Median of Each Columns]")
    fig, ax = plt.subplots()
    sns.barplot(x=median_values.index, y=median_values.values, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_ylabel("Median Value")
    st.pyplot(fig)

    st.markdown("### :rainbow[Standard deviation]")
    data = data.apply(pd.to_numeric, errors='coerce')
    numeric_data = data.select_dtypes(include=[np.number])
    data.fillna(data.mean(), inplace=True)
    std_dev = numeric_data.std()
    std_dev = data.std()
    std_dev

    st.markdown("### :rainbow[To calculate quantiles]")
    quantiles = data.quantile([0.25, 0.5, 0.75])
    quantiles
    st.markdown("### :rainbow[Correlation matrix]")
    correlation_matrix = data.corr(numeric_only=True)
    correlation_matrix
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.figure(figsize=(25,15))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    st.pyplot()
    
    st.markdown('### :rainbow[Skewness]')
    skewness = data.skew()
    skewness

elif selected == "Feature Engineering ":
    st.markdown("## :green[Feature Engineering]")
    st.markdown('### :rainbow[Visualize the distribution of a numerical variable]')
    data = cleaning_1(data)
# Assuming we want to predict 'Main Workers - Total - Persons'
    target = 'Main Workers - Total -  Persons'
    features = data.columns.drop(target)

    numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.drop(target)
    # Fill missing values with the mean (simple imputation)
    data[numerical_features] = data[numerical_features].fillna(data[numerical_features].mean())
    data[numerical_features]

    st.markdown("## :green[Model Buiding & Testing]")
# Splitting the dataset into training and testing sets
    X = data[numerical_features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardizing the numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = st.selectbox("Please select your model",options=['Linear Regression Model', "DecisionTreeRegression Model"])
    if model == 'Linear Regression Model':
        st.markdown("### :rainbow[Using Linear Regression Model]")
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.markdown(":green[Mean Squared Error (MSE)]")
        mse
        st.markdown(":green[R-squared (R2)]")
        r2
# Creating the decision tree regressor model
    if model == 'DecisionTreeRegression Model':
        st.markdown("### :rainbow[DecisionTreeRegression Model]") 
        tree_model = DecisionTreeRegressor()
        tree_model.fit(X_train_scaled, y_train)
        y_pred_tree = tree_model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred_tree)
        r2 = r2_score(y_test, y_pred_tree)
        st.markdown(":green[Mean Squared Error (MSE)]")
        mse
        st.markdown(":green[R-squared (R2)]")
        r2
      
elif selected == "Data Visualization":
    cleaning_1(data)
    st.markdown('## :green[Data Visualization]')  
    def plot_histogram(column):
        plt.figure(figsize=(10, 6))
        sns.histplot(data[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        st.pyplot()

# Function to plot a bar chart
    def plot_bar_chart(column):
        plt.figure(figsize=(10, 6))
        data[column].value_counts().plot(kind='bar')
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        st.pyplot()

    # Select a column to visualize
    column = st.selectbox("### :rainbow[Select a column for visualization]", data.columns)

    # Select a type of plot
    plot_type = st.selectbox("### :rainbow[Select the type of plot]", ["Histogram", "Bar Chart"])

    # Display the selected plot
    if plot_type == "Histogram":
        plot_histogram(column)
    elif plot_type == "Bar Chart":
        plot_bar_chart(column)

    st.markdown("## :rainbow[Natural Language Processing]") 
    # Tokenization
    series = ','.join(data['NIC Name'])
    tokens = word_tokenize(series)
    # Remove stopwords
    additional_stopwords = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+', '-', '=', '/', '\\', '|', ';', ':', '"', "'", '<', '>', ',', '.', '?', '[', ']', '{', '}']
    stop_words = set(stopwords.words('english') + additional_stopwords)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

    # Word frequency analysis
    word_freq = Counter(filtered_tokens)
    st.dataframe(word_freq.most_common(20))  # Print the 10 most common words
