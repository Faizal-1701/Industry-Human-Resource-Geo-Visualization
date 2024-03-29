import io
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import plotly.express as px
import numpy as np

# Read the CSV files
data=pd.read_csv("D:/New folder/Data Science Materials/HR data/CSV FILES/IndustrialHumanResource.csv", encoding='cp1252')
def load_data():
   return data

# Ensure correct data types and handle missing values
# Make sure to replace 'your_column_name' with actual column names from your data
data['Main Workers - Total - Males'] = pd.to_numeric(data['Main Workers - Total - Males'], errors='coerce').fillna(0)
data['Main Workers - Total - Females'] = pd.to_numeric(data['Main Workers - Total - Females'], errors='coerce').fillna(0)

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
                <h2 style = 'text-color:yellow;'>:green[Domain:]<br> :violet[Resource Management]</h2></b>""", unsafe_allow_html=True)
    
selected = option_menu("Main Menu", ["Data Exploration", "Data Cleaning", "Statistical Metrics", "Feature Engineering", "Data Visualization"], 
                       icons=['search', 'brush', 'calculator', 'gear', 'graph-up'], default_index=0,
                    styles={"nav-link": {"font-size": "20px", "text-align": "center", "margin": "-2px", "--hover-color": "gray"},
                    "nav-link-selected": {"background-color": "gray"}}, orientation="horizontal")

# Main app
if selected == "Data Exploration":
    state_data = data.groupby('ï»¿State Code').agg({
    'Main Workers - Total -  Persons': 'sum',
    'Main Workers - Total - Males': 'sum',
    'Main Workers - Total - Females': 'sum'
}).reset_index()
    
    def load_data():
    # Your code to load and return the DataFrame
        return state_data
    st.markdown('## :green[Data Exploration]')
    exp = st.radio(label = '## :green[Data Exploration]',label_visibility='hidden', options= [':orange[Dataset]', ':red[Main Workers Total Persons - State wise]',':green[Identify Numerical Columns]',':blue[Distribution of Numerical Columns]'],
                   index=0, horizontal= False)
    if exp == ':orange[Dataset]':
        st.header(":rainbow[Loaded Dataset]")
        st.write(data, '\n', data.shape)
    if exp == ':red[Main Workers Total Persons - State wise]':
        st.subheader(":rainbow[Main Workers Total Persons State Wise]")
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(data['Main Workers - Total - Females'].value_counts(), width=500)
        with col2:
            st.dataframe(data['Main Workers - Total - Males'].value_counts(), width=500)
    if exp == ':green[Identify Numerical Columns]':
        st.subheader(":rainbow[Identify Numerical Columns]")
        numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
        numerical_cols
    if exp == ':blue[Distribution of Numerical Columns]':
        plt.figure(figsize=(8, 6))
        sns.histplot(data['ï»¿State Code'], kde=True)
        plt.title('Distribution of Your Numerical Column')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.xticks(rotation=45)
        plt.xlabel('State Code')
        plt.ylabel('Frequency') 
        st.pyplot()

        data = load_data()
elif selected == "Data Cleaning":
    st.markdown("## :green[Data Cleaning]")
    def main(data):
            # Display the first few rows of the DataFrame
            st.markdown("## :rainbow[Head of the Dataset]")
            st.write(data.head())
            st.dataframe(data)

            # Display descriptive statistics
            st.markdown("## :rainbow[Descriptive Statistics]")
            st.write(data.describe())

            data.fillna(data.mean(numeric_only=True), inplace=True)  # For numerical columns
            data.fillna('Unknown', inplace=True)  # For categorical columns

            # Remove duplicates
            data['State Code'] = data['ï»¿State Code'].str.replace('ï»¿', '').astype(int)
            # Normalize data (Example: Min-Max scaling)
            scaler = MinMaxScaler()
            data[['Main Workers - Total -  Persons']] = scaler.fit_transform(data[['Main Workers - Total -  Persons']])
            # Convert categorical data to numerical (Example: One-hot encoding)
            data = pd.get_dummies(data, columns=['State Code'])

            # Split data
            from sklearn.model_selection import train_test_split
            #X_train, X_test, y_train, y_test = train_test_split(df.drop('Main Workers - Total - Persons', axis=1), df['Main Workers - Total - Persons'], test_size=0.2, random_state=42)
            X = data.drop('Main Workers - Total -  Persons', axis=1)
            y = data['Main Workers - Total -  Persons']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                # Visualize distribution
            st.markdown("## :rainbow[Categorical Data Consistency]")
            categorical_columns = ['State Code', 'District Code', 'India/States Division Group', 'Class', 'NIC Name']
            for col in categorical_columns:
                if col in data:
                    st.write(f":green[Unique values in {col}:]")
                    st.write(data[col].unique(),)

                    # Show effect of One-Hot Encoding
            if st.checkbox(" :green[Show One-Hot Encoded Data]"):
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

                # Display the Matplotlib plot in Streamlit
                st.pyplot(fig)

    if __name__ == "__main__":
        main(data)

        # Show statistics of numerical columns
    data = load_data()

elif selected == "Statistical Metrics":
    st.markdown("## :green[Statistical Metrics]")

    def plot_missing_values_heatmap(data):
        plt.figure(figsize=(12, 6))
        sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
        plt.title("Heatmap of Missing Values in the Dataset")
        return plt
    null_counts = data.isnull().sum()

# Display the result in Streamlit
    st.markdown("### :rainbow[Null Values in Each Columns]")
    st.write(null_counts)

# In your Streamlit app
    if st.button("Show Missing Values Heatmap"):
        plt = plot_missing_values_heatmap(data)
        st.pyplot(plt)

    st.markdown("### :rainbow[Mean Values]")
    mean_values = data.mean(numeric_only=True)
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
    correlation_matrix = data.corr()

    plt.figure(figsize=(25,15))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Matrix')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    correlation_matrix
    plt.tight_layout()
    st.pyplot()
    
    st.markdown('### :rainbow[Skewnwss]')
    skewness = data.skew()
    skewness


elif selected == "Feature Engineering":
    st.markdown("## :green[Feature Engineering]")
    st.markdown('### :rainbow[Visualize the distribution of a numerical variable]')
    data_info = data.info()

# Selecting features and target variable for the regression model
# Assuming we want to predict 'Main Workers - Total - Persons'
    target = 'Main Workers - Total -  Persons'
    features = data.columns.drop(target)

# For simplicity, let's select only numerical features for now
# In a real-world scenario, you would also want to consider categorical variables and potentially encode them
    numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.drop(target)

    # Fill missing values with the mean (simple imputation)
    data[numerical_features] = data[numerical_features].fillna(data[numerical_features].mean())

# Splitting the dataset into training and testing sets
    X = data[numerical_features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardizing the numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Linear Regression Model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)

    # Evaluating the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mse, r2
    sns.set_style("whitegrid")
    try:
        data = pd.read_table(data,  encoding='ISO-8859-1')
    except Exception as e:
        error_message = str(e)

# Check if the data is loaded successfully or not
    if 'data' in locals():
        success = True
        preview = data.head()
    else:
        success = False
        preview = None

    success, preview
    st.markdown("### :rainbow[Model Evaluation Metrics:]")
    st.metric(label=":green[Mean Squared Error (MSE)]", value=mse)
    st.metric(label=":green[R-squared (R2)]", value=r2)
    grouped_data = data.groupby('ï»¿State Code')['Main Workers - Total -  Persons'].sum()

    fig, ax = plt.subplots()
    grouped_data.plot(kind='bar', ax=ax)
    ax.set_title('Total Main Workers by State')
    ax.set_xlabel('State Code')
    ax.set_ylabel('Main Workers - Total -  Persons')

# Displaying the plot in Streamlit
    st.pyplot(fig)

elif selected == "Data Visualization":
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
