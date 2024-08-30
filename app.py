# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns

# Load Data
df = pd.read_csv('diabetes.csv')

# HEADINGS
st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())

# X AND Y DATA
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# FUNCTION
def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0, 67, 20)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
    age = st.sidebar.slider('Age', 21, 88, 33)

    user_report_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# PATIENT DATA
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

# MODEL
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
user_result = rf.predict(user_data)

# VISUALISATIONS
st.title('Visualised Patient Report')

# COLOR FUNCTION
if user_result[0] == 0:
    color = 'blue'
else:
    color = 'red'

# Plotting Functions
def plot_graph(x, y, user_x, user_y, title, palette, xticks, yticks):
    fig = plt.figure()
    ax = sns.scatterplot(x=x, y=y, data=df, hue='Outcome', palette=palette)
    sns.scatterplot(x=[user_x], y=[user_y], s=150, color=color)
    plt.xticks(np.arange(*xticks))
    plt.yticks(np.arange(*yticks))
    plt.title(title)
    st.pyplot(fig)

# Graphs
plot_graph('Age', 'Pregnancies', user_data['Age'][0], user_data['Pregnancies'][0], 'Pregnancy count Graph (Others vs Yours)', 'Greens', (10, 100, 5), (0, 20, 2))
plot_graph('Age', 'Glucose', user_data['Age'][0], user_data['Glucose'][0], 'Glucose Value Graph (Others vs Yours)', 'magma', (10, 100, 5), (0, 220, 10))
plot_graph('Age', 'BloodPressure', user_data['Age'][0], user_data['BloodPressure'][0], 'Blood Pressure Value Graph (Others vs Yours)', 'Reds', (10, 100, 5), (0, 130, 10))
plot_graph('Age', 'SkinThickness', user_data['Age'][0], user_data['SkinThickness'][0], 'Skin Thickness Value Graph (Others vs Yours)', 'Blues', (10, 100, 5), (0, 110, 10))
plot_graph('Age', 'Insulin', user_data['Age'][0], user_data['Insulin'][0], 'Insulin Value Graph (Others vs Yours)', 'rocket', (10, 100, 5), (0, 900, 50))
plot_graph('Age', 'BMI', user_data['Age'][0], user_data['BMI'][0], 'BMI Value Graph (Others vs Yours)', 'rainbow', (10, 100, 5), (0, 70, 5))
plot_graph('Age', 'DiabetesPedigreeFunction', user_data['Age'][0], user_data['DiabetesPedigreeFunction'][0], 'DPF Value Graph (Others vs Yours)', 'YlOrBr', (10, 100, 5), (0, 3, 0.2))

# OUTPUT
st.subheader('Your Report:')
output = 'You are not Diabetic' if user_result[0] == 0 else 'You are Diabetic'
st.title(output)
st.subheader('Accuracy:')
st.write(f"{accuracy_score(y_test, rf.predict(x_test)) * 100:.2f}%")
