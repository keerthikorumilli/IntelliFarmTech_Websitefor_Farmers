import numpy as np
import pandas as pd
import streamlit as st
import base64
import pickle
import io
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

# Load your machine learning model and data
model = pickle.load(open('model2.pkl', 'rb'))
df = pd.read_csv("Soil.csv")  # Replace with your dataset file

def predict_crop(N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B):
    input = np.array([[N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B]]).astype(np.float64)
    prediction = model.predict(input)
    return prediction[0]

def scatterPlotDrawer(x, y):
    fig = plt.figure(figsize=(20, 15))
    sns.set_style("whitegrid")
    sns.scatterplot(data=df, x=x, y=y, hue="Output", size="Output", palette="deep", sizes=(20, 200), legend="full")
    plt.xlabel(x, fontsize=22)
    plt.ylabel(y, fontsize=22)
    plt.xticks(rotation=90, fontsize=18)
    plt.legend(prop={'size': 18})
    plt.yticks(fontsize=16)
    st.write(fig)

def barPlotDrawer(x, y):
    fig = plt.figure(figsize=(20, 15))
    sns.set_style("whitegrid")
    sns.barplot(data=df, x=x, y=y)
    plt.xlabel("Crops", fontsize=22)
    plt.ylabel(y, fontsize=22)
    plt.xticks(rotation=90, fontsize=18)
    plt.legend(prop={'size': 18})
    plt.yticks(fontsize=16)
    st.write(fig)

def boxPlotDrawer(x, y):
    fig = plt.figure(figsize=(20, 15))
    sns.set_style("whitegrid")
    sns.boxplot(x=x, y=y, data=df)
    sns.despine(offset=10, trim=True)
    plt.xlabel("Crops", fontsize=22)
    plt.ylabel(y, fontsize=22)
    plt.xticks(rotation=90, fontsize=18)
    plt.legend(prop={'size': 18})
    plt.yticks(fontsize=16)
    st.write(fig)

@st.cache_data
def get_img_as_base64_with_transparency(file, transparency):
    img = Image.open(file)

    # Make the image transparent
    img.putalpha(int(255 * (1 - transparency)))  # 0 is fully transparent, 255 is fully opaque

    buffered = io.BytesIO()
    img.save(buffered, format="PNG")

    return base64.b64encode(buffered.getvalue()).decode()

def main():
    transparency = 0.6  # 30% transparency
    img = get_img_as_base64_with_transparency("bg5.jpeg", transparency)

    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
        background-image: url("data:image/png;base64,{img}");
        background-size: cover;
    }}
    </style>
    """

    st.markdown(page_bg_img, unsafe_allow_html=True)

    html_temp_vis = """
    <div style="padding:10px;margin-bottom:30px">
    <h2 style="color:white;text-align:center;"> Visualize Soil Properties </h2>
    </div>
    """

    html_temp_pred = """
    <div style="padding:10px;margin-bottom:30px">
    <h2 style="color:white;text-align:center;"> Which Crop To Cultivate? </h2>
    </div>
    """

    st.sidebar.title("Select One")
    select_type = st.sidebar.radio("", ('Graph', 'Predict Your Crop'))

    if select_type == 'Graph':
        st.markdown(html_temp_vis, unsafe_allow_html=True)
        plot_type = st.selectbox("Select plot type", ('Bar Plot', 'Scatter Plot', 'Box Plot'))
        st.subheader("Relation between features")

        # Plot!
        x = ""
        y = ""

        if plot_type == 'Bar Plot':
            x = 'Output'
            y = st.selectbox("Select a feature to compare between crops",
                ('P', 'N', 'pH', 'K', 'EC', 'OC', 'S', 'Zn', 'Fe', 'Cu', 'Mn', 'B'))
        if plot_type == 'Scatter Plot':
            x = st.selectbox("Select a property for 'X' axis",
                ('P', 'N', 'pH', 'K', 'EC', 'OC', 'S', 'Zn', 'Fe', 'Cu', 'Mn', 'B'))
            y = st.selectbox("Select a property for 'Y' axis",
                ('N', 'P', 'pH', 'K', 'EC', 'OC', 'S', 'Zn', 'Fe', 'Cu', 'Mn', 'B'))
        if plot_type == 'Box Plot':
            x = "Output"
            y = st.selectbox("Select a feature",
                ('P', 'N', 'pH', 'K', 'EC', 'OC', 'S', 'Zn', 'Fe', 'Cu', 'Mn', 'B'))

        if st.button("Visualize"):
            if plot_type == 'Bar Plot':
                barPlotDrawer(x, y)
            if plot_type == 'Scatter Plot':
                scatterPlotDrawer(x, y)
            if plot_type == 'Box Plot':
                boxPlotDrawer(x, y)

    if select_type == "Predict Your Crop":
        st.markdown(html_temp_pred, unsafe_allow_html=True)
        st.header("To predict your crop, give values")
        st.subheader("Drag to Give Values")
        # Calculate the minimum and maximum values from the dataset
        min_N = float(df['N'].min())
        max_N = float(df['N'].max())
        min_P = float(df['P'].min())
        max_P = float(df['P'].max())
        min_K = float(df['K'].min())
        max_K = float(df['K'].max())
        min_pH = float(df['pH'].min())
        max_pH = float(df['pH'].max())
        min_EC = float(df['EC'].min())
        max_EC = float(df['EC'].max())
        min_OC = float(df['OC'].min())
        max_OC = float(df['OC'].max())
        min_S = float(df['S'].min())   
        max_S = float(df['S'].max())       
        min_Zn = float(df['Zn'].min())
        max_Zn = float(df['Zn'].max())
        min_Fe = float(df['Fe'].min())
        max_Fe = float(df['Fe'].max())
        min_Cu = float(df['Cu'].min())
        max_Cu = float(df['Cu'].max())
        min_Mn = float(df['Mn'].min())
        max_Mn = float(df['Mn'].max())
        min_B = float(df['B'].min())
        max_B = float(df['B'].max())

        N = st.slider('Nitrogen (N)', min_value=min_N, max_value=max_N, value=min_N, step=1.0)
        P = st.slider('Phosphorus (P)', min_value=min_P, max_value=max_P, value=min_P, step=1.0)
        K = st.slider('Potassium (K)', min_value=min_K, max_value=max_K, value=min_K, step=1.0)
        pH = st.slider('pH', min_value=min_pH, max_value=max_pH, value=min_pH, step=0.01)
        EC = st.slider('EC', min_value=min_EC, max_value=max_EC, value=min_EC, step=0.01)
        OC = st.slider('OC', min_value=min_OC, max_value=max_OC, value=min_OC, step=0.01)
        S = st.slider('S', min_value=min_S, max_value=max_S, value=min_S, step=1.0)
        Zn = st.slider('Zn', min_value=min_Zn, max_value=max_Zn, value=min_Zn, step=1.0)
        Fe = st.slider('Fe', min_value=min_Fe, max_value=max_Fe, value=min_Fe, step=1.0)
        Cu = st.slider('Cu', min_value=min_Cu, max_value=max_Cu, value=min_Cu, step=1.0)
        Mn = st.slider('Mn', min_value=min_Mn, max_value=max_Mn, value=min_Mn, step=1.0)
        B = st.slider('B', min_value=min_B, max_value=max_B, value=min_B, step=0.01)

        if st.button("Predict your crop"):
            output = predict_crop(N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B)
            output_str = str(output).capitalize()
            if output_str == 0:
                st.success(f'The most suitable crop for your field is: This soil is not Good for cultivate')
            else:
                st.success(f'The most suitable crop for your field is: This soil is Good for cultivate')    

            

if __name__ == '__main__':
    main()
