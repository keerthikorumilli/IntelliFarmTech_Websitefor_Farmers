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
model = pickle.load(open('model.pkl', 'rb'))
df = pd.read_csv("crop_prediction_model_one.csv")

# Load your machine learning model and data
model2 = pickle.load(open('model2.pkl', 'rb'))
df2 = pd.read_csv("Soil.csv")

converts_dict = {
    'Nitrogen': 'N',
    'Phosphorus': 'P',
    'Potassium': 'K',
    'Temperature': 'temperature',
    'Humidity': 'humidity',
    'Rainfall': 'rainfall',
    'ph': 'ph'
}

def predict_crop(n, p, k, temperature, humidity, ph, rainfall):
    input = np.array([[n, p, k, temperature, humidity, ph, rainfall]]).astype(np.float64)
    prediction = model.predict(input)
    return prediction[0]

def predict_soil(N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B):
    input = np.array([[N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B]]).astype(np.float64)
    prediction = model2.predict(input)
    return prediction[0]


def scatterPlotDrawer(x,y):
    fig = plt.figure(figsize=(20,15))
    sns.set_style("whitegrid")
    sns.scatterplot(data=df, x=x, y=y, hue="label", size="label", palette="deep", sizes=(20, 200), legend="full")
    plt.xlabel(x, fontsize=22)
    plt.ylabel(y, fontsize=22)
    plt.xticks(rotation=90, fontsize=18)
    plt.legend(prop={'size': 18})
    plt.yticks(fontsize=16)
    st.write(fig)

def scatterPlotDrawer2(x,y):
    fig = plt.figure(figsize=(20,15))
    sns.set_style("whitegrid")
    sns.scatterplot(data=df2, x=x, y=y, hue="label", size="label", palette="deep", sizes=(20, 200), legend="full")
    plt.xlabel(x, fontsize=22)
    plt.ylabel(y, fontsize=22)
    plt.xticks(rotation=90, fontsize=18)
    plt.legend(prop={'size': 18})
    plt.yticks(fontsize=16)
    st.write(fig)

def barPlotDrawer(x,y):
    fig = plt.figure(figsize=(20,15))
    sns.set_style("whitegrid")
    sns.barplot(data=df, x=x, y=y)
    plt.xlabel("Crops", fontsize=22)
    plt.ylabel(y, fontsize=22)
    plt.xticks(rotation=90, fontsize=18)
    plt.legend(prop={'size': 18})
    plt.yticks(fontsize=16)
    st.write(fig)

def barPlotDrawer2(x,y):
    fig = plt.figure(figsize=(20,15))
    sns.set_style("whitegrid")
    sns.barplot(data=df2, x=x, y=y)
    plt.xlabel("Crops", fontsize=22)
    plt.ylabel(y, fontsize=22)
    plt.xticks(rotation=90, fontsize=18)
    plt.legend(prop={'size': 18})
    plt.yticks(fontsize=16)
    st.write(fig)        

def boxPlotDrawer(x,y):
    fig = plt.figure(figsize=(20,15))
    sns.set_style("whitegrid")
    sns.boxplot(x=x, y=y, data=df)
    sns.despine(offset=10, trim=True)
    plt.xlabel("Crops", fontsize=22)
    plt.ylabel(y, fontsize=22)
    plt.xticks(rotation=90, fontsize=18)
    plt.legend(prop={'size': 18})
    plt.yticks(fontsize=16)
    st.write(fig)

def boxPlotDrawer2(x,y):
    fig = plt.figure(figsize=(20,15))
    sns.set_style("whitegrid")
    sns.boxplot(x=x, y=y, data=df2)
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
    <h2 style="color:white;text-align:center;"> Visualize crop Properties </h2>
    </div>
    """

    html_temp_pred = """
    <div style="padding:10px;margin-bottom:30px">
    <h2 style="color:white;text-align:center;"> Which Crop To Cultivate? </h2>
    </div>
    """

    html_temp_vis2 = """
    <div style="padding:10px;margin-bottom:30px">
    <h2 style="color:white;text-align:center;"> Visualize Soil Properties </h2>
    </div>
    """

    html_temp_pred2 = """
    <div style="padding:10px;margin-bottom:30px">
    <h2 style="color:white;text-align:center;"> Is soil good To Cultivate? </h2>
    </div>
    """

    st.sidebar.title("Select one")
    select_type = st.sidebar.radio("", ('crop prediction','soil prediction'))

    if select_type == 'crop prediction':
        st.sidebar.title("Select One")
        select_type = st.sidebar.radio("", ('Home','Data Info','visualize crop', 'Predict Your Crop'))
         

        if select_type == 'Home':
           st.title("IntelliFarmTech - For Farmers")
           st.image("./images/homepage.PNG")
           st.markdown(
    """<p style="font-size:20px;">
            IntelliFarm Tech is a precision farming solution designed to revolutionize modern agriculture by leveraging machine learning, Python, and Streamlit. The project aims to provide farmers with real-time, data-driven insights to enhance crop yield, optimize resource utilization, and foster sustainable farming practices.
        </p>
    """, unsafe_allow_html=True)



        if select_type == 'Data Info':
             # Add title to the page
            st.title("Data Info page")

            # Add subheader for the section
            st.subheader("View Data")

            # Create an expansion option to check the data
            with st.expander("View data"):
                st.dataframe(df)

            # Create a section to columns values
            # Give subheader
            st.subheader("Columns Description:")

            # Create a checkbox to get the summary.
            if st.checkbox("View Summary"):
                st.dataframe(df.describe())

            # Create multiple check box in row
            col_name, col_dtype, col_data = st.columns(3)

            # Show name of all dataframe
            with col_name:
                if st.checkbox("Column Names"):
                    st.dataframe(df.columns)

            # Show datatype of all columns 
            with col_dtype:
                if st.checkbox("Columns data types"):
                    dtypes = df.dtypes.apply(lambda x: x.name)
                    st.dataframe(dtypes)
            
            # Show data for each columns
            with col_data: 
                if st.checkbox("Columns Data"):
                    col = st.selectbox("Column Name", list(df.columns))
                    st.dataframe(df[col])

            # Add the link to you dataset
            st.markdown("""
                            <p style="font-size:24px">
                                <a 
                                    href="https://www.kaggle.com/uciml/pima-indians-diabetes-database"
                                    target=_blank
                                    style="text-decoration:none;"
                                >Get Dataset
                                </a> 
                            </p>
                        """, unsafe_allow_html=True
            )
        elif select_type == 'visualize crop':
            st.markdown(html_temp_vis, unsafe_allow_html=True)
            plot_type = st.selectbox("Select plot type", ('Bar Plot', 'Scatter Plot', 'Box Plot'))
            st.subheader("Relation between features")

            # Plot!
            x = ""
            y = ""

            if plot_type == 'Bar Plot':
                x = 'label'
                y = st.selectbox("Select a feature to compare between crops",
                    ('Phosphorus', 'Nitrogen', 'ph', 'Potassium', 'Temperature', 'Humidity', 'Rainfall'))
            if plot_type == 'Scatter Plot':
                x = st.selectbox("Select a property for 'X' axis",
                    ('Phosphorus', 'Nitrogen', 'ph', 'Potassium', 'Temperature', 'Humidity', 'Rainfall'))
                y = st.selectbox("Select a property for 'Y' axis",
                    ('Nitrogen', 'Phosphorus', 'ph', 'Potassium', 'Temperature', 'Humidity', 'Rainfall'))
            if plot_type == 'Box Plot':
                x = "label"
                y = st.selectbox("Select a feature",
                    ('Phosphorus', 'Nitrogen', 'ph', 'Potassium', 'Temperature', 'Humidity', 'Rainfall'))

            if st.button("Visualize"):
                if plot_type == 'Bar Plot':
                    y = converts_dict[y]
                    barPlotDrawer(x, y)
                if plot_type == 'Scatter Plot':
                    x = converts_dict[x]
                    y = converts_dict[y]
                    scatterPlotDrawer(x, y)
                if plot_type == 'Box Plot':
                    y = converts_dict[y]
                    boxPlotDrawer(x, y)

        if select_type == "Predict Your Crop":
            st.markdown(html_temp_pred, unsafe_allow_html=True)
            st.header("To predict your crop give values")
            st.subheader("Drag to Give Values")
            n = st.slider('Nitrogen', 0, 140)
            p = st.slider('Phosphorus', 5, 145)
            k = st.slider('Potassium', 5, 205)
            temperature = st.slider('Temperature', 8.83, 43.68)
            humidity = st.slider('Humidity', 14.26, 99.98)
            ph = st.slider('pH', 3.50, 9.94)
            rainfall = st.slider('Rainfall', 20.21, 298.56)
            
            if st.button("Predict your crop"):
                output=predict_crop(n, p, k, temperature, humidity, ph, rainfall)
                res = "“"+ output.capitalize() + "”"
                st.success('The most suitable crop for your field is {}'.format(res))
                

    elif select_type == 'soil prediction':
        st.sidebar.title("Select One")
        select_type = st.sidebar.radio("", ('Home','Data Info','visualize soil columns', 'Predict Your soil'))

        if select_type == 'Home':
           st.title("IntelliFarmTech - For Farmers")
           st.image("./images/homepage.PNG")
           st.markdown(
    """<p style="font-size:20px;">
            IntelliFarm Tech is a precision farming solution designed to revolutionize modern agriculture by leveraging machine learning, Python, and Streamlit. The project aims to provide farmers with real-time, data-driven insights to enhance crop yield, optimize resource utilization, and foster sustainable farming practices.
        </p>
    """, unsafe_allow_html=True)

        if select_type == 'Data Info':
             # Add title to the page
            st.title("Data Info page")

            # Add subheader for the section
            st.subheader("View Data")

            # Create an expansion option to check the data
            with st.expander("View data"):
                st.dataframe(df2)

            # Create a section to columns values
            # Give subheader
            st.subheader("Columns Description:")

            # Create a checkbox to get the summary.
            if st.checkbox("View Summary"):
                st.dataframe(df2.describe())

            # Create multiple check box in row
            col_name, col_dtype, col_data = st.columns(3)

            # Show name of all dataframe
            with col_name:
                if st.checkbox("Column Names"):
                    st.dataframe(df2.columns)

            # Show datatype of all columns 
            with col_dtype:
                if st.checkbox("Columns data types"):
                    dtypes = df2.dtypes.apply(lambda x: x.name)
                    st.dataframe(dtypes)
            
            # Show data for each columns
            with col_data: 
                if st.checkbox("Columns Data"):
                    col = st.selectbox("Column Name", list(df2.columns))
                    st.dataframe(df2[col])

            # Add the link to you dataset
            st.markdown("""
                            <p style="font-size:24px">
                                <a 
                                    href="https://www.kaggle.com/uciml/pima-indians-diabetes-database"
                                    target=_blank
                                    style="text-decoration:none;"
                                >Get Dataset
                                </a> 
                            </p>
                        """, unsafe_allow_html=True
            )
        elif select_type == 'visualize soil columns':
            st.markdown(html_temp_vis2, unsafe_allow_html=True)
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
                    barPlotDrawer2(x, y)
                if plot_type == 'Scatter Plot':
                    scatterPlotDrawer2(x, y)
                if plot_type == 'Box Plot':
                    boxPlotDrawer2(x, y)

        if select_type == "Predict Your soil":
            st.markdown(html_temp_pred2, unsafe_allow_html=True)
            st.header("To predict your crop, give values")
            st.subheader("Drag to Give Values")
            # Calculate the minimum and maximum values from the dataset
            min_N = float(df2['N'].min())
            max_N = float(df2['N'].max())
            min_P = float(df2['P'].min())
            max_P = float(df2['P'].max())
            min_K = float(df2['K'].min())
            max_K = float(df2['K'].max())
            min_pH = float(df2['pH'].min())
            max_pH = float(df2['pH'].max())
            min_EC = float(df2['EC'].min())
            max_EC = float(df2['EC'].max())
            min_OC = float(df2['OC'].min())
            max_OC = float(df2['OC'].max())
            min_S = float(df2['S'].min())   
            max_S = float(df2['S'].max())       
            min_Zn = float(df2['Zn'].min())
            max_Zn = float(df2['Zn'].max())
            min_Fe = float(df2['Fe'].min())
            max_Fe = float(df2['Fe'].max())
            min_Cu = float(df2['Cu'].min())
            max_Cu = float(df2['Cu'].max())
            min_Mn = float(df2['Mn'].min())
            max_Mn = float(df2['Mn'].max())
            min_B = float(df2['B'].min())
            max_B = float(df2['B'].max())

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
                output = predict_soil(N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B)
                output_str = str(output).capitalize()
                if output_str == 0:
                    st.success(f'The most suitable crop for your field is: This soil is not Good for cultivate')
                else:
                    st.success(f'The most suitable crop for your field is: This soil is Good for cultivate')


if __name__ == '__main__':
    main()