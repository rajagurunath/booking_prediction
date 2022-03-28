import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import pandas as pd
import lux
import missingno


@st.cache
def get_data():
    df = pd.read_csv("data/hotel_booking.zip")
    return df

def app():
    st.title('EDA Visualization of Hotel Booking Analaysis')
    
    st.write("Missing values analysis")
    
    df= get_data()

    st.pyplot(missingno.heatmap(df).figure)
    st.pyplot(missingno.matrix(df).figure)

    
    st.write('General Visualizations')
    html_content = df.save_as_html(output=True)
    components.html(html_content, width=800, height=350)



    st.title(f"Analysing the columns using target as `is_canceled` ")
    df.intent = ["is_canceled"]
    html_content = df.save_as_html(output=True)
    components.html(html_content, width=800, height=350)


app()
