import streamlit as st
from generate_instructions import generate_tuning_and_learning_instructions
from instrument_tuning import tune_instrument_ui
from learn_music_playing import learn_music_playing_ui

# Set up Streamlit App
st.set_page_config(page_title="Learn and Tune Musical Instruments with PiTunes", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page:", ["Generate Instructions", "Tune an Instrument", "Learn Music by Playing"])

# Page 1: Generate Tuning and Music Learning Instructions
if page == "Generate Instructions":
    st.title("Generate Instrument Tuning & Music Learning Instructions")
    generate_tuning_and_learning_instructions()

# Page 2: Tune an Instrument
elif page == "Tune an Instrument":
    st.title("Tune an Instrument")
    tune_instrument_ui()

# Page 3: Learn Music by Playing
elif page == "Learn Music by Playing":
    st.title("Learn Music by Playing")
    learn_music_playing_ui()
