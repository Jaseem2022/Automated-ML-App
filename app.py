import streamlit as st
import pandas as pd
import os

#importing related to profiling
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

#importing related to ML stuff

from pycaret.regression import setup,compare_models,pull,save_model

#add a sidebar
with st.sidebar:
    st.image("https://www.google.com/url?sa=i&url=https%3A%2F%2Fdeveloper.apple.com%2Fmachine-learning%2Fcreate-ml%2F&psig=AOvVaw2WctK7XCdIKHJKJkPTMAs_&ust=1712769330527000&source=images&cd=vfe&opi=89978449&ved=0CBIQjRxqFwoTCMCY3MPStYUDFQAAAAAdAAAAABAE")
    st.title("AutoStreamML")
    choice = st.radio("Navigation", ["Upload","Profiling","ML","Download"])
    st.info("This Website Allows you to build automated ML pipeline using StreamLit, Pandas Profiling and PyCaret and it is straight up BUZZZzzziiiing")

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None) #to use the csv file wherever we need


if choice == "Upload":
    st.title("Upload your data for modelling!")
    file = st.file_uploader("Upload your dataset here")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv",index = None) #save the uploaded file as sourcedata.csv
        #render the dataframe to website
        st.dataframe(df)

if choice == "Profiling":
    st.title("Automated Exploratary Data Analysis")
    profile_report = df.profile_report()
    st_profile_report(profile_report)
if choice == "ML":
    st.title("Machine Learning")
    target = st.selectbox("Select Your Target",df.columns)
    if st.button("Train Model"):
        setup(df,target=target,silent = True)
        setup_df = pull()
        st.info("This is the ML Experiment settings")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("This is the ML Model")
        st.dataframe(compare_df)
        save_model(best_model, "best_model")

if choice == "Download":
    with("best_model.pkl", "rb") as f:
        st.download_button("Download the file",f,"train_model.pkl")

