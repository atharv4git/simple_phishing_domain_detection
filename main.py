import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import robust_scale as rob
import streamlit as st

# Load the trained model
model = tf.keras.models.load_model('./model1.h5')

def main():
    st.title('Phishing Website Detection')

    st.image("pics/meme1.jpeg")

    directory_length = st.number_input('directory_length', value=0)
    qty_slash_url = st.number_input('qty_slash_url', value=0)
    qty_dot_directory = st.number_input('qty_dot_directory', value=0)
    file_length = st.number_input('file_length', value=0)
    qty_hyphen_directory = st.number_input('qty_hyphen_directory', value=0)
    qty_percent_file = st.number_input('qty_percent_file', value=0)
    qty_hyphen_file = st.number_input('qty_hyphen_file', value=0)
    qty_underline_directory = st.number_input('qty_underline_directory', value=0)

    avg_values = pd.read_csv("phishing_avg_values.csv")
    avg_values.drop(columns=["Unnamed: 0"], inplace=True)


    data = []
    features = ["directory_length", "qty_slash_url", "qty_dot_directory", "file_length", "qty_hyphen_directory",
                "qty_percent_file", "qty_hyphen_file", "qty_underline_directory"]

    # Create a button to trigger the prediction
    if st.button('Predict'):
        input_data = np.array([directory_length, qty_slash_url, qty_dot_directory, file_length, qty_hyphen_directory,qty_percent_file, qty_hyphen_file, qty_underline_directory])

        j = 0
        for i in input_data:
            avg_values.loc[avg_values["feature"] == features[j], "avg_value"] = i

            j += 1

        input_data = rob(avg_values["avg_value"].values.reshape(1, -1))[:, :111]
        predictions = model.predict(input_data)

        prediction_text = 'Phishing' if predictions[0] > 0.5 else 'Legitimate'

        # Display the prediction result
        st.subheader('Prediction:')
        st.write(prediction_text)

if __name__ == '__main__':
    main()