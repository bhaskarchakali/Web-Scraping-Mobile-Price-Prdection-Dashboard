import streamlit as st
import pickle
import numpy as np

# import the model
pipe = pickle.load(open('interview project model.pkl', 'rb'))
df = pickle.load(open('interview project df.pkl', 'rb'))


def main():
    string = "Price Predictor"
    st.set_page_config(page_title=string, page_icon="📱")
    st.title("Mobile Price Predictor")
    st.image(
        "https://dictionary.cambridge.org/images/thumb/mobile_noun_002_23642.jpg?version=5.0.247",
        width=170 # Manually Adjust the width of the image as per requirement
    )
    st.write('')
    st.write('')

    Brand = st.selectbox('Mobile Company', df['BRAND'].unique())
    Product_Name = st.selectbox('Product Name', df['PRODUCT_NAME'].unique())
    Colour = st.selectbox('Colour', df['COLOR'].unique())
    Memory = st.selectbox('Memory (ROM) in GB', df['MEMORY'].unique())
    Starrating = st.selectbox('Rating given by customers', [1,2,2.5,3,3.5,3.8,4,4.2,4.3,4.5,4.6,4.8,5])
    Noofratings = st.selectbox('Select no of ratings given by customers',df['RATINGS'].unique() )
    noofreviews = st.selectbox('Select no of Reviews given by customers',df['REVIEWS'].unique() )
    RAM = st.selectbox('RAM in GB', df['RAM'].unique())
    Battery = st.selectbox('Battery in mah', df['BATTERY_C'].unique())
    Display_Size = st.selectbox('Display Size in inches', df['SCREEN_SIZE'].unique())
    
    if st.button('Predict Price'):
        query = np.array(
            [ Brand,Product_Name,Colour,Memory,Starrating,Noofratings,noofreviews,RAM,Battery,Display_Size])

        query = query.reshape(1, 10)
        st.title("The predicted price  " + str(int(np.exp(pipe.predict(query)[0]))) + 'Rupees')


if __name__ == '__main__':
    main()
