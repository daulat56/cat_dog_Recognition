import numpy as np
from PIL import Image
import streamlit as st
from skimage.io import imread
from skimage.transform import resize
import pickle
st.title('Image classifier using machine learning')
st.text('upload the image')
model=pickle.load(open('image.p','rb'))

uploaded_file=st.file_uploader("choose an image...",type="jpg")
if uploaded_file is not None:
    img=Image.open(uploaded_file)
    st.image(img,caption="Uploaded Image")
    if st.button('PREDICT'):
        CATAGORIES=['dog','cat']
        st.write('Result..')
        flat_data=[]
        img=np.array(img)
        img_resized=resize(img,(150,150,3))
        flat_data.append(img_resized.flatten())
        flat_data=np.array(flat_data)
        y_out=model.predict(flat_data)
        y_out=CATAGORIES[y_out[0]]
        st.write(' PREDICTED OUTPUT -:', y_out)
        q=model.predict_proba(flat_data)
        for index ,item in enumerate(CATAGORIES):
            st.write(f'{item}:{q[0][index]*100}')
