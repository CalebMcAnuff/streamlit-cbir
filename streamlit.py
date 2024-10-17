"""
# My first app
"""

import streamlit as st
import pandas as pd
from io import StringIO
from PIL import Image
from pytorch_VGG_1 import VGGNet
import numpy as np
import h5py
from scipy.spatial import distance
import pandas as pd

st.title("Content Based Image Retrieval")
st.markdown("""
This project implements a Content-Based Image Retrieval (CBIR) system using a pre-trained VGGNet model. CBIR allows users to search for images based on their visual content rather than metadata or tags. Given a query image, the system retrieves visually similar images from a dataset.

**Model Description**

The system utilizes the VGGNet model (specifically, VGG16) for feature extraction. The VGG16 network is pre-trained on the ImageNet dataset and is fine-tuned for content-based image retrieval. The visual features of the images in the dataset are extracted and stored as feature vectors in an HDF5 file. When a user uploads a query image, the system extracts its features, computes the cosine similarity between the query and the indexed images, and retrieves the most visually similar images.

**Features**

Upload an image for querying.
Retrieve the most similar images based on visual features.
Adjust the number of similar images returned (1-5).
""")
uploaded_file = st.file_uploader("Choose a file")
option = st.selectbox('How many similar images would you like to recieve?',(1,2,3,4,5))
if uploaded_file is not None:
    # To read file as bytes:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img,caption = "Uploaded image")


    # Load the indexed features and image names from the HDF5 file
    h5f = h5py.File("VGG16Features_pytorch.h5", 'r')
    feats = h5f['dataset_1'][:]
    imgNames = h5f['dataset_2'][:]
    h5f.close()

    # Define the query image
    queryImg = uploaded_file # "./fashion-dataset/fashion-dataset/images/46885.jpg"
    with st.spinner("Searching for similar images"):

        # Initialize the VGGNet model
        model = VGGNet()

        # Extract features from the query image
        X = model.extract_feat(queryImg)

        # Compute the cosine distance between the query image and all indexed images
        scores = []
        for feat in feats:
            score = 1 - distance.cosine(X, feat)
            scores.append(score)
        scores = np.array(scores)

        # Rank images by similarity score (highest to lowest)
        rank_ID = np.argsort(scores)[::-1]
        rank_score = scores[rank_ID]

        # Get top 3 matches
        maxres = option # 3
        imlist = [imgNames[index] for index in rank_ID[:maxres]]
        print(f"Top {maxres} images in order are: {imlist}")
        st.write(imlist)
        for i in imlist:
            img = Image.open(i.decode("utf-8")).convert("RGB")
            st.image(img,caption = f"{i}")
        # df_styles = pd.read_csv("./fashion-dataset/fashion-dataset/styles.csv")
        # for i in imlist:
        #    print([df_styles.id==i])

