# Content-Based Image Retrieval (CBIR) using VGGNet

This project implements a Content-Based Image Retrieval (CBIR) system using a pre-trained VGGNet model. CBIR allows users to search for images based on their visual content rather than metadata or tags. Given a query image, the system retrieves visually similar images from a dataset.

## Model Description

The system utilizes the VGGNet model (specifically, VGG16) for feature extraction. The VGG16 network is pre-trained on the ImageNet dataset and is fine-tuned for content-based image retrieval. The visual features of the images in the dataset are extracted and stored as feature vectors in an HDF5 file. When a user uploads a query image, the system extracts its features, computes the cosine similarity between the query and the indexed images, and retrieves the most visually similar images.

## Features

Upload an image for querying.
Retrieve the most similar images based on visual features.
Adjust the number of similar images returned (1-5).
Instructions to Run the App Locally
## Prerequisites

Install Python 3.8+.
Install the required Python packages:

pip install streamlit pandas pillow h5py numpy scipy

## Running the Streamlit App
Clone this repository:
streamlit run streamlit.py

Upload an image file and select how many similar images you'd like to retrieve. The system will then display the top similar images based on the selected option.
### HDF5 File
The VGG16Features_pytorch.h5 file contains precomputed features of the images in the dataset. It is used for efficient image retrieval based on cosine similarity. Ensure that this file is in the same directory as the script.

### Model (VGGNet)
The VGGNet is imported from pytorch_VGG_1.py, which contains the necessary code to load the model and extract features from images.

### Usage
When the user uploads an image, the system displays the uploaded image along with the selected number of similar images:
