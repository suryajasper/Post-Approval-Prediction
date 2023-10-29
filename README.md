# Post-Approval-Prediction
Project for TAMU Datathon 2023. Predicts a user's engagement with social media posts based on Marky survey data

# How to use
## Networks
Networks can be found in `networks/` folder with instructions on how to configure and use them. `training_loop.py` already implements a comprehensive pipeline with several configurable parameters.
- **Structural Embedding Network *(ours)***: 4D Convolutional Neural Network leading to a feed-forward linear network that produces vector embeddings for images. Images are downscaled by pooling layers to 64x64x3 resolution before convolutional network to force the network to prioritize understanding structure, color scheme, and layout over fine details
- **Google OCR**: We implement pretrained google cloud vision model to find text in post images. These text inputs are tokenized and converted into embeddings using the pretrained BERT model
- **BERT Tokenizer**: tokenizes sentences and produces vector embeddings through a forward pass of pretrained BERT network
- **Self-Attention Network *(ours)***: Merges and normalizes collection of text and image embeddings into a single comphrensive post vector embedding. Learns contextual relationships between text data and image layout.
- **Classification Network *(ours)***: Learns binary classification of post approval based on post embeddings from attention network. Loss of classification network is backpropagated to optimize itself, the attention network, and the image embedding network. 
## Preprocessing and Data Collection
The images corresponding to each post can be downloaded with the load_data.py file found under the data_processing directory.
**Note:** for some users this process may take an extended period of time. A smaller sample of the dataset can be downloaded with the load_small_data.py file.

After downloading the image data, the text within each image needs to be parsed using the Google Vision OCR. Run data_processing/image_text to assign text for each image and place it in a csv for use by the model. Tokenization is automatically handled by our custom dataset loader.

To ensure that the convolutional neural network learns only important features of the image data, run data_processing/shrink_images.py to reduce the image sizes. Defaulting to convolutional downscaling in structural embedding network will marginally introduce nuance at the expense of higher computational costs.

## Running the model

**Work in progress**

Full writeup can be found on devpost
