# Post-Approval-Prediction
Project for TAMU Datathon 2023. Predicts a user's engagement with social media posts based on Marky survey data

# How to use
## Preprocessing and Data Collection
The images corresponding to each post can be downloaded with the load_data.py file found under the data_processing directory.
**Note:** for some users this process may take an extended period of time. A smaller sample of the dataset can be downloaded with the load_small_data.py file.

After downloading the image data, the text within each image needs to be parsed using the Google Vision OCR. Run data_processing/image_text to assign text for each image and place it in a csv for use by the model.

To ensure that the convolutional neural network learns only important features of the image data, run data_processing/shrink_images.py to reduce the image sizes.

## Running the model

**Work in progress**

Full writeup can be found on devpost
