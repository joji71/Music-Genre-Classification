# Music Genre Classification Repository
- This repository contains a classification project to classify music based on genre.
- Used a model based hard voting  for classification. This Flask-based web application classifies the genre of an uploaded WAV file.

# Features
- Accepts WAV file as input.
- Extracts metadata from the audio file using `librosa`.
- Uses a pre-trained machine learning model for genre prediction.
- Provides genre to the user through an intuitive web interface.

# Files and Folders
- MGC.py: contains the code to run the local host web application and classify the genre of the uploaded wav file.
- templates: contains the code of the web pages to upload the audio file and view the corresponding genre.
- test.pynb: contains the code and data related to the Music Genre Classification project.
- README.md: provides an overview of the repository.

# Getting Started
To run the code in this repository, you need to have Python installed along with the following libraries: numpy, pandas, pickle, librosa, FLASK . Run MGC.py and will get the link to the web application on local host.  

# Conclusion
Overall, the classification models were able to predict the music genre classification with reasonable accuracy. The FLASK library was used to deploy the models as a web app.

Thank you for visiting our Music Genre Classification repository!
