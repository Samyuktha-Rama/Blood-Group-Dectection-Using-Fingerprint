Fingerprint-Based Blood Group Detection System

A deep learning–based Flask web application that predicts a person’s blood group from their fingerprint image.

Features
• Fingerprint Verification:
Validates that the uploaded image is a fingerprint using OpenCV techniques such as edge detection, variance, and frequency analysis.
• Blood Group Prediction:
A trained CNN model classifies the fingerprint into one of eight groups:
A+, A-, B+, B-, AB+, AB-, O+, O-.
• User-Friendly Web Interface:
A simple Flask-based web app for image upload and prediction display.
• Cloud Model Hosting:
The trained model is hosted on Hugging Face Hub and dynamically loaded at runtime.

Project Structure
.
├── app.py                  
├── train.py                
├── is_fingerprint.py       
├── requirements.txt        
├── Procfile                
└── templates/
    └── index.html     

Setup and Installation
Prerequisites
•	Python 3.x
•	Kaggle API credentials (for training via train.py)
•	Hugging Face API token (for model hosting)
Local Setup
1.	Clone the Repository:
2.	git clone <Samyuktha-Rama/Blood-Group-Dectection-Using-Fingerprint>
3.	cd fingerprint-blood-group-detection
4.	Create a Virtual Environment:
5.	python -m venv venv
6.	source venv/bin/activate       # On Windows: venv\Scripts\activate
7.	Install Dependencies:
8.	pip install -r requirements.txt

Running the Application
1.	Ensure the model file (blood_group_model.h5) is available locally
— or modify app.py to download it from Hugging Face at runtime.
2.	Start the Flask server:
3.	python app.py
4.	Access the app:
Open your browser and visit http://127.0.0.1:5000

Dataset
The project uses the Fingerprint-Based Blood Group Dataset available on Kaggle.
The train.py script automatically downloads and preprocesses this dataset.

Technologies Used
• Backend: Python, Flask, Gunicorn
• Machine Learning: TensorFlow/Keras, OpenCV, NumPy
• Model Hosting: Hugging Face Hub
• Deployment: Render
• Frontend: HTML, CSS, JavaScript



