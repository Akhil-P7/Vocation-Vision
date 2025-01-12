from flask import Flask, render_template, request
import pandas as pd
import pickle
import os
import requests
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Google Drive File IDs
FILE_IDS = {
    'vectorizer.pkl': 'your_vectorizer_pkl_file_id',
    'vectors.pkl': '1gLrEiYgOqlM76TJs2g6vp7409VT71ecc',
    'processed_jobs.csv': '1AueURcaYr7pYDumBwWmcbEtoHh_oTnQL'
}

# Function to download files from Google Drive
def download_file_from_google_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

# Ensure required files are downloaded
for file_name, file_id in FILE_IDS.items():
    if not os.path.exists(file_name):
        print(f"{file_name} is missing. Downloading...")
        download_file_from_google_drive(file_id, file_name)
        print(f"{file_name} downloaded.")

# Load the saved vectorizer, job vectors, and dataset
try:
    with open('vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
except FileNotFoundError:
    raise FileNotFoundError("The file 'vectorizer.pkl' was not found. Ensure it is in the working directory.")

try:
    with open('vectors.pkl', 'rb') as file:
        job_vectors = pickle.load(file)
except FileNotFoundError:
    raise FileNotFoundError("The file 'vectors.pkl' was not found. Ensure it is in the working directory.")

try:
    jobs_df = pd.read_csv('processed_jobs.csv')
except FileNotFoundError:
    raise FileNotFoundError("The file 'processed_jobs.csv' was not found. Ensure it is in the working directory.")

@app.route('/')
def index():
    """Render the home page with the job search form."""
    return render_template('index.html')

@app.route('/results', methods=['POST'])
def results():
    """Handle form submission and return job recommendations."""
    try:
        # Get user input from the form
        skills = request.form.get('skills', '').strip()
        job_role = request.form.get('job_role', '').strip()
        company_preference = request.form.get('company_preference', '').strip()
        qualification = request.form.get('qualification', '').strip()

        # Validate input
        if not skills and not job_role and not company_preference and not qualification:
            return render_template('results.html', error="Please provide at least one search criterion.")

        # Combine user inputs for the recommendation system
        user_input = f"{skills} {job_role} {company_preference} {qualification}".strip()
        user_vector = vectorizer.transform([user_input])

        # Calculate similarity scores
        similarity_scores = cosine_similarity(user_vector, job_vectors)

        # Add similarity scores to the DataFrame
        jobs_df['similarity'] = similarity_scores[0]

        # Get the top 5 recommendations
        recommendations = jobs_df.sort_values(by='similarity', ascending=False).head(5)

        # Check if recommendations are empty
        if recommendations.empty:
            return render_template('results.html', error="No job recommendations found. Try different search criteria.")

        # Pass recommendations to the results page
        return render_template('results.html', recommendations=recommendations.to_dict('records'))

    except Exception as e:
        return render_template('results.html', error=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
