import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# File IDs for Google Drive
FILE_IDS = {
    'jobs.csv': '1ZiG8R9LeoWhE639buSzJoQGRjBJ7cDzg',
    'processed_jobs.csv': '1AueURcaYr7pYDumBwWmcbEtoHh_oTnQL',
    'vectors.pkl': '1gLrEiYgOqlM76TJs2g6vp7409VT71ecc'
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

# Load the dataset
df = pd.read_csv('jobs.csv')  # Already downloaded or present locally

# Combine relevant columns into a single 'combined' column
df['combined'] = (
    df['Job Title'].astype(str) + " " +
    df['Role'].astype(str) + " " +
    df['skills'].astype(str) + " " +
    df['Company'].astype(str) + " " +
    df['Qualifications'].astype(str) + " " +
    df['Work Type'].astype(str)
)

# Handle missing values in the 'combined' column
df['combined'] = df['combined'].fillna('')

# Use the 'combined' column for recommendations
documents = df['combined'].tolist()

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(documents)

# Get user input for job recommendation
user_input = input("Enter your skills, job role, company preference, or qualification: ")
user_vector = vectorizer.transform([user_input])

# Calculate similarity between user input and job postings
similarity_scores = cosine_similarity(user_vector, vectors)

# Add similarity scores to the DataFrame
df['similarity'] = similarity_scores[0]

# Get the top 5 job recommendations
recommendations = df.sort_values(by='similarity', ascending=False).head(5)

# Display recommendations
print("\nTop Job Recommendations:")
print(recommendations[['Job Title', 'Role', 'skills', 'Company', 'Qualifications', 'Work Type']])

# Save the TF-IDF vectorizer
with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

# Save the job vectors
with open('vectors.pkl', 'wb') as file:
    pickle.dump(vectors, file)

# Save the dataset
df.to_csv('processed_jobs.csv', index=False)
