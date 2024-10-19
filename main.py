import PyPDF2
import re
import nltk
from nltk.corpus import stopwords
import spacy
from fuzzywuzzy import process
import pandas as pd

nltk.download('stopwords')

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text

# Preprocess text (removes special characters and digits)
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and digits
    text = text.lower()  # Convert to lowercase
    stop_words = set(stopwords.words('english'))
    words = text.split()
    cleaned_text = ' '.join([word for word in words if word not in stop_words])
    return cleaned_text

# # Extract named entities (skills)
# nlp = spacy.load('en_core_web_sm')
# def extract_entities(text):
#     doc = nlp(text)
#     skills = []
#     for entity in doc.ents:
#         if entity.label_ == 'ORG':  # Modify this based on relevant entities
#             skills.append(entity.text)
#     return skills

# Load the dataset and remove duplicates for unique job profiles
df = pd.read_csv('roles-based-on-skills.csv')
df.drop_duplicates(subset=['Target'], inplace=True)  
df = df.dropna(subset=['ALL'])  

# Convert skills column to a dictionary of {job_profile: [skills]}
job_profiles = {}
for index, row in df.iterrows():
    job_profiles[row['Target']] = row['ALL'].split() 



# Fuzzy match function for skills
def fuzzy_match_skills(resume_text, skills_list):
    words = resume_text.split()
    matched_skills = []
    for word in words:
        
        best_match = process.extractOne(word, skills_list)
        if best_match and best_match[1] > 85:  
            matched_skills.append(best_match[0])
    return list(set(matched_skills))  

# Main function to process resume based on chosen job profile
def process_resume_for_job_profile(pdf_path, job_profile):
    resume_text = extract_text_from_pdf(pdf_path)
    cleaned_text = preprocess_text(resume_text)
    
    if job_profile not in job_profiles:
        print(f"Job profile '{job_profile}' not found.")
        return
    
    # Get the skills associated with the selected job profile
    skills_list = job_profiles[job_profile]
    
    
    extracted_skills = fuzzy_match_skills(cleaned_text, skills_list)
    print(f"Fuzzy Matched Skills for {job_profile}: {extracted_skills}")

chosen_job_profile = "Machine Learning Engineer"  
process_resume_for_job_profile('resume.pdf', chosen_job_profile)
