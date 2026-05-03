import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Job Description
job_description = """
We are hiring a Junior Data Analyst.

Responsibilities:
Analyze large datasets using Python.
Work with Pandas, NumPy, and SQL.
Create data visualization and reports.
Perform basic machine learning tasks.
Communicate insights to the team.

Required Skills:
Python, Pandas, NumPy, SQL,
Data Analysis, Data Visualization,
Machine Learning



"""

# Read resumes
resume_texts = []
resume_names = []

resume_folder = "resumes"

for file in os.listdir(resume_folder):
    with open(os.path.join(resume_folder, file), 'r') as f:
        resume_texts.append(f.read())
        resume_names.append(file)

# Combine JD and resumes
documents = [job_description] + resume_texts

# Vectorization
vectorizer = CountVectorizer()
vectors = vectorizer.fit_transform(documents)

# Cosine Similarity
similarity_scores = cosine_similarity(vectors[0:1], vectors[1:])[0]

# Display results
print("ATS Resume Matching Results:\n")
for i in range(len(resume_names)):
    score = round(similarity_scores[i] * 100, 2)
    print(f"{resume_names[i]} → Match Score: {score}%")
