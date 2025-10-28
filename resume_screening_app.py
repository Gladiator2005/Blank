import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import zipfile
import PyPDF2
import docx2txt
import spacy
from sentence_transformers import SentenceTransformer, util

# Load models once and cache
@st.cache_resource
def load_models():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        st.error("Spacy model 'en_core_web_sm' not found. Please install it using: python -m spacy download en_core_web_sm")
        st.stop()
    
    try:
        sbert = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Failed to load SentenceTransformer model: {str(e)}")
        st.stop()
    
    return nlp, sbert

nlp, sbert = load_models()

def extract_text(file_path):
    """Extract text from PDF or DOCX files."""
    try:
        if file_path.endswith('.pdf'):
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                return ''.join(page.extract_text() or '' for page in reader.pages)
        elif file_path.endswith('.docx'):
            return docx2txt.process(file_path)
    except Exception as e:
        st.warning(f"Error extracting text from {os.path.basename(file_path)}: {str(e)}")
    return ''

def extract_skills(text):
    """Extract skills from text using NLP."""
    if not text or not text.strip():
        return []
    
    try:
        doc = nlp(text)
        # Heuristic: collect proper nouns and ORG/GPE/PERSON entities as potential skills
        skills = set()
        for ent in doc.ents:
            if ent.label_ in {"ORG", "GPE", "PERSON"}:
                skills.add(ent.text)
        for token in doc:
            if token.pos_ == "PROPN":
                skills.add(token.text)
        return [s.strip() for s in skills if len(s) > 2]
    except Exception as e:
        st.warning(f"Error extracting skills: {str(e)}")
        return []

def skill_gap_analysis(candidate_skills, required_skills):
    """Analyze skill gaps between candidate and job requirements."""
    matched = list(set(candidate_skills).intersection(required_skills))
    missing = list(set(required_skills) - set(candidate_skills))
    extra = list(set(candidate_skills) - set(required_skills))
    return matched, missing, extra

def get_resumes_from_zip(uploaded_file):
    """Extract resumes from uploaded ZIP file."""
    folder = tempfile.mkdtemp()
    
    try:
        # Save the uploaded file to a temporary location
        temp_zip_path = os.path.join(folder, "uploaded.zip")
        with open(temp_zip_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract the ZIP file
        with zipfile.ZipFile(temp_zip_path, 'r') as z:
            z.extractall(folder)
        
        # Find all PDF and DOCX files
        files = []
        for root, dirs, filenames in os.walk(folder):
            for filename in filenames:
                if filename.endswith(('.pdf', '.docx')):
                    files.append(os.path.join(root, filename))
        
        return files
    except Exception as e:
        st.error(f"Error processing ZIP file: {str(e)}")
        return []

st.title("🚀 Innovative Resume Screening App")

with st.form("main_form"):
    job_desc = st.text_area("Enter Job Description", height=150)
    uploaded_zip = st.file_uploader("Upload ZIP file with Resumes (PDF/DOCX)", type=["zip"])
    num_top = st.number_input("Number of candidates to display", min_value=1, max_value=100, value=10)
    submitted = st.form_submit_button("Analyze")

if submitted:
    # Validate inputs
    if not job_desc or not job_desc.strip():
        st.error("Please enter a job description.")
    elif not uploaded_zip:
        st.error("Please upload a ZIP file containing resumes.")
    else:
        with st.spinner("Extracting resumes and processing..."):
            try:
                # Extract resumes from ZIP
                resume_paths = get_resumes_from_zip(uploaded_zip)
                
                if not resume_paths:
                    st.error("No PDF or DOCX files found in the uploaded ZIP file.")
                    st.stop()
                
                # Extract text from resumes
                resumes_texts = [extract_text(p) for p in resume_paths]
                resumes_names = [os.path.basename(p) for p in resume_paths]
                
                # Filter out empty resumes
                valid_resumes = [(name, text, path) for name, text, path in zip(resumes_names, resumes_texts, resume_paths) if text.strip()]
                
                if not valid_resumes:
                    st.error("No valid text could be extracted from the resumes.")
                    st.stop()
                
                resumes_names = [item[0] for item in valid_resumes]
                resumes_texts = [item[1] for item in valid_resumes]
                
                # Extract required skills from job description
                req_skills = extract_skills(job_desc)
                
                if not req_skills:
                    st.warning("No skills could be extracted from the job description. The analysis will be based on semantic similarity only.")
                
                # Encode job description and resumes
                job_embedding = sbert.encode([job_desc], convert_to_tensor=True)
                resume_embeddings = sbert.encode(resumes_texts, convert_to_tensor=True, show_progress_bar=True)
                
                # Calculate similarity scores
                scores = util.cos_sim(job_embedding, resume_embeddings)[0].cpu().numpy()
                
                # Build results
                results = []
                for i, score in enumerate(scores):
                    candidate_skills = extract_skills(resumes_texts[i])
                    matched, missing, extra = skill_gap_analysis(candidate_skills, req_skills)
                    summary = f"Matched skills: {', '.join(matched) if matched else 'None'}; Missing skills: {', '.join(missing) if missing else 'None'}; Extra skills: {', '.join(extra) if extra else 'None'}"
                    results.append({
                        "Name": resumes_names[i],
                        "Score": round(float(score)*100, 2),
                        "Skills Matched": ", ".join(matched) if matched else "None",
                        "Missing Skills": ", ".join(missing) if missing else "None",
                        "Extra Skills": ", ".join(extra) if extra else "None",
                        "Summary": summary
                    })
                
                # Create DataFrame and sort by score
                df = pd.DataFrame(results).sort_values(by="Score", ascending=False).head(int(num_top))
                
                st.success("Analysis Complete")
                st.dataframe(df)
                
                # Download button with proper MIME type
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name="screening_results.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
