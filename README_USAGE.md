# Resume Screening App - Quick Start Guide

## Overview
The Resume Screening App is a Streamlit application that uses NLP and semantic similarity to match candidate resumes with job descriptions. It has been **thoroughly debugged and tested**.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Install Dependencies
```bash
pip install -r requirements_resume.txt
```

### Step 2: Download Spacy Language Model
```bash
python -m spacy download en_core_web_sm
```

## Running the Application

```bash
streamlit run resume_screening_app.py
```

The app will open in your browser at `http://localhost:8501`

## How to Use

### 1. Prepare Your Resume Files
- Save all candidate resumes in PDF or DOCX format
- Place them in a folder (can include subdirectories)
- Create a ZIP file of the folder

### 2. Using the App
1. **Enter Job Description**: Paste or type the job description in the text area
2. **Upload ZIP File**: Upload your ZIP file containing resumes
3. **Set Number of Results**: Choose how many top candidates to display (default: 10)
4. **Click Analyze**: Process the resumes

### 3. Review Results
The app will display:
- **Name**: Resume filename
- **Score**: Similarity score (0-100) based on semantic matching
- **Skills Matched**: Skills found in both job description and resume
- **Missing Skills**: Required skills not found in resume
- **Extra Skills**: Additional skills in resume not in job description
- **Summary**: Complete overview of skill matching

### 4. Download Results
Click "Download CSV" to save the results as a spreadsheet.

## Example Job Description

```
We are looking for a Senior Software Engineer with expertise in:
- Python programming
- Machine learning and deep learning
- Experience with TensorFlow or PyTorch
- Cloud platforms (AWS, GCP, or Azure)
- Docker and Kubernetes
- Strong problem-solving skills
- 5+ years of experience
```

## Features

### ✅ Fixed Issues
All critical bugs have been fixed:
- Handles Streamlit UploadedFile objects correctly
- Robust error handling for corrupted files
- Validates empty inputs and missing data
- Supports nested directories in ZIP files
- Graceful model loading with helpful error messages

### Key Capabilities
- **Semantic Matching**: Uses sentence transformers for deep semantic similarity
- **Skill Extraction**: Automatically identifies skills using NLP
- **Skill Gap Analysis**: Shows what skills match, are missing, or are extra
- **Batch Processing**: Process multiple resumes at once
- **Easy Export**: Download results as CSV

## Troubleshooting

### Issue: "Spacy model not found"
**Solution**: Run `python -m spacy download en_core_web_sm`

### Issue: "No PDF or DOCX files found"
**Solution**: Ensure your ZIP file contains PDF or DOCX files. Check that files have correct extensions.

### Issue: "No valid text could be extracted"
**Solution**: 
- Check if PDFs are scanned images (app requires text-based PDFs)
- Ensure DOCX files are not corrupted
- Try opening files manually to verify they contain text

### Issue: Slow processing
**Solution**: 
- The first run downloads ML models (~100MB) - this is normal
- Subsequent runs use cached models and are faster
- Large numbers of resumes take longer to process

## Technical Details

### Technologies Used
- **Streamlit**: Web interface
- **Spacy**: Named entity recognition and NLP
- **Sentence Transformers**: Semantic similarity matching
- **PyPDF2**: PDF text extraction
- **docx2txt**: DOCX text extraction
- **Pandas**: Data processing and CSV export

### How Scoring Works
1. **Text Extraction**: Extracts text from each resume
2. **Skill Identification**: Uses NLP to identify entities and proper nouns as potential skills
3. **Semantic Encoding**: Converts job description and resumes to vector embeddings
4. **Similarity Calculation**: Computes cosine similarity between vectors
5. **Ranking**: Sorts candidates by similarity score

### Performance
- Processes ~10 resumes in 5-10 seconds (after initial model load)
- Supports up to 100 resumes per batch
- Memory efficient with caching

## File Structure

```
resume_screening_app.py       # Main application (fixed version)
requirements_resume.txt        # Dependencies
RESUME_APP_BUGS.md            # Detailed bug documentation
COMPARISON.md                  # Before/after code comparison
verify_fixes.py                # Test script for bug fixes
test_resume_app.py            # Unit tests
README_USAGE.md               # This file
```

## Best Practices

### For Best Results:
1. **Write detailed job descriptions** with specific skills and requirements
2. **Use consistent terminology** (e.g., "ML" vs "Machine Learning")
3. **Include technical skills** explicitly in job descriptions
4. **Use text-based PDFs** (not scanned images)
5. **Name resume files** clearly (e.g., "John_Smith_Resume.pdf")

### What to Look For:
- High scores (>70) indicate strong semantic match
- Check "Missing Skills" to identify training needs
- "Extra Skills" can reveal hidden talents
- Review actual resumes for top candidates (automated scoring isn't perfect)

## Support

For issues, refer to:
- `RESUME_APP_BUGS.md` - Complete bug fix documentation
- `COMPARISON.md` - Code changes and fixes
- `verify_fixes.py` - Run to verify all fixes are working

## License
This code is provided as-is for educational and commercial use.
