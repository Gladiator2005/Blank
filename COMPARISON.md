# Resume Screening App - Before and After Comparison

## Original Code Issues vs Fixed Code

### Issue 1: ZIP File Handling

**ORIGINAL CODE (BROKEN):**
```python
def get_resumes_from_zip(zip_file):
    folder = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_file, 'r') as z:  # ❌ CRASHES: zip_file is UploadedFile, not path
        z.extractall(folder)
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.pdf', '.docx'))]
    return files
```

**FIXED CODE:**
```python
def get_resumes_from_zip(uploaded_file):
    folder = tempfile.mkdtemp()
    try:
        # ✅ FIX: Save UploadedFile to disk first
        temp_zip_path = os.path.join(folder, "uploaded.zip")
        with open(temp_zip_path, "wb") as f:
            f.write(uploaded_file.getbuffer())  # ✅ Use getbuffer() for UploadedFile
        
        with zipfile.ZipFile(temp_zip_path, 'r') as z:
            z.extractall(folder)
        
        # ✅ FIX: Use os.walk() for recursive search
        files = []
        for root, dirs, filenames in os.walk(folder):
            for filename in filenames:
                if filename.endswith(('.pdf', '.docx')):
                    files.append(os.path.join(root, filename))
        return files
    except Exception as e:
        st.error(f"Error processing ZIP file: {str(e)}")  # ✅ FIX: Error handling
        return []
```

### Issue 2: Model Loading

**ORIGINAL CODE (MISSING ERROR HANDLING):**
```python
@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")  # ❌ Crashes if model not installed
    sbert = SentenceTransformer("all-MiniLM-L6-v2")  # ❌ No error handling
    return nlp, sbert
```

**FIXED CODE:**
```python
@st.cache_resource
def load_models():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # ✅ FIX: User-friendly error with instructions
        st.error("Spacy model 'en_core_web_sm' not found. Please install it using: python -m spacy download en_core_web_sm")
        st.stop()
    
    try:
        sbert = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        # ✅ FIX: Catch and display errors
        st.error(f"Failed to load SentenceTransformer model: {str(e)}")
        st.stop()
    
    return nlp, sbert
```

### Issue 3: Extract Text Function

**ORIGINAL CODE (NO ERROR HANDLING):**
```python
def extract_text(file_path):
    if file_path.endswith('.pdf'):
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)  # ❌ Can crash on corrupted PDFs
            return ''.join(page.extract_text() or '' for page in reader.pages)
    elif file_path.endswith('.docx'):
        return docx2txt.process(file_path)  # ❌ Can crash on corrupted files
    return ''
```

**FIXED CODE:**
```python
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
        # ✅ FIX: Warn but don't crash
        st.warning(f"Error extracting text from {os.path.basename(file_path)}: {str(e)}")
    return ''
```

### Issue 4: Download Button

**ORIGINAL CODE (INCOMPLETE):**
```python
st.download_button("Download CSV", 
                   data=df.to_csv(index=False).encode(),  # ❌ Missing encoding spec
                   file_name="screening_results.csv")  # ❌ Missing MIME type
```

**FIXED CODE:**
```python
csv_data = df.to_csv(index=False).encode('utf-8')  # ✅ FIX: Explicit encoding
st.download_button(
    label="Download CSV",
    data=csv_data,
    file_name="screening_results.csv",
    mime="text/csv"  # ✅ FIX: Added MIME type
)
```

### Issue 5: Input Validation

**ORIGINAL CODE (WEAK VALIDATION):**
```python
if submitted and job_desc and uploaded_zip:  # ❌ Doesn't check for whitespace-only
    with st.spinner("Extracting resumes and processing..."):
        resume_paths = get_resumes_from_zip(uploaded_zip)  # ❌ No check if empty
        resumes_texts = [extract_text(p) for p in resume_paths]  # ❌ Can have empty texts
```

**FIXED CODE:**
```python
if submitted:
    # ✅ FIX: Proper validation with user feedback
    if not job_desc or not job_desc.strip():
        st.error("Please enter a job description.")
    elif not uploaded_zip:
        st.error("Please upload a ZIP file containing resumes.")
    else:
        with st.spinner("Extracting resumes and processing..."):
            try:
                resume_paths = get_resumes_from_zip(uploaded_zip)
                
                # ✅ FIX: Validate we found files
                if not resume_paths:
                    st.error("No PDF or DOCX files found in the uploaded ZIP file.")
                    st.stop()
                
                resumes_texts = [extract_text(p) for p in resume_paths]
                
                # ✅ FIX: Filter out empty resumes
                valid_resumes = [(name, text, path) for name, text, path in 
                                zip(resumes_names, resumes_texts, resume_paths) 
                                if text.strip()]
                
                if not valid_resumes:
                    st.error("No valid text could be extracted from the resumes.")
                    st.stop()
```

### Issue 6: Empty Skills Display

**ORIGINAL CODE (AWKWARD DISPLAY):**
```python
results.append({
    "Skills Matched": ", ".join(matched),  # ❌ Shows "" for empty
    "Missing Skills": ", ".join(missing),  # ❌ Shows "" for empty
    "Extra Skills": ", ".join(extra),  # ❌ Shows "" for empty
})
```

**FIXED CODE:**
```python
results.append({
    "Skills Matched": ", ".join(matched) if matched else "None",  # ✅ FIX: Shows "None"
    "Missing Skills": ", ".join(missing) if missing else "None",  # ✅ FIX: Shows "None"
    "Extra Skills": ", ".join(extra) if extra else "None",  # ✅ FIX: Shows "None"
})
```

## Summary of All Fixes

| Bug | Severity | Status |
|-----|----------|--------|
| UploadedFile object handling | CRITICAL | ✅ FIXED |
| Missing error handling | IMPORTANT | ✅ FIXED |
| Download button MIME type | IMPORTANT | ✅ FIXED |
| Empty resume validation | IMPORTANT | ✅ FIXED |
| Input validation | MEDIUM | ✅ FIXED |
| Nested file support | MEDIUM | ✅ FIXED |
| Spacy model check | LOW | ✅ FIXED |
| Empty skills display | LOW | ✅ FIXED |

## Testing Results

All core fixes have been validated:
- ✅ UploadedFile object handling works correctly
- ✅ Recursive file search finds files in subdirectories
- ✅ Empty input validation prevents crashes
- ✅ Error handling catches and reports issues gracefully
- ✅ Download button includes proper MIME type
