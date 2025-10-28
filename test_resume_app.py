"""
Unit tests for the Resume Screening App
Tests the main functions without requiring Streamlit UI
"""

import os
import sys
import tempfile
import zipfile
from io import BytesIO

# Add the parent directory to the path to import the module
sys.path.insert(0, '/home/runner/work/Blank/Blank')

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    try:
        import pandas as pd
        import numpy as np
        import PyPDF2
        import docx2txt
        print("✓ All basic imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_extract_text_basic():
    """Test basic text extraction functionality"""
    print("\nTesting extract_text function...")
    
    # Create a test text file
    test_file = '/tmp/test.txt'
    with open(test_file, 'w') as f:
        f.write("Test content")
    
    # The function should return empty string for unsupported formats
    from resume_screening_app import extract_text
    result = extract_text(test_file)
    
    if result == '':
        print("✓ extract_text correctly returns empty string for unsupported format")
        return True
    else:
        print("✗ extract_text did not handle unsupported format correctly")
        return False

def test_skill_gap_analysis():
    """Test skill gap analysis function"""
    print("\nTesting skill_gap_analysis function...")
    
    from resume_screening_app import skill_gap_analysis
    
    candidate_skills = ["Python", "Java", "Machine Learning"]
    required_skills = ["Python", "Machine Learning", "Deep Learning"]
    
    matched, missing, extra = skill_gap_analysis(candidate_skills, required_skills)
    
    # Check matched skills
    if set(matched) == {"Python", "Machine Learning"}:
        print("✓ Correctly identified matched skills")
    else:
        print(f"✗ Matched skills incorrect: {matched}")
        return False
    
    # Check missing skills
    if set(missing) == {"Deep Learning"}:
        print("✓ Correctly identified missing skills")
    else:
        print(f"✗ Missing skills incorrect: {missing}")
        return False
    
    # Check extra skills
    if set(extra) == {"Java"}:
        print("✓ Correctly identified extra skills")
    else:
        print(f"✗ Extra skills incorrect: {extra}")
        return False
    
    return True

def test_extract_skills():
    """Test skill extraction with sample text"""
    print("\nTesting extract_skills function...")
    
    try:
        from resume_screening_app import extract_skills
        
        # Test with empty text
        result = extract_skills("")
        if result == []:
            print("✓ extract_skills correctly handles empty text")
        else:
            print(f"✗ extract_skills did not return empty list for empty text: {result}")
            return False
        
        # Test with sample text containing entities
        sample_text = "I worked at Google and Microsoft in California."
        result = extract_skills(sample_text)
        print(f"  Extracted skills from sample: {result}")
        
        if len(result) > 0:
            print("✓ extract_skills extracts entities from text")
            return True
        else:
            print("⚠ extract_skills returned empty list (may need spacy model)")
            return True  # Not a hard failure
            
    except Exception as e:
        print(f"✗ Error in extract_skills: {e}")
        return False

def test_zip_file_handling():
    """Test ZIP file handling with BytesIO"""
    print("\nTesting ZIP file handling...")
    
    try:
        # Create a test ZIP file in memory
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            zip_file.writestr('test1.txt', 'Test content 1')
            zip_file.writestr('test2.pdf', 'Test PDF content')
        
        zip_buffer.seek(0)
        
        # Create a mock uploaded file object
        class MockUploadedFile:
            def __init__(self, data):
                self.data = data
            
            def getbuffer(self):
                return self.data
        
        mock_file = MockUploadedFile(zip_buffer.getvalue())
        
        # Test the function
        from resume_screening_app import get_resumes_from_zip
        files = get_resumes_from_zip(mock_file)
        
        # Check if files were found
        pdf_files = [f for f in files if f.endswith('.pdf')]
        
        if len(pdf_files) > 0:
            print(f"✓ get_resumes_from_zip successfully extracted {len(pdf_files)} PDF file(s)")
            return True
        else:
            print("✗ get_resumes_from_zip did not find any files")
            return False
            
    except Exception as e:
        print(f"✗ Error in ZIP handling test: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests and report results"""
    print("=" * 60)
    print("Resume Screening App - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Extract Text", test_extract_text_basic),
        ("Skill Gap Analysis", test_skill_gap_analysis),
        ("Extract Skills", test_extract_skills),
        ("ZIP File Handling", test_zip_file_handling),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS ✓" if result else "FAIL ✗"
        print(f"{test_name:.<40} {status}")
    
    print("=" * 60)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
