from dotenv import load_dotenv
import google.generativeai as genai
import os
from pathlib import Path
from PIL import Image


# Load environment variables from .env file
load_dotenv()


def process_image(image_path, model):
    """Process a single image and return extracted text"""
    try:
        # Open image with PIL
        img = Image.open(image_path)
        
        # Use Gemini to extract text
        response = model.generate_content([
            "Extract all text visible in this image. Maintain the original structure and layout as much as possible.",
            img
        ])
        
        return response.text
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None
    

def save_ocr_result(image_path, ocr_text):
    """Save OCR text to a file"""
    if ocr_text is None:
        return False
    
    # Create output path with _ocr suffix
    output_path = str(image_path).rsplit('.', 1)[0] + '_ocr.txt'
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(ocr_text)
        print(f"OCR result saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving OCR result to {output_path}: {str(e)}")
        return False
    

def process_directory():
    # Get API key and root directory from .env file
    api_key = os.getenv('GEMINI_API_KEY')
    root_dir = os.getenv('ROOT_DIRECTORY')
    model_name = os.getenv('MODEL_NAME')

    if not api_key:
        print("Error: GEMINI_API_KEY not found in .env file")
        return
    
    if not root_dir:
        print("Error: ROOT_DIRECTORY not found in .env file")
        return
    
    if not model_name:
        print("Error: MODEL_NAME not found in .env file")
        return
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    # Supported image formats
    supported_formats = {'.jpg', '.jpeg', '.tiff', '.tif'}
    
    # Statistics
    processed = 0
    failed = 0
    total = 0
    
    print(f"Starting to process images in: {root_dir}")
    
    # Walk through all directories
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            file_ext = os.path.splitext(filename)[1].lower()
            
            # Check if it's an image file we support
            if file_ext in supported_formats:
                total += 1
                print(f"Processing {file_path}")
                
                # Extract text from image
                ocr_text = process_image(file_path, model)
                
                # Save the result
                if ocr_text and save_ocr_result(file_path, ocr_text):
                    processed += 1
                else:
                    failed += 1
    
    # Print summary
    print(f"Processing complete. Total: {total}, Successful: {processed}, Failed: {failed}")


if __name__ == "__main__":
    process_directory()