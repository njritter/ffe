import os
import google.generativeai as genai
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
import concurrent.futures
import time
from tqdm import tqdm
from io import BytesIO

# Load environment variables from .env file
load_dotenv()

def process_image(image_path, model):
    """Process a single image and return extracted text"""
    try:
        # Open image with PIL
        img = Image.open(image_path)
        
        # Convert TIFF to JPEG if needed
        if image_path.lower().endswith(('.tif', '.tiff')):
            # Convert to RGB mode (in case it's CMYK or another mode)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save to in-memory JPEG
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=95)
            buffer.seek(0)
            
            # Reload the image from the buffer
            img = Image.open(buffer)
        
        # Use Gemini to extract text
        response = model.generate_content([
            "Extract all text visible in this image. Maintain the original structure and layout as much as possible.",
            img
        ])
        
        return image_path, response.text, None
    except Exception as e:
        return image_path, None, str(e)

def save_ocr_result(image_path, ocr_text):
    """Save OCR text to a file"""
    if ocr_text is None:
        return False
    
    # Create output path with _ocr suffix
    output_path = str(image_path).rsplit('.', 1)[0] + '_ocr.txt'
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(ocr_text)
        return True
    except Exception as e:
        tqdm.write(f"Error saving OCR result to {output_path}: {str(e)}")
        return False

def worker(file_path, model):
    """Worker function for processing a single file"""
    start_time = time.time()
    
    # Process and save
    image_path, ocr_text, error = process_image(file_path, model)
    if ocr_text:
        success = save_ocr_result(image_path, ocr_text)
        status = "Success" if success else "Failed to save"
    else:
        status = f"Failed: {error}"
    
    elapsed = time.time() - start_time
    return image_path, status, elapsed

def process_directory():
    # Get API key and root directory from .env file
    api_key = os.getenv('GEMINI_API_KEY')
    root_dir = os.getenv('ROOT_DIRECTORY')
    model_name = os.getenv('MODEL_NAME')
    max_workers = int(os.getenv('MAX_WORKERS', '5'))
    
    if not api_key:
        print("Error: GEMINI_API_KEY not found in .env file")
        return
    
    if not root_dir:
        print("Error: ROOT_DIRECTORY not found in .env file")
        return
    
    if not model_name:
        print("Error: MODEL_NAME not found in .env file")
        return
    
    print(f"Starting parallel processing with {max_workers} workers")
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    # Supported image formats
    supported_formats = {'.jpg', '.jpeg', '.tiff', '.tif'}
    
    # Find all eligible files first
    files_to_process = []
    print("Scanning directories for images...")
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext in supported_formats:
                # Skip files that already have an OCR text file
                ocr_file = str(file_path).rsplit('.', 1)[0] + '_ocr.txt'
                if not os.path.exists(ocr_file):
                    files_to_process.append(file_path)
    
    total_files = len(files_to_process)
    print(f"Found {total_files} images to process in {root_dir}")
    
    if total_files == 0:
        print("No new files to process. Exiting.")
        return
    
    # Process files in parallel
    start_time = time.time()
    successful = 0
    failed = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(worker, file_path, model): file_path 
            for file_path in files_to_process
        }
        
        # Process results with a progress bar
        with tqdm(total=total_files, desc="Processing images", unit="file") as pbar:
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    image_path, status, elapsed = future.result()
                    if status.startswith("Success"):
                        successful += 1
                        tqdm.write(f"✓ {os.path.basename(image_path)} - {elapsed:.2f}s")
                    else:
                        failed += 1
                        tqdm.write(f"✗ {os.path.basename(image_path)} - {status}")
                except Exception as e:
                    failed += 1
                    tqdm.write(f"✗ {os.path.basename(file_path)} - Exception: {e}")
                
                pbar.update(1)
    
    total_time = time.time() - start_time
    print(f"\nProcessing complete in {total_time:.2f} seconds")
    print(f"Total: {total_files}, Successful: {successful}, Failed: {failed}")
    
    if total_files > 0:
        print(f"Average processing time: {total_time/total_files:.2f} seconds per image")
        if successful > 0:
            print(f"Average successful processing time: {total_time/successful:.2f} seconds per successful image")

if __name__ == "__main__":
    process_directory() 