# Extract Text From Images

Iterate through all folders in directory, throw all .tif(tiff) and .jpg (jpeg) images at AI to extract text, and add file (image_path)_ocr.txt to directory.

Step 1 -- Create virtual environment

python -m venv ./.venv

Step 2 -- Install requirements

pip install -r requirements.txt

Step 3 -- .env file

Make sure there is a .env file with:

GEMINI_API_KEY=
ROOT_DIRECTORY=
MODEL_NAME=

Step 4 -- Run extraction

python extract_text_parallel.py