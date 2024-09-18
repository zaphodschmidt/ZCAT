from flask import Flask, request, jsonify, send_file
import openai
import os
from dotenv import load_dotenv
from fpdf import FPDF
import fitz
import tiktoken
import time
import random
from openai import RateLimitError, APIError, APIConnectionError

load_dotenv()

# Debugging: Check if the environment variable is set
api_key = os.getenv("CHATGPT_API_KEY")
if not api_key:
    raise ValueError("CHATGPT_API_KEY environment variable is not set")

client = openai.OpenAI(api_key=api_key)

app = Flask(__name__)

MAX_TOKENS = 4096

def count_tokens(text):
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Changed to correct method
    tokens = tokenizer.encode(text)
    return len(tokens)

def split_text_into_chunks(text, max_tokens):
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = tokenizer.encode(text)
    chunks = []
    chunk_size = MAX_TOKENS - 100  # Leave some room for system message and response
    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i:i+chunk_size]
        chunks.append(tokenizer.decode(chunk))
    return chunks

@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    try:
        # Ensure the file is present in the request
        if 'resume' not in request.files:
            return jsonify({"error": f"No file part\nRequest.files:{request.files}"}), 400

        file = request.files['resume']

        # Process the file (e.g., read content, etc.)
        pdf_document = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()

        print("text: ", text, flush=True)

        if count_tokens(text) > MAX_TOKENS:
            text_chunks = split_text_into_chunks(text, MAX_TOKENS)
        else:
            text_chunks = [text]
        
        print("count_tokens: ", count_tokens(text), flush=True)
        print("text_chunks: ", text_chunks, flush=True)
        
        # Extract metadata from the entire text
        metadata = extract_metadata_with_retry(text)

        print(metadata)

        # Logic to store resume information and metadata in the database
        # ...

        return jsonify({"message": "Resume uploaded and processed successfully", "metadata": metadata}), 200

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

def extract_metadata_with_retry(text, max_retries=5, base_delay=1):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Extract key metadata from the resume text. Focus on name, contact information, education, work experience, skills, and any other relevant information. Provide the output in a structured format."},
                    {"role": "user", "content": text}
                ],
                max_tokens=MAX_TOKENS
            )
            return response.choices[0].message.content
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            delay = (base_delay * 2 ** attempt) + random.uniform(0, 1)
            time.sleep(delay)
        except (APIError, APIConnectionError) as e:
            print(f"API error occurred: {str(e)}")
            if attempt == max_retries - 1:
                raise
            delay = (base_delay * 2 ** attempt) + random.uniform(0, 1)
            time.sleep(delay)
    raise Exception("Maximum retries reached for metadata extraction")

# Endpoint to process job qualifications and generate resume
@app.route('/generate_resume', methods=['POST'])
def generate_resume():
    try:
        data = request.json
        job_qualifications = data.get('job_qualifications')

        if count_tokens(job_qualifications) > MAX_TOKENS:
            return jsonify({"error": "Job qualifications text exceeds the maximum token limit."}), 400

        # Logic to generate resume using OpenAI API
        response = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Generate a resume based on these qualifications."},
            {"role": "user", "content": job_qualifications}
        ],
        max_tokens=1500)
        resume_content = response.choices[0].message.content
        # Logic to generate PDF from resume content
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, resume_content)
        pdf_output = f"{data.get('name')}_{data.get('company')}.pdf"
        pdf.output(pdf_output)
        return send_file(pdf_output, as_attachment=True)

    except openai.RateLimitError as e:
        return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
