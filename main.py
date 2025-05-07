import nest_asyncio
import chromadb
import os
import json
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.prompts import PromptTemplate
import fitz  # PyMuPDF
import cv2
import numpy as np
from gtts import gTTS
import re
from PIL import Image
import pytesseract
from fuzzywuzzy import process
import uuid
from datetime import datetime
from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel
from pdf2image import convert_from_path
import pandas as pd
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import codecs

# Load environment variables
load_dotenv()

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# In-memory cache for query results
QUERY_CACHE = {}

# Metadata and JSON files
METADATA_FILE = "pdf_metadata.json"
JSON_FILE = "query_responses_8.json"

def multiple_image(query):
    page_list = []
    excel_path = 'cliplevel.xlsx'
    output_folder = 'output_images'
    poppler_path = r"poppler-24.08.0/Library/bin/"

    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_excel(excel_path)

    query = query.lower()
    image_list = []
    for _, row in df.iterrows():
        question = str(row['Question']).strip()
        if question == query:
            pdf_path = str(row['PDF_path']).strip()
            page_numbers = str(row['page_no']).strip()

            if not question or not pdf_path or not page_numbers:
                continue

            if not os.path.isfile(pdf_path):
                print(f"❌ Missing PDF: {pdf_path}")
                continue

            try:
                page_list = [int(p.strip()) for p in page_numbers.split(',')]
            except ValueError:
                print(f"❌ Invalid page numbers: {page_numbers}")
                continue
                        
            for page_num in page_list:
                try:
                    images = convert_from_path(
                        pdf_path,
                        first_page=page_num,
                        last_page=page_num,
                        poppler_path=poppler_path
                    )
                    for img in images:
                        safe_question = "".join(c if c.isalnum() else "_" for c in question)[:50]
                        img_name = f"{safe_question}_page_{page_num}.png"
                        img.save(os.path.join(output_folder, img_name))
                        print(f"✅ Saved: {img_name}")
                        img_name = f"https://chatbot.chervicaon.com/output_images/{img_name}"
                        image_list.append(img_name)
                except Exception as e:
                    print(f"❌ Error processing {pdf_path} page {page_num}: {e}")
    return page_list, image_list

# Pydantic model for query input
class QueryInput(BaseModel):
    query: str

# Function to initialize or load JSON file
def init_json_file():
    if not os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'w') as f:
            json.dump({}, f)

# Function to initialize or load metadata file
def init_metadata_file():
    if not os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'w') as f:
            json.dump({}, f)

# Function to load PDF metadata
def load_pdf_metadata():
    init_metadata_file()
    with open(METADATA_FILE, 'r') as f:
        return json.load(f)

# Function to load vector index for a single PDF
def load_vector_index(file_path, embed_model):
    try:
        metadata = load_pdf_metadata()
        if file_path not in metadata:
            print(f"No index found for {file_path}. Run pdf_indexer.py first.")
            return None
        collection_name = metadata[file_path]["collection_name"]
        chroma_collection = chroma_client.get_collection(collection_name)
        print(f"Collection {collection_name} has {chroma_collection.count()} documents.")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        vector_index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
        print(f"Loaded vector index for {file_path} from collection {collection_name}.")
        return vector_index
    except Exception as e:
        print(f"Error loading vector index for {file_path}: {e}")
        return None

# Function to format response and extract metadata
def format_response(response_text, file_path, query_text):
    page_number = response_text['metadata'].get("page_number", "Unknown")
    query_lower = query_text.lower()
    response_str = response_text['text'].strip()

    # Handle approval flow queries with "clip level"
    if "approval" in query_lower and "clip level" in query_lower:
        if "no specific information" in response_str.lower():
            context = response_text['metadata'].get('context', response_str)
            # Look for monetary amounts or thresholds in the context
            amounts = re.findall(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b', context)
            if amounts:
                # Infer clip level as a threshold and describe the approval flow
                return (f"While 'clip level' is not explicitly mentioned, the approval flow likely involves thresholds based on amounts like {', '.join(amounts)}. The context mentions approval chains with first, second, and third approvers, suggesting that for amounts above certain thresholds, additional approvals may be required."), page_number
        return response_str, page_number

    # Handle dashboard-related queries
    if "dashboard" in query_lower and "procurement tasks" in query_lower:
        if "no specific information" in response_str.lower():
            context = response_text['metadata'].get('context', response_str)
            if "procurement" in context.lower():
                return ("The user dashboard likely provides visibility into procurement tasks by displaying key stages such as purchase order creation, order confirmation, and goods receipt, as mentioned in the context. However, specific dashboard features are not detailed."), page_number
        return response_str, page_number

    # Existing formatting logic for other query types
    if any(keyword in query_lower for keyword in ["who", "name"]):
        match = re.search(r'([A-Za-z\s]+)', response_str)
        name = match.group(0).strip() if match else "Not specified"
        return name, page_number
    elif any(keyword in query_lower for keyword in ["process", "steps", "flow"]):
        # Remove introductory text like "The buying process consists of..."
        response_str = re.sub(
            r'^(.*?following steps:?)\s*\n*',
            '',
            response_str,
            flags=re.IGNORECASE | re.DOTALL
        ).strip()
        # Split on common step prefixes
        steps = re.split(
            r'\n*(?:\d+\.\s*|\(\d+\)\s*|Step \d+:\s*|step \d+:\s*)',
            response_str,
            flags=re.IGNORECASE
        )
        steps = [step.strip() for step in steps if step.strip()]  # Remove empty steps
        if steps:
            formatted_steps = [f"step {i+1}: {step}" for i, step in enumerate(steps)]
            return "\n".join(formatted_steps), page_number
        return "No specific steps found in the provided context.", page_number
    return response_str, page_number

# Function to handle the query for a single PDF
def query_document(vector_index, query_text, llm, file_path):
    try:
        qa_template = PromptTemplate(
            "Given the context, provide a precise answer to the question. Ensure the answer is directly relevant to the query. For approval-related queries involving terms like 'clip level', interpret 'clip level' as a possible monetary threshold for approvals and look for related information (e.g., amounts, approval chains, or levels of approvers). If the question involves a user dashboard or tracking procurement tasks, look for related information (e.g., visibility into processes, task statuses, or system features) and infer how a dashboard might assist. For process-related queries, return the relevant steps if present, or infer a general process flow if no exact details are found. If no relevant information is available, return 'No specific information found.' Context: {context_str}\nQuestion: {query_str}\nAnswer:"
        )
        query_engine = vector_index.as_query_engine(
            llm=llm,
            response_mode="compact",
            similarity_top_k=30,  # Increased to retrieve more context
            text_qa_template=qa_template
        )
        response = query_engine.query(query_text)
        if not response.source_nodes:
            print(f"No matching document found in {file_path}.")
            return None, None, None
        metadata = response.source_nodes[0].metadata
        page_number = metadata.get("page_number")
        score = response.source_nodes[0].score if hasattr(response.source_nodes[0], 'score') else 0
        if not page_number:
            print(f"No page_number found in metadata for {file_path}.")
            return None, None, None
        print(f"Retrieved context for {file_path} (page {page_number}, score {score}):\n{response.source_nodes[0].text[:500]}")
        metadata['context'] = response.source_nodes[0].text
        formatted_response, _ = format_response({"text": str(response), "metadata": metadata}, file_path, query_text)
        # Remove "Based on the provided context" and capitalize first letter
        formatted_response = re.sub(r'^(Based on the provided context,?\s*)', '', formatted_response, flags=re.IGNORECASE)
        if formatted_response:
            formatted_response = formatted_response[0].upper() + formatted_response[1:]
        return formatted_response, page_number, score
    except Exception as e:
        print(f"Error querying document {file_path}: {e}")
        return None, None, None

# Function to extract text from scanned PDFs (OCR)
def extract_text_from_image(pdf_path, page_number):
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_number - 1)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = img.convert("L")  # Convert to grayscale
        img = img.point(lambda x: 0 if x < 128 else 255, "1")  # Binarize
        text = pytesseract.image_to_string(img)
        print(f"OCR Extracted Text for {pdf_path} (page {page_number}):\n{text[:500]}")
        return text
    except Exception as e:
        print(f"Error performing OCR on {pdf_path} (page {page_number}): {e}")
        return ""

# Function to find similar text for highlighting
def find_similar_text(page, query):
    text = page.get_text("text")
    lines = text.split("\n")
    best_match = process.extractOne(query, lines, score_cutoff=80)
    return best_match[0] if best_match else None

# Function to extract and highlight the answer in an image
def extract_and_highlight(file_path, page_number, answer, query_id, IMAGE_DIR):
    try:
        doc = fitz.open(file_path)
        page = doc.load_page(page_number - 1)
        extracted_text = page.get_text("text")
        if not extracted_text.strip():
            print(f"Page {page_number} has no extracted text. Trying OCR...")
            extracted_text = extract_text_from_image(file_path, page_number)
            if not extracted_text.strip():
                print(f"OCR failed to extract text for {file_path} (page {page_number}).")
                return None
        print(f"Extracted Text on Page {page_number} for {file_path}:\n{extracted_text[:500]}")
        text_instances = page.search_for(answer)
        if not text_instances:
            print(f"No exact match found for: {answer}")
            approx_answer = find_similar_text(page, answer)
            if approx_answer:
                print(f"Using fuzzy match: {approx_answer}")
                text_instances = page.search_for(approx_answer)
            else:
                print(f"No fuzzy match found for {file_path} (page {page_number}).")
                return None
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = np.array(img)
        for rect in text_instances:
            x0, y0, x1, y1 = int(rect.x0), int(rect.y0), int(rect.x1), int(rect.y1)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        highlighted_img = Image.fromarray(img)
        img_path = os.path.join(IMAGE_DIR, f"{query_id}_highlighted.png")
        highlighted_img.save(img_path)
        print(f"Highlighted answer saved as '{img_path}'")
        return f"https://chatbot.chervicaon.com/outputs/images/{os.path.basename(img_path)}"
    except Exception as e:
        print(f"Error extracting and highlighting answer for {file_path}: {e}")
        return None

# Function to create audio response
def generate_audio(response_text, query_id, AUDIO_DIR):
    try:
        tts = gTTS(text=response_text, lang="en")
        audio_path = os.path.join(AUDIO_DIR, f"{query_id}_response.mp3")
        tts.save(audio_path)
        print(f"Audio response saved as '{audio_path}'")
        return f"https://chatbot.chervicaon.com/outputs/audio/{os.path.basename(audio_path)}"
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None

# Function to save query and response to JSON
def save_to_json(query_id, query_text, response, page_number=None, img_path=None, audio_path=None, pdf_path=None):
    init_json_file()
    with open(JSON_FILE, 'r+') as f:
        data = json.load(f)
        data[query_id] = {
            "query": query_text,
            "response": response,
            "page_number": page_number,
            "image_path": img_path if img_path else "https://chatbot.chervicaon.com/outputs/images/None",
            "audio_path": audio_path,
            "pdf_path": f"https://chatbot.chervicaon.com/uploads/{os.path.basename(pdf_path)}" if pdf_path else None,
            "timestamp": str(datetime.now())
        }
        f.seek(0)
        json.dump(data, f, indent=4)
    return data[query_id]

# Function to check cached query
def check_cached_query(query_text, max_age_hours=24):
    query_text_lower = query_text.lower()
    cache_key = query_text_lower
    if cache_key in QUERY_CACHE:
        print(f"Returning in-memory cached response for query: {query_text}")
        # Ensure cached response has full URLs
        cached = QUERY_CACHE[cache_key]
        return {
            "response": cached["response"],
            "page_number": cached["page_number"],
            "image_path": cached["image_path"],
            "audio_path": cached["audio_path"],
            "pdf_path": cached["pdf_path"],
            "timestamp": datetime.now().isoformat()
        }
    init_json_file()
    with open(JSON_FILE, 'r') as f:
        data = json.load(f)
        for query_id, entry in data.items():
            if isinstance(entry, dict) and entry["query"].lower() == query_text_lower:
                timestamp = datetime.fromisoformat(entry["timestamp"])
                age = (datetime.now() - timestamp).total_seconds() / 3600
                if age < max_age_hours:
                    print(f"Returning JSON-cached response for query: {query_text}")
                    # Ensure full URLs in cached response
                    cached_response = {
                        "response": entry["response"],
                        "page_number": entry["page_number"],
                        "image_path": entry["image_path"],
                        "audio_path": entry["audio_path"],
                        "pdf_path": entry["pdf_path"],
                        "timestamp": datetime.now().isoformat()
                    }
                    QUERY_CACHE[cache_key] = cached_response
                    return cached_response
    return None

# Function to handle queries
def process_query(query_text, embed_model, llm, IMAGE_DIR, AUDIO_DIR):
    cached_response = check_cached_query(query_text)
    if cached_response:
        return {
            "query": query_text,
            "response": cached_response["response"],
            "page_number": cached_response["page_number"],
            "image_path": cached_response["image_path"],
            "audio_path": cached_response["audio_path"],
            "pdf_path": cached_response["pdf_path"],
            "timestamp": datetime.now().isoformat()
        }
    query_id = str(uuid.uuid4())
    json_save = None
    metadata = load_pdf_metadata()
    pdf_files = list(metadata.keys())
    if not pdf_files:
        print("No indexed PDFs found. Run pdf_indexer.py.")
        json_save = save_to_json(query_id, query_text, "No indexed PDFs found.")
        return json_save
    best_response = None
    best_score = -1
    best_page_number = None
    best_file_path = None
    for file_path in pdf_files:
        vector_index = load_vector_index(file_path, embed_model)
        if not vector_index:
            continue
        response, page_number, score = query_document(vector_index, query_text, llm, file_path)
        if response and page_number and score is not None:
            if score > best_score:
                best_score = score
                best_response = response
                best_page_number = page_number
                best_file_path = file_path
    if best_response:
        img_path = extract_and_highlight(best_file_path, best_page_number, query_text, query_id, IMAGE_DIR)
        audio_path = generate_audio(best_response, query_id, AUDIO_DIR)
        json_save = save_to_json(query_id, query_text, best_response, best_page_number, img_path, audio_path, best_file_path)
        cache_key = query_text.lower()
        QUERY_CACHE[cache_key] = {
            "response": best_response,
            "page_number": best_page_number,
            "image_path": img_path if img_path else "https://chatbot.chervicaon.com/outputs/images/None",
            "audio_path": audio_path,
            "pdf_path": f"https://chatbot.chervicaon.com/uploads/{os.path.basename(best_file_path)}" if best_file_path else None
        }
        return json_save
    json_save = save_to_json(query_id, query_text, "No specific information found.")
    return json_save

def main_method(query_text):
    nest_asyncio.apply()
    hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
    if not hf_token:
        raise ValueError("Missing HUGGINGFACE_API_TOKEN in .env file.")
    OUTPUT_DIR = "outputs"
    IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
    AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    llm = HuggingFaceInferenceAPI(
        model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
        contextWindow=32768,
        maxTokens=1024,
        temperature=0.05,
        topP=0.9,
        frequencyPenalty=0.5,
        presencePenalty=0.5,
        token=hf_token
    )
    response = process_query(query_text, embed_model, llm, IMAGE_DIR, AUDIO_DIR)
    return response

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://s3.ariba.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs/images", StaticFiles(directory="outputs/images"), name="images")
app.mount("/outputs/audio", StaticFiles(directory="outputs/audio"), name="audio")
app.mount("/uploads", StaticFiles(directory="uploads"), name="pdf")

@app.post("/query")
async def handle_query(value: str):
    query = value
    response = main_method(query)
    page_list, image_list = multiple_image(query)
    if page_list:
        page = ",".join(str(item) for item in page_list).replace('"', '')
        response['page_number'] = page 
        img_string = ", ".join(image_list)
        response['image_path'] = img_string
    if "steps:" in response['response'].lower():
        response['response'] = response['response'].replace("\n\n", "\n")
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.1", port=8000)
