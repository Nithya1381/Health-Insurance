import json
import os
import tempfile
import uuid
from urllib.parse import quote

try:
    import PyPDF2
except ImportError:
    raise ImportError("Please install PyPDF2: pip install PyPDF2")

try:
    import pytesseract
except ImportError:
    raise ImportError("Please install pytesseract: pip install pytesseract")

try:
    from pdf2image import convert_from_path
except ImportError:
    raise ImportError("Please install pdf2image: pip install pdf2image")

try:
    import tiktoken
    from langchain.chains import ConversationalRetrievalChain, LLMChain
    from langchain.prompts import PromptTemplate
    from langchain.schema import Document
    from langchain_community.chat_models import ChatOpenAI
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
except ImportError as e:
    if "tiktoken" in str(e):
        raise ImportError(
            "Please install tiktoken: pip install tiktoken\n"
            "This package is required for OpenAI embeddings."
        )
    raise ImportError(
        "Please install required packages: "
        "pip install langchain langchain-community openai faiss-cpu tiktoken\n"
        f"Error: {str(e)}"
    )

VECTOR_STORE_DIR = "vector_store"
TEMP_DIR = "temp_files"

# Create required directories
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

def check_tesseract():
    """Check if Tesseract is properly installed"""
    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError(
            "Tesseract is not installed or not in PATH. "
            "Please install Tesseract OCR:\n"
            "- Ubuntu/Debian: sudo apt-get install tesseract-ocr\n"
            "- macOS: brew install tesseract\n"
            "- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"
        )

def check_poppler():
    """Check if Poppler is properly installed"""
    try:
        convert_from_path(os.path.join(TEMP_DIR, "test.pdf"))
    except Exception as e:
        if "poppler" in str(e).lower():
            raise RuntimeError(
                "Poppler is not installed or not in PATH. "
                "Please install Poppler:\n"
                "- Ubuntu/Debian: sudo apt-get install poppler-utils\n"
                "- macOS: brew install poppler\n"
                "- Windows: Download from http://blog.alivate.com.au/poppler-windows/"
            )

def get_safe_filename(filename):
    """Generate a safe filename with UUID to avoid conflicts"""
    ext = os.path.splitext(filename)[1]
    safe_name = f"{uuid.uuid4()}{ext}"
    return safe_name

def extract_text_from_images(pdf_path):
    """Extract text from PDF using Tesseract OCR"""
    check_tesseract()
    images = convert_from_path(pdf_path)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img) + "\n"
    return text

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF directly"""
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def extract_insurance_info(text):
    """Extract structured information from insurance document"""
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=2000)
    prompt_template = PromptTemplate(
        input_variables=["document"],
        template="""
        Extract the following fields from the insurance document:
        - Policy Type
        - Coverage Amount
        - Premium Amount
        - Policy Period
        - Key Benefits
        - Exclusions
        Output the result in JSON format.
        Document:
        {document}
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    result = chain.run(document=text)
    return result

def process_pdf(pdf_file):
    # Check dependencies
    check_tesseract()
    check_poppler()
    
    # Generate a safe temporary filename
    safe_filename = get_safe_filename(pdf_file.name)
    temp_path = os.path.join(TEMP_DIR, safe_filename)
    
    try:
        # Save the uploaded file temporarily
        with open(temp_path, 'wb+') as destination:
            for chunk in pdf_file.chunks():
                destination.write(chunk)

        # Extract text using both methods
        print("Extracting text from PDF...")
        pdf_text = extract_text_from_pdf(temp_path)
        print("Combining text...")
        combined_text = pdf_text

        print(combined_text, "combined_text")
        
        if not combined_text.strip():
            raise Exception("No text could be extracted from the PDF")

        # Extract structured information
        extracted_data = extract_insurance_info(combined_text)
        extracted_json = json.loads(extracted_data)
        print(extracted_json, "extracted_json")

        # Create a Document object with metadata
        print("Document creating...")
        doc = Document(
            page_content=combined_text,
            metadata={
                "filename": pdf_file.name,
                "extracted_info": extracted_json
            }
        )
        print("Document created...")
        # Store in FAISS
        embedding = OpenAIEmbeddings()
        print("Embedding...")
        # Check if there's an existing vector store
        if os.path.exists(os.path.join(VECTOR_STORE_DIR, "index.faiss")):
            # Load existing vector store and add new documents
            print("Loading existing vector store...")
            vectordb = FAISS.load_local(
                VECTOR_STORE_DIR, 
                embedding,
                allow_dangerous_deserialization=True
            )
            print("Adding documents to vector store...")
            vectordb.add_documents([doc])
        else:
            # Create new vector store
            print("Creating new vector store...")
            vectordb = FAISS.from_documents([doc], embedding)
        
        # Save the vector store
        print("Saving vector store...")
        vectordb.save_local(VECTOR_STORE_DIR)
        print("Vector store saved...")

        return "Document processed and stored successfully!"
    
    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

def get_conversation_chain():
    try:
        if not os.path.exists(os.path.join(VECTOR_STORE_DIR, "index.faiss")):
            raise Exception("No documents have been processed yet. Please upload a document first.")

        # Create a conversation chain using FAISS
        embedding = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            VECTOR_STORE_DIR, 
            embedding,
            allow_dangerous_deserialization=True
        )

        # Create a conversation chain
        llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo"
        )
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            verbose=True
        )
        
        return conversation_chain
    except Exception as e:
        raise Exception(f"Error loading conversation chain: {str(e)}")

def search_documents(query):
    """Search documents in FAISS"""
    try:
        embedding = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            VECTOR_STORE_DIR, 
            embedding,
            allow_dangerous_deserialization=True
        )
        results = vectorstore.similarity_search(query, k=5)
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]
    except Exception as e:
        raise Exception(f"Error searching documents: {str(e)}")

def get_all_documents():
    """Retrieve all documents stored in FAISS vector store"""
    try:
        if not os.path.exists(os.path.join(VECTOR_STORE_DIR, "index.faiss")):
            return {"error": "No documents found in vector store"}
            
        embedding = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            VECTOR_STORE_DIR, 
            embedding,
            allow_dangerous_deserialization=True
        )
        
        # Get all documents
        docs = vectorstore.similarity_search("", k=1000)  # Get up to 1000 documents
        print(docs, "docs")
        # Format the results
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata,
                "filename": doc.metadata.get("filename", "Unknown"),
                "extracted_info": doc.metadata.get("extracted_info", {})
            })
            
        return {
            "total_documents": len(results),
            "documents": results
        }
        
    except Exception as e:
        return {"error": f"Error retrieving documents: {str(e)}"}