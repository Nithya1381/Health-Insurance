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
    from langchain.text_splitter import RecursiveCharacterTextSplitter
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
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,  # Smaller chunks to stay within token limits
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Split text into chunks
    chunks = text_splitter.split_text(text)
    
    # Process each chunk and combine results
    all_results = []
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=1000)
    
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1} of {len(chunks)}...")
        
        prompt_template = PromptTemplate(
            input_variables=["document_chunk"],
            template="""Extract any of the following fields present in this portion of the insurance document:
            - Policy Type
            - Coverage Amount
            - Premium Amount
            - Policy Period
            - Key Benefits
            - Exclusions
            
            Only include fields that are clearly mentioned in this text. Output in JSON format.
            If no relevant information is found, return an empty JSON object {}.
            
            Document chunk:
            {document_chunk}
            """
        )
        
        chain = LLMChain(llm=llm, prompt=prompt_template)
        try:
            result = chain.run(document_chunk=chunk)
            chunk_data = json.loads(result)
            if chunk_data:  # Only append if we found some information
                all_results.append(chunk_data)
        except Exception as e:
            print(f"Error processing chunk {i+1}: {str(e)}")
            continue
    
    # Combine results from all chunks
    combined_result = {
        "Policy Type": None,
        "Coverage Amount": None,
        "Premium Amount": None,
        "Policy Period": None,
        "Key Benefits": [],
        "Exclusions": []
    }
    
    # Merge all results into one
    for result in all_results:
        for key, value in result.items():
            if key in ["Key Benefits", "Exclusions"]:
                if value and isinstance(value, list):
                    combined_result[key].extend(value)
            else:
                if value and not combined_result[key]:
                    combined_result[key] = value
    
    # Remove duplicates from lists
    combined_result["Key Benefits"] = list(set(combined_result["Key Benefits"]))
    combined_result["Exclusions"] = list(set(combined_result["Exclusions"]))
    
    return json.dumps(combined_result)

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

        # Extract text from PDF
        print("Extracting text from PDF...")
        pdf_text = extract_text_from_pdf(temp_path)
        
        if not pdf_text.strip():
            raise Exception("No text could be extracted from the PDF")

        # Split text for vector storage
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Extract structured information
        print("Extracting insurance information...")
        extracted_data = extract_insurance_info(pdf_text)
        extracted_json = json.loads(extracted_data)
        print("Extracted info:", extracted_json)

        # Create document chunks
        print("Creating document chunks...")
        text_chunks = text_splitter.split_text(pdf_text)
        docs = []
        for i, chunk in enumerate(text_chunks):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "filename": pdf_file.name,
                    "chunk": i+1,
                    "total_chunks": len(text_chunks),
                    "extracted_info": extracted_json
                }
            ))

        print(f"Created {len(docs)} document chunks")

        # Store in FAISS
        embedding = OpenAIEmbeddings()
        print("Creating embeddings...")
        
        if os.path.exists(os.path.join(VECTOR_STORE_DIR, "index.faiss")):
            print("Loading existing vector store...")
            vectordb = FAISS.load_local(
                VECTOR_STORE_DIR, 
                embedding,
                allow_dangerous_deserialization=True
            )
            print("Adding documents to vector store...")
            vectordb.add_documents(docs)
        else:
            print("Creating new vector store...")
            vectordb = FAISS.from_documents(docs, embedding)
        
        print("Saving vector store...")
        vectordb.save_local(VECTOR_STORE_DIR)
        print("Vector store saved...")

        return "Document processed and stored successfully!"
    
    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")
    
    finally:
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

def classify_question(question: str) -> dict:
    """Classify if the question is insurance-related or general chat"""
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    prompt_template = PromptTemplate(
        input_variables=["question"],
        template="""Classify if the following question is related to insurance/policies or if it's general chat.
        Question: {question}
        
        Return response in JSON format with two fields:
        - "is_insurance_related": boolean (true/false)
        - "type": string ("insurance" or "general_chat")
        - "confidence": float (0 to 1)
        """
    )
    
    chain = LLMChain(llm=llm, prompt=prompt_template)
    result = chain.run(question=question)
    return json.loads(result)

def get_general_chat_response(question: str) -> str:
    """Handle general chat questions"""
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    
    prompt_template = PromptTemplate(
        input_variables=["question"],
        template="""You are a friendly AI assistant for a health insurance company.
        Respond to the following general question in a professional and helpful manner.
        If the user tries to get specific policy information, politely inform them to ask about their policy directly.
        
        Question: {question}
        """
    )
    
    chain = LLMChain(llm=llm, prompt=prompt_template)
    return chain.run(question=question)

def get_insurance_response(question: str, conversation_chain) -> str:
    """Handle insurance-related questions using the document knowledge base"""
    try:
        # First, search for relevant documents
        embedding = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            VECTOR_STORE_DIR, 
            embedding,
            allow_dangerous_deserialization=True
        )
        
        # Get relevant documents
        relevant_docs = vectorstore.similarity_search(question, k=3)
        
        # Extract content and metadata from relevant documents
        context = []
        for doc in relevant_docs:
            context.append({
                "content": doc.page_content,
                "metadata": doc.metadata.get("extracted_info", {})
            })
        
        # Create a prompt that includes the context
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        prompt_template = PromptTemplate(
            input_variables=["question", "context"],
            template="""You are an AI assistant for a health insurance company. Use the following context from insurance documents to answer the user's question.
            If the information is not in the context, politely say that you don't have that specific information.

            Context from insurance documents:
            {context}

            User Question: {question}

            Please provide a clear, professional response based on the provided context. Include specific details from the documents when available.
            """
        )
        
        # Format context for the prompt
        context_text = "\n\n".join([
            f"Document {i+1}:\n"
            f"Content: {doc['content']}\n"
            f"Policy Details: {json.dumps(doc['metadata'], indent=2)}"
            for i, doc in enumerate(context)
        ])
        
        # Get response using the context
        chain = LLMChain(llm=llm, prompt=prompt_template)
        response = chain.run(
            question=question,
            context=context_text
        )
        
        return response

    except Exception as e:
        return f"I apologize, but I encountered an error retrieving your insurance information: {str(e)}"