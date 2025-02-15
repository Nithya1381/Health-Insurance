import os
import tempfile
import shutil

from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import redirect, render
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from django.views.decorators.http import require_POST
from django.contrib.auth.decorators import login_required

from .models import ChatMessage, Document
from .utils import (classify_question, generate_speech, get_all_documents,
                    get_conversation_chain, get_general_chat_response,
                    get_insurance_response, process_pdf, speech_to_text,
                    VECTOR_STORE_DIR)

load_dotenv()

# Create your views here.

def home(request):
    # Sample context data
    context = {
        'title': 'Welcome to Health Insurance Agent',
        'message': 'This is a sample template page'
    }
    return render(request, 'main/home.html', context)

def chat(request):
    if request.method == 'POST':
        if 'message' in request.POST:
            user_message = request.POST.get('message')
            ChatMessage.objects.create(role='user', content=user_message)
            
            try:
                # First, classify the question
                classification = classify_question(user_message)
                
                if classification['is_insurance_related']:
                    # Get conversation chain for insurance questions
                    conversation_chain = get_conversation_chain()
                    response_text = get_insurance_response(user_message, conversation_chain)
                else:
                    # Handle general chat
                    response_text = get_general_chat_response(user_message)
                
                # Save AI response
                ChatMessage.objects.create(
                    role='assistant',
                    content=response_text
                )
                
                # Generate speech from response
                speech_data = generate_speech(response_text, "en-IN", "meera")
                
                # Return both text and speech data
                return JsonResponse({
                    'message': response_text,
                    'speech': speech_data
                })
            
            except Exception as e:
                return JsonResponse({
                    'error': str(e)
                }, status=500)
    
    chat_messages = ChatMessage.objects.filter(role__in=['user', 'assistant'])
    return render(request, 'main/chat.html', {'chat_messages': chat_messages})

def uploads(request):
    if request.method == 'POST' and request.FILES.get('document'):
        print("request.FILES",request.FILES)
        document = request.FILES['document']
        if not document.name.endswith('.pdf'):
            messages.error(request, 'Please upload a PDF file')
            return redirect('main:uploads')
        
        try:
            # Save document
            doc = Document.objects.create(file=document)
            
            # Process and store in vector database
            print("Processing document...")
            result = process_pdf(document)
            doc.processed = True
            doc.save()
            messages.success(request, 'Document uploaded and processed successfully!')
            
        except Exception as e:
            messages.error(request, f'Error processing document: {str(e)}')
            if doc.id:
                doc.delete()  # Clean up the document if processing failed
    
    documents = Document.objects.all().order_by('-uploaded_at')
    return render(request, 'main/uploads.html', {'documents': documents})

def view_documents(request):
    try:
        docs_data = get_all_documents()
        return render(request, 'main/documents.html', {'data': docs_data})
    except Exception as e:
        messages.error(request, str(e))
        return redirect('main:uploads')

def convert_speech(request):
    if request.method == 'POST' and request.FILES.get('audio'):
        try:
            audio_file = request.FILES['audio']
            
            # Save the audio file temporarily
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                for chunk in audio_file.chunks():
                    temp_file.write(chunk)
                temp_path = temp_file.name
            
            try:
                # Convert speech to text
                text = speech_to_text(request, temp_path, 'en-IN')  # Assuming English language
                
                # Check if the response is an error message
                if "Error:" in text:
                    raise Exception(text)
                
                # Clean up the temporary file
                os.unlink(temp_path)
                
                return JsonResponse({'text': text})
                
            except Exception as e:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                return JsonResponse({'error': f'Speech to text error: {str(e)}'}, status=500)
                
        except Exception as e:
            return JsonResponse({'error': f'File processing error: {str(e)}'}, status=500)
            
    return JsonResponse({'error': 'Invalid request'}, status=400)

@require_POST
def delete_document(request):
    """Delete a document and its associated vector store data"""
    try:
        document_id = request.POST.get('document_id')
        if not document_id:
            return JsonResponse({'error': 'Document ID is required'}, status=400)
            
        # Get the document
        document = Document.objects.get(id=document_id)
        
        # Delete the file from media storage
        if document.file:
            document.file.delete()
        
        # Delete the document record
        document.delete()
        
        # Rebuild vector store without this document
        # First, delete existing vector store
        if os.path.exists(VECTOR_STORE_DIR):
            shutil.rmtree(VECTOR_STORE_DIR)
            os.makedirs(VECTOR_STORE_DIR)
        
        # Reprocess remaining documents
        remaining_docs = Document.objects.filter(processed=True)
        for doc in remaining_docs:
            process_pdf(doc.file)
            
        return JsonResponse({'success': 'Document deleted successfully'})
        
    except Document.DoesNotExist:
        return JsonResponse({'error': 'Document not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@require_POST
def delete_all_documents(request):
    """Delete all documents and reset the vector store"""
    try:
        # Get all documents
        documents = Document.objects.all()
        
        # Delete all files from media storage
        for document in documents:
            if document.file:
                document.file.delete()
        
        # Delete all document records
        Document.objects.all().delete()
        
        # Reset vector store
        if os.path.exists(VECTOR_STORE_DIR):
            shutil.rmtree(VECTOR_STORE_DIR)
            os.makedirs(VECTOR_STORE_DIR)
            
        return JsonResponse({'success': 'All documents deleted successfully'})
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

