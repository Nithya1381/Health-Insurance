import os

from django.contrib import messages
from django.shortcuts import redirect, render
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage

from .models import ChatMessage, Document
from .utils import get_all_documents, get_conversation_chain, process_pdf

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
                # Get conversation chain
                conversation_chain = get_conversation_chain()
                
                # Get chat history
                chat_history = []
                for msg in ChatMessage.objects.filter(role__in=['user', 'assistant']):
                    if msg.role == 'user':
                        chat_history.append((msg.content, ''))
                
                # Get response from LLM
                response = conversation_chain({
                    "question": user_message,
                    "chat_history": chat_history
                })
                
                # Save AI response
                ChatMessage.objects.create(
                    role='assistant',
                    content=response['answer']
                )
            
            except Exception as e:
                messages.error(request, f'Error: {str(e)}')
    
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
