import os

from django.conf import settings
from django.shortcuts import render
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage

from .models import ChatMessage

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
        user_message = request.POST.get('message')
        
        # Save user message to database
        ChatMessage.objects.create(role='user', content=user_message)
        
        # Initialize ChatOpenAI
        chat = ChatOpenAI(
            temperature=0.7,
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            model_name="gpt-3.5-turbo"
        )
        
        # Get chat history
        messages = []
        for msg in ChatMessage.objects.all():
            if msg.role == 'user':
                messages.append(HumanMessage(content=msg.content))
            else:
                messages.append(AIMessage(content=msg.content))
        
        # Get AI response
        response = chat(messages)
        
        # Save AI response to database
        ChatMessage.objects.create(role='assistant', content=response.content)
    
    # Get all messages for display
    messages = ChatMessage.objects.all()
    return render(request, 'main/chat.html', {'messages': messages})
