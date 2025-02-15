# Health Insurance Document Assistant

A Django-based web application that serves as an intelligent health insurance document management and chat system. The application uses AI to process insurance documents and provide intelligent responses to user queries.

## Features

### 1. Document Management
- PDF document upload and processing
- Automatic text extraction from PDFs
- Storage of processed documents
- Document deletion (single/bulk)
- Vector-based document storage for efficient retrieval

### 2. Interactive Chat Interface
- Real-time chat with AI assistant
- Voice input support
- Text-to-Speech response capability
- Context-aware responses based on uploaded documents
- Support for both insurance-specific and general queries

### 3. Document Analysis
- Automatic extraction of key insurance information:
  - Policy Type
  - Coverage Amount
  - Premium Amount
  - Policy Period
  - Key Benefits
  - Exclusions
- Vector-based similarity search
- Document preview and metadata display

## Technology Stack

### Backend Framework
- Django 4.2.19
- Python 3.x

### AI and Machine Learning
- OpenAI GPT-3.5 (for chat and document analysis)
- LangChain (for document processing and chat)
- FAISS (for vector storage and similarity search)
- Tiktoken (for token management)

### Document Processing
- PyPDF2 (PDF text extraction)
- Tesseract OCR (image-based text extraction)
- pdf2image (PDF to image conversion)
- Poppler (PDF processing)

### Speech Processing
- Sarvam AI API (Speech-to-Text)
- OpenAI TTS API (Text-to-Speech)

### Frontend
- HTML5
- CSS3
- JavaScript (Vanilla)
- Fetch API for AJAX requests

## Key Dependencies
python
langchain==0.1.0
langchain-community==0.0.10
openai==1.0.0
faiss-cpu==1.7.4
tiktoken==0.5.1
PyPDF2==3.0.0
pytesseract==0.3.10
pdf2image==1.16.3
requests==2.31.0
python-dotenv==1.0.0

## Project Structure

### Core Components
1. Main Application (`main/`)
   - Views for handling requests
   - Models for data storage
   - Utils for document processing
   - URL routing

2. Templates (`templates/main/`)
   - Chat interface
   - Document upload interface
   - Document viewer

3. Static Files
   - CSS styles
   - JavaScript functions
   - Media storage

## User Flow

1. **Document Management**
   - User uploads insurance documents (PDF format)
   - System processes and extracts information
   - Documents are stored in vector database
   - User can view or delete stored documents

2. **Chat Interaction**
   - User can type questions or use voice input
   - System classifies query (insurance-specific or general)
   - AI provides context-aware responses
   - Responses can be played back as audio

## Setup and Installation

1. Clone the repository

```bash
git clone [repository-url]
cd health-insurance-agent
```

2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Install system dependencies
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr poppler-utils

# macOS
brew install tesseract poppler

# Windows
# Download and install Tesseract and Poppler manually
```

5. Set up environment variables
```bash
OPENAI_API_KEY=your_key_here
SARVAM_API_KEY=your_key_here
```

6. Run migrations
```bash
python manage.py migrate
```

7. Start the development server
```bash
python manage.py runserver
```

## Development Notes

### Security Considerations
- Secret key management
- Document access control
- API key protection
- CSRF protection implemented

### Performance Optimizations
- Vector-based document storage
- Efficient document chunking
- Caching mechanisms
- Asynchronous processing

### Future Enhancements
- User authentication
- Document version control
- Enhanced document analysis
- Multi-language support
- Real-time collaboration

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

