# Insights AI - An AI-Powered PDF Document Question-Answering System 🤖

A sophisticated document analysis system that uses RAG (Retrieval-Augmented Generation) to enable intelligent question-answering from PDF documents. Built with Streamlit and powered by the Deepseek-R1 model.

## 🌟 Features

- **PDF Processing**: Upload and analyze PDF documents seamlessly
- **Interactive Chat**: Modern, dark-themed UI for natural conversation
- **Intelligent Responses**: Context-aware answers using RAG technology
- **Real-time Analysis**: Immediate document processing and response generation

## 🎯 Use Cases

- Document research and analysis
- Educational material review
- Technical documentation Q&A
- Legal document analysis
- Research paper review

## 🛠️ Prerequisites

- Python 3.8 or higher
- 8GB RAM (minimum)
- Ollama installed
- Deepseek model

## 💻 Tech Stack

- **Frontend**: Streamlit
- **Language Model**: Ollama (Deepseek-r1:1.5b)
- **PDF Processing**: PDFPlumber
- **RAG Framework**: LangChain
- **Document Storage**: Local filesystem

## 📁 Project Structure

- pdf-qa-system/
├── deep_rag.py # Main application file
├── requirements.txt # Python dependencies
├── document_store/ # Document storage directory
│ └── pdfs/ # PDF storage
└── README.md # Project documentation


## 🔧 Configuration

The application uses default settings optimized for most use cases. Key configurations:

- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Model**: Deepseek-r1:1.5b
- **Temperature**: 0.7

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.

## 📧 Contact - https://www.linkedin.com/in/maddula-pavan/

Built with ❤️ by Pavan Maddula
