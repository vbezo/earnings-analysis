from typing import TypedDict, Dict, Optional, List
from pathlib import Path
from langchain_openai import ChatOpenAI
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

class EarningsAnalysisState(TypedDict):
    ticker: str
    industry: str
    documents: Dict[str, str]
    document_analysis: Optional[str]
    financial_analysis: Optional[str]
    credit_analysis: Optional[str]
    industry_analysis: Optional[str]
    final_comment: Optional[str]
    status: str
    errors: List[str]

def load_pdf(file_path: str) -> List[Document]:
    """
    Custom PDF loader function
    """
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        documents = []
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            documents.append(
                Document(
                    page_content=text,
                    metadata={"page": page_num + 1, "source": file_path}
                )
            )
    return documents

def document_handler_agent(state: EarningsAnalysisState) -> EarningsAnalysisState:
    """
    Document handler agent with RAG capabilities.
    """
    try:
        print(f"Processing documents for {state['ticker']}...")
        
        # Get document path
        earnings_path = Path(input("Please enter the path to the earnings release PDF: ").strip('"'))
        
        # Validate file exists
        if not earnings_path.exists():
            raise FileNotFoundError(f"File not found: {earnings_path}")
            
        state['documents'] = {
            "earnings_release": str(earnings_path)
        }
        
        # Process documents
        documents = {}
        embeddings = OpenAIEmbeddings()
        
        for doc_type, file_path in state['documents'].items():
            print(f"\nProcessing {doc_type}...")
            
            # Load PDF using custom function
            pages = load_pdf(file_path)
            print(f"Successfully loaded {len(pages)} pages")
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            chunks = text_splitter.split_documents(pages)
            print(f"Created {len(chunks)} text chunks")
            
            # Create vector store
            vectorstore = FAISS.from_documents(chunks, embeddings)
            documents[doc_type] = vectorstore
            print(f"Successfully processed {doc_type}")
        
        state['document_analysis'] = documents
        state['status'] = 'financial_analysis_complete'
        print("\nDocument processing completed successfully!")
        
    except Exception as e:
        state['errors'].append(f"Document processing error: {str(e)}")
        state['status'] = 'error'
        print(f"Error in document processing: {str(e)}")
    
    return state