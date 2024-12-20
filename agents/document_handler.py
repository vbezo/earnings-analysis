def document_handler_agent(state: EarningsAnalysisState) -> EarningsAnalysisState:
    """
    Document handler agent with RAG capabilities:
    1. Loads PDF documents
    2. Splits them into chunks
    3. Creates vector embeddings
    4. Stores in FAISS for efficient retrieval
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
            
            # Load PDF
            loader = PyPDFLoader(file_path)
            pages = loader.load()
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
        state['status'] = 'financial_analysis_needed'
        print("\nDocument processing completed successfully!")
        
    except Exception as e:
        state['errors'].append(f"Document processing error: {str(e)}")
        state['status'] = 'error'
        print(f"Error in document processing: {str(e)}")
    
    return state