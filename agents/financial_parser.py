# document_handler.py
from typing import TypedDict, Dict, Optional, List
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Import or define the state type
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
    
def financial_parser_agent(state: EarningsAnalysisState) -> EarningsAnalysisState:
    """
    Financial Parser agent that:
    1. Uses RAG to find relevant financial information
    2. Analyzes key metrics and trends
    3. Structures the findings for further analysis
    """
    try:
        print(f"\nAnalyzing financial information for {state['ticker']}...")
        
        # Get vector store from state
        vectorstore = state['document_analysis']['earnings_release']
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Define key metrics we want to analyze
        analysis_queries = [
            "What was the revenue and revenue growth in the most recent quarter?",
            "What were the EBITDA and EBITDA margins?",
            "What was mentioned about traffic or operational metrics?",
            "What is the company's guidance or outlook?",
            "What are the key balance sheet metrics and leverage ratios?"
        ]

        # Collect findings for each query
        findings = {}
        for query in analysis_queries:
            print(f"\nAnalyzing: {query}")
            
            # Get relevant document chunks
            relevant_docs = vectorstore.similarity_search(query, k=3)
            
            # Combine relevant text
            context = "\n\n".join(doc.page_content for doc in relevant_docs)
            
            # Create analysis prompt
            prompt = f"""
            Based on the following context, answer the query: {query}
            
            Only include information that is explicitly stated in the context.
            If the information is not available, say "Information not found."
            
            Context:
            {context}
            """
            
            # Get LLM response
            response = llm.invoke([HumanMessage(content=prompt)])
            findings[query] = response.content

        # Create comprehensive analysis
        analysis_prompt = f"""
        Create a credit financial analysis for {state['ticker']} based on these findings:
        
        {findings}
        
        Format the analysis as follows:
        1. Revenue Performance
        2. Profitability (EBITDA/margins)
        3. Operational Metrics
        4. Forward Guidance
        5. Balance Sheet & Leverage
        
        Be concise and focus on key metrics and their changes.
        """
        
        final_analysis = llm.invoke([HumanMessage(content=analysis_prompt)])
        
        # Update state
        state['financial_analysis'] = final_analysis.content
        state['status'] = 'parallel_analysis_needed'
        print("\nFinancial analysis completed successfully!")
        
    except Exception as e:
        state['errors'].append(f"Financial analysis error: {str(e)}")
        state['status'] = 'error'
        print(f"Error in financial analysis: {str(e)}")
    
    return state