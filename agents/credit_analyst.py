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
    
def credit_analyst_agent(state: EarningsAnalysisState) -> EarningsAnalysisState:
    """
    Credit Analyst agent that:
    1. Analyzes credit metrics and financial health
    2. Evaluates leverage and coverage
    3. Assesses liquidity position
    """
    try:
        print(f"\nPerforming credit analysis for {state['ticker']}...")
        
        # Get vector store and initialize LLM
        vectorstore = state['document_analysis']['earnings_release']
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Define credit-specific queries
        credit_queries = [
            "What are the current leverage ratios and how have they changed?",
            "What is mentioned about debt structure, maturities, and interest coverage?",
            "What are the working capital and liquidity metrics?",
            "What is mentioned about cash flow and cash generation?",
            "What are the capital allocation priorities and any mentioned refinancing plans?"
        ]

        # Collect credit findings
        findings = {}
        for query in credit_queries:
            print(f"\nAnalyzing: {query}")
            
            # Get relevant document chunks
            relevant_docs = vectorstore.similarity_search(query, k=3)
            context = "\n\n".join(doc.page_content for doc in relevant_docs)
            
            # Create analysis prompt
            prompt = f"""
            You are a credit analyst focusing on {state['ticker']}'s financial health.
            Based on the following context, answer the query: {query}
            
            Only include information that is explicitly stated in the context.
            If the information is not available, say "Information not found."
            When possible, provide specific numbers and compare to previous periods.
            
            Context:
            {context}
            
            Previous financial analysis:
            {state['financial_analysis']}
            """
            
            response = llm.invoke([HumanMessage(content=prompt)])
            findings[query] = response.content

        # Create comprehensive credit analysis
        credit_prompt = f"""
        As a credit analyst, create a detailed credit assessment for {state['ticker']} based on:

        Findings:
        {findings}

        Previous Financial Analysis:
        {state['financial_analysis']}

        Please structure your analysis as follows:
        1. Leverage Analysis
           - Current ratios
           - Trends and changes
        2. Debt Structure & Coverage
           - Debt composition
           - Interest coverage
        3. Liquidity Position
           - Working capital
           - Cash position
        4. Cash Flow Analysis
           - Operating cash flow trends
           - Cash conversion
        5. Credit Outlook
           - Key strengths
           - Main risks
           - Overall trend

        Focus on credit metrics and their implications for financial health.
        Be specific about numbers but also provide analytical insights.
        """
        
        final_credit_analysis = llm.invoke([HumanMessage(content=credit_prompt)])
        
        # Update state
        state['credit_analysis'] = final_credit_analysis.content
        
        # Only update status if industry analysis is already complete
        if state['industry_analysis']:
            state['status'] = 'summary_needed'
        
        print("\nCredit analysis completed successfully!")
        
    except Exception as e:
        state['errors'].append(f"Credit analysis error: {str(e)}")
        state['status'] = 'error'
        print(f"Error in credit analysis: {str(e)}")
    
    return state