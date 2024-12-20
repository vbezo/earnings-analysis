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
    
def summary_agent(state: EarningsAnalysisState) -> EarningsAnalysisState:
    """
    Summary agent that:
    1. Combines insights from all previous analyses
    2. Creates concise credit comment
    3. Follows standardized format
    4. Highlights key changes and risks
    """
    try:
        print(f"\nGenerating final credit comment for {state['ticker']}...")
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Create comprehensive summary prompt
        summary_prompt = f"""
        Create a concise credit comment for {state['ticker']} combining all analyses:

        Financial Analysis:
        {state['financial_analysis']}

        Credit Analysis:
        {state['credit_analysis']}

        Industry Analysis:
        {state['industry_analysis']}

        Requirements:
        1. Follow this exact structure:
           - Start with revenue/sales performance
           - Then margins/profitability
           - Then operational metrics
           - Then guidance/outlook
           - End with leverage/credit metrics
        2. Keep it concise (max 200 words)
        3. Focus on YoY changes and trends
        4. Include specific numbers
        5. Highlight any key risks or improvements

        Example format:
        "Net sales +X% yoy with margins [improved/stable/declined]. [Key operational metric] was [positive/negative]. 
        The company [raised/maintained/lowered] guidance: [specifics]. Net leverage [increased/decreased] to X.X times."
        """
        
        final_comment = llm.invoke([HumanMessage(content=summary_prompt)])
        
        # Create more structured filepath
        base_path = Path("output")  # Base directory for all output
        company_path = base_path / state['ticker']  # Company-specific directory
        year_path = company_path / datetime.now().strftime("%Y")  # Year directory
        
        # Create all directories
        year_path.mkdir(parents=True, exist_ok=True)
        
        # Create filename with quarter information
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = year_path / f"{state['ticker']}_CreditComment_{timestamp}.txt"
        
        # Enhanced format for the credit comment file
        content = f"""
=============================================================
CREDIT COMMENT: {state['ticker']}
=============================================================
Generated Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Industry: {state['industry']}

SUMMARY
-------
{final_comment.content}

DETAILED ANALYSIS
----------------
Financial Analysis:
------------------
{state['financial_analysis']}

Credit Analysis:
---------------
{state['credit_analysis']}

Industry Context:
----------------
{state['industry_analysis']}
=============================================================
"""
        
        # Save to file
        filename.write_text(content)
        
        # Update state
        state['final_comment'] = final_comment.content
        state['status'] = 'complete'
        
        print("\nCredit comment generated successfully!")
        print(f"Saved to: {filename}")
        
    except Exception as e:
        state['errors'].append(f"Summary error: {str(e)}")
        state['status'] = 'error'
        print(f"Error in summary generation: {str(e)}")
    
    return state