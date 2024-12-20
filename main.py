from typing import TypedDict, Dict, Optional, List
from pathlib import Path
from datetime import datetime
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage
from IPython.display import Image, display

# Import agents
from agents.document_handler import document_handler_agent
from agents.financial_parser import financial_parser_agent
from agents.credit_analyst import credit_analyst_agent
from agents.industry_expert import industry_expert_agent
from agents.summary import summary_agent

# Define state
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

# Routing functions
def route_after_financial(state: EarningsAnalysisState):
    """Determine routing after financial analysis"""
    if state["status"] == "financial_analysis_complete":
        return ["credit_analyst", "industry_expert"]
    return "__end__"

def should_summarize(state: EarningsAnalysisState):
    """Determine if we should move to summary"""
    if state["credit_analysis"] and state["industry_analysis"]:
        return "summary"
    return "__end__"

def create_workflow():
    """
    Creates and configures the workflow graph
    """
    # Create workflow
    builder = StateGraph(EarningsAnalysisState)

    # Add nodes
    builder.add_node("document_handler", document_handler_agent)
    builder.add_node("financial_parser", financial_parser_agent)
    builder.add_node("credit_analyst", credit_analyst_agent)
    builder.add_node("industry_expert", industry_expert_agent)
    builder.add_node("summary", summary_agent)

    # Add edges
    builder.add_edge(START, "document_handler")
    builder.add_edge("document_handler", "financial_parser")

    # Add conditional edges
    builder.add_conditional_edges(
        "financial_parser",
        route_after_financial,
        ["credit_analyst", "industry_expert", "__end__"]
    )

    builder.add_conditional_edges(
        "credit_analyst",
        should_summarize,
        ["summary", "__end__"]
    )
    builder.add_conditional_edges(
        "industry_expert",
        should_summarize,
        ["summary", "__end__"]
    )

    builder.add_edge("summary", "__end__")

    # Compile the graph
    workflow = builder.compile()

    # Display the graph
    display(Image(workflow.get_graph(xray=1).draw_mermaid_png()))

    return workflow

def run_analysis():
    """
    Run the earnings analysis workflow with user input for ticker and industry
    """
    try:
        # Get user input
        ticker = input("Enter company ticker: ").strip().upper()
        industry = input("Enter company industry: ").strip()
        
        # Create workflow
        workflow = create_workflow()
        
        # Initialize state
        initial_state = {
            "ticker": ticker,
            "industry": industry,
            "documents": {},
            "document_analysis": None,
            "financial_analysis": None,
            "credit_analysis": None,
            "industry_analysis": None,
            "final_comment": None,
            "status": "start",
            "errors": []
        }
        
        print(f"\nStarting analysis for {ticker} ({industry})...")
        result = workflow.invoke(initial_state)
        
        if result["status"] == "complete":
            print(f"\nAnalysis completed successfully for {ticker}")
            print("\nFinal Credit Comment:")
            print("-" * 50)
            print(result["final_comment"])
            
            # Create output directory if it doesn't exist
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            
            # Save to file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            filename = output_dir / f"{ticker}_analysis_{timestamp}.txt"
            
            with open(filename, "w", encoding='utf-8') as f:
                f.write(f"=============================================================\n")
                f.write(f"CREDIT COMMENT: {ticker}\n")
                f.write(f"=============================================================\n")
                f.write(f"Generated Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
                f.write(f"Industry: {industry}\n\n")
                f.write("SUMMARY\n")
                f.write("-------\n")
                f.write(f"{result['final_comment']}\n\n")
                f.write("DETAILED ANALYSIS\n")
                f.write("----------------\n")
                f.write(f"Financial Analysis:\n{result['financial_analysis']}\n\n")
                f.write(f"Credit Analysis:\n{result['credit_analysis']}\n\n")
                f.write(f"Industry Context:\n{result['industry_analysis']}\n")
                f.write("=============================================================\n")
            
            print(f"\nAnalysis saved to: {filename}")
        else:
            print(f"\nAnalysis failed for {ticker}")
            print("Errors:", result["errors"])
        
        return result
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    while True:
        result = run_analysis()
        
        # Ask if user wants to analyze another company
        again = input("\nWould you like to analyze another company? (y/n): ").lower().strip()
        if again != 'y':
            break
            
    print("\nAnalysis complete. Thank you!")