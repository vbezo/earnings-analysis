def industry_expert_agent(state: EarningsAnalysisState) -> EarningsAnalysisState:
    """
    Industry Expert agent that:
    1. Analyzes company's market position
    2. Evaluates competitive dynamics
    3. Provides sector context and trends
    4. Assesses performance vs peers
    """
    try:
        print(f"\nPerforming industry analysis for {state['ticker']} in {state['industry']} sector...")
        
        # Get vector store and initialize LLM
        vectorstore = state['document_analysis']['earnings_release']
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Define industry-specific queries
        industry_queries = [
            "What is mentioned about market position and market share?",
            "What are the key industry trends or market dynamics mentioned?",
            "What is discussed about competition or competitive advantages?",
            "What operational or industry-specific metrics are highlighted?",
            "What is mentioned about industry outlook or sector challenges?"
        ]

        # Collect industry insights
        findings = {}
        for query in industry_queries:
            print(f"\nAnalyzing: {query}")
            
            # Get relevant document chunks
            relevant_docs = vectorstore.similarity_search(query, k=3)
            context = "\n\n".join(doc.page_content for doc in relevant_docs)
            
            # Create analysis prompt
            prompt = f"""
            You are an industry expert analyzing {state['ticker']}'s position in the {state['industry']} sector.
            Based on the following context, answer the query: {query}
            
            Consider industry-specific factors and competitive dynamics.
            Only include information that is explicitly stated in the context.
            If the information is not available, say "Information not found."
            
            Context:
            {context}
            
            Previous analyses:
            Financial Analysis: {state['financial_analysis']}
            Credit Analysis: {state['credit_analysis']}
            """
            
            response = llm.invoke([HumanMessage(content=prompt)])
            findings[query] = response.content

        # Create comprehensive industry analysis
        industry_prompt = f"""
        As an industry expert in the {state['industry']} sector, create a comprehensive analysis for {state['ticker']} based on:

        Findings:
        {findings}

        Previous Analyses:
        Financial Analysis: {state['financial_analysis']}
        Credit Analysis: {state['credit_analysis']}

        Please structure your analysis as follows:
        1. Market Position
           - Market share
           - Competitive advantages
           - Brand strength
        2. Industry Dynamics
           - Current trends
           - Sector challenges
           - Growth drivers
        3. Competitive Analysis
           - Key competitors
           - Relative performance
           - Competitive threats
        4. Operational Excellence
           - Industry-specific metrics
           - Operational efficiency
           - Best practices
        5. Sector Outlook
           - Industry trends
           - Growth opportunities
           - Key risks

        Focus on providing industry context to the financial and credit metrics.
        Highlight any sector-specific insights that impact credit quality.
        """
        
        final_industry_analysis = llm.invoke([HumanMessage(content=industry_prompt)])
        
        # Update state
        state['industry_analysis'] = final_industry_analysis.content
        
        # Only update status if credit analysis is already complete
        if state['credit_analysis']:
            state['status'] = 'summary_needed'
        
        print("\nIndustry analysis completed successfully!")
        
    except Exception as e:
        state['errors'].append(f"Industry analysis error: {str(e)}")
        state['status'] = 'error'
        print(f"Error in industry analysis: {str(e)}")
    
    return state