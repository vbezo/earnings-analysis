# Earnings Analysis Workflow

This project uses LangGraph to create a multi-agent workflow for analyzing company earnings announcements. The workflow processes earnings documents, performs financial and credit analysis, and generates structured credit comments.

## Setup

1. Clone the repository
```bash
git clone https://github.com/yourusername/earnings-analysis.git
cd earnings-analysis
```
2. install dependensies
pip install -r requirements.txt

3.set up environment variables and add your openai api key: OPENAI_API_KEY=your_key_here

Usage
Run the analysis by running main.py. The program will:

Ask for company ticker and industry
Request path to earnings PDF
Process and analyze the document
Generate a structured credit comment
Save the analysis to the output folder



THERE IS ALSO A YNPB FILE TO RUN THE CODE FROM 1 JUPYER NOTEBOOK