import os
from langchain_community.chat_models import Ollama
from langchain_aws import BedrockLLM
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

TOOL_PROMPT = """
Decide whether the following user query needs to operate tools on the uploaded Excel file ("excel_pipeline")
or use the general RAG tools ("rag_pipeline"). 
Respond ONLY with valid JSON: {{"next": "excel_pipeline"}} or {{"next": "rag_pipeline"}}.

Query: "{query}"
"""

def lambda_handler(event, context):
    query = event.get("query", "")
    if not query:
        return {"error": "No query provided"}

    # Initialize Bedrock (Llama 3, Claude, or Mistral)
    llm = BedrockLLM(
        model_id="meta.llama3-8b-instruct-v1:0",  # or "anthropic.claude-3-sonnet-20240229-v1:0"
        region_name="us-east-1"
    )

    # Create the LangChain pipeline
    prompt = ChatPromptTemplate.from_messages([
        ("system", TOOL_PROMPT),
        ("user", "Query: {query}")
    ])    
    
    # Create chain
    output_parser = JsonOutputParser()
    chain = prompt | llm | output_parser
    
    # Invoke the chain
    try:
        tool_obj = chain.invoke({"query": query})
        return {"next": tool_obj.get("next", "search")}  # Default to "search" if not found
    except Exception as e:
        return {"error": f"Failed to process query: {str(e)}"}