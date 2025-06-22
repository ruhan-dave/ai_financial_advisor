import os
from dotenv import load_dotenv

from langchain_community.llms import Ollama
# import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import OllamaEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# If using Ollama embeddings, use: from langchain_community.embeddings import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from tqdm.auto import tqdm

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, Runnable
from langchain_core.output_parsers import PydanticOutputParser
from operator import itemgetter # Useful for extracting nested dictionary values

load_dotenv()  # Load environment variables from .env file

# --- Configuration (Ensure these are set) ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "YOUR_PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", "us-east-1")

# Add a check for LANGCHAIN_API_KEY
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
if not langchain_api_key:
    raise ValueError("LANGCHAIN_API_KEY environment variable not set. Please set it in your .env file or environment.")
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGSMITH_PROJECT")

if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
    raise ValueError("Please set PINECONE_API_KEY and PINECONE_ENVIRONMENT environment variables.")

# ———— Ollama embeddings ————:
llm = Ollama(model="llama3")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# ———— Pinecone Setup ————
index_name = "financial-advisor-index"
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
dimension = 1024
metric = "cosine"

# if index_name not in pc.list_indexes():
#     print(f"Creating index '{index_name}'...")
#     spec = ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
#     pc.create_index(name=index_name, dimension=dimension, metric=metric, spec=spec)
#     print(f"Index '{index_name}' created successfully.")
# else:
#     print(f"Index '{index_name}' already exists.")

vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# ———— 1. Define the Schema for Extracted Information ————
class FinancialDetails(BaseModel):
    goals: Optional[str] = Field(None, description="The user's primary financial goals (e.g., 'eliminate debt', 'save for retirement').")
    income_level: Optional[str] = Field(None, description="The user's current income level (e.g., '$2000 per month', 'unknown').")
    time_frame: Optional[str] = Field(None, description="The desired time frame to achieve the goals (e.g., '4 months', '5 years').")
    blockages: Optional[str] = Field(None, description="Any perceived roadblocks or concerns preventing the user from achieving their goals (e.g., 'high interest rates', 'lack of savings').")

# ———— 2. Create the Pydantic Output Parser ————
parser = PydanticOutputParser(pydantic_object=FinancialDetails)

# ———— 3. Define the Extraction Prompt with Format Instructions ————
extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at extracting financial details from user queries. Your sole task is to output a JSON object containing the extracted information in string format. If a piece of info is not provided, return 'unknown' for that field. Ensure the JSON is valid and contains no other explanational text outside of the JSON object. Ensure to always return string to string mappings, not any lists, inside the JSON object.\n{format_instructions}"),
    ("human", "User query: {query}")
])

# ———— 4. Create the Extraction Chain ————
extraction_chain = (
    extraction_prompt.partial(format_instructions=parser.get_format_instructions())
    | llm
    | parser
)

# ———— 5. Define your Main Financial Advice Prompt ————
full_prompt_string = """
You are a financial assistant aimed at creating constructive, clear, and sound advice or tips to help achieve the user's goals.
Refer to the current conversation summary for previously established details: {chat_history_summary}

{conditional_guidance}

Based on the user's current query, and considering the current conversation summary, you should take the user's {goals}, {income_level}, and desired {time_frame} to achieve them into consideration.
Be sure to address their {blockages}. Offer realistic goal-setting and be sure to suggest a few changes to the plan if deemed beneficial for the user.
Refer to given {context} to guide your constructive advice.
"""

financial_advice_prompt = PromptTemplate(
    input_variables=["goals", "income_level", "time_frame", "blockages", "context", "chat_history_summary", "conditional_guidance"],
    template=full_prompt_string
)

# ———— 6. Context Formatting Function ————
def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

# A class to hold and update our session state
class ConversationState(BaseModel):
    # hold the consolidated financial details
    financial_details: FinancialDetails = Field(default_factory=FinancialDetails)
    # hold the original user question for each turn
    # might want to store more, like the LLM's response, if needed
    history: List[Dict[str, str]] = Field(default_factory=list) # Store Q&A pairs if needed

    # A method to update financial details
    def update_financial_details(self, new_details: FinancialDetails):
        # Merge new details into existing details
        for field in new_details.model_fields:
            new_value = getattr(new_details, field)
            if new_value is not None: # Only update if new_details provides a non-null value
                setattr(self.financial_details, field, new_value)
        return self

    # A method to generate a summary for the LLM
    def get_summary_for_llm(self) -> str:
        summary_parts = []
        details = self.financial_details.model_dump(exclude_none=True) # Get non-null fields as dict
        for key, value in details.items():
            if value: # Ensure value is not empty string either
                summary_parts.append(f"{key.replace('_', ' ').capitalize()}: {value}")
        if not summary_parts:
            return "No specific financial details have been established yet."
        return "Current financial details established in conversation:\n- " + "\n- ".join(summary_parts)


# This function will be the entry point for the chain, managing the session.
def manage_state_and_prepare_inputs(input_dict: Dict[str, Any], state: ConversationState) -> Dict[str, Any]:
    user_query = input_dict["question"]

    # 1. Extract new details from the current user query
    newly_extracted_details = extraction_chain.invoke({"query": user_query})

    # 2. Update the session state with new details
    state.update_financial_details(newly_extracted_details)

    # 3. Get the summary for the LLM
    chat_history_summary = state.get_summary_for_llm()

    # 4. Check for missing critical information and formulate a follow-up question
    missing_fields = []
    if state.financial_details.goals == "unknown" or state.financial_details.goals is None:
        missing_fields.append("your main financial goals")
    if state.financial_details.income_level == "unknown" or state.financial_details.income_level is None:
        missing_fields.append("your current income level")
    if state.financial_details.time_frame == "unknown" or state.financial_details.time_frame is None:
        missing_fields.append("your desired time frame for these goals")
    # You might consider 'blockages' as less critical for the *initial* information gathering
    # but could add it here if it's essential for a first pass.

    missing_info_question = None
    if missing_fields:
        if len(missing_fields) == 1:
            missing_info_question = f"Could you please tell me about {missing_fields[0]}?"
        elif len(missing_fields) == 2:
            missing_info_question = f"Could you please tell me about {missing_fields[0]} and {missing_fields[1]}?"
        else:
            missing_info_question = f"To give you the best advice, could you please provide more details on {', '.join(missing_fields[:-1])}, and {missing_fields[-1]}?"

    # 5. Prepare the final input dictionary for the financial_advice_prompt
    inputs_for_rag = {
        "question": user_query, # Original question for the retriever
        "goals": state.financial_details.goals or "not provided", # Use "not provided" for prompt
        "income_level": state.financial_details.income_level or "not provided",
        "time_frame": state.financial_details.time_frame or "not provided",
        "blockages": state.financial_details.blockages or "not provided",
        "chat_history_summary": chat_history_summary,
        "missing_info_question": missing_info_question # Pass this along
    }
    return inputs_for_rag

# --- 7. Build the Complete RAG Chain with Extraction and History ---
conversation_state = ConversationState()

# Define a function to conditionally return the question or full advice
def decide_output(inputs: Dict[str, Any]) -> str:
    if inputs.get("missing_info_question"):
        return inputs["missing_info_question"]
    else:
        # If no missing info, prepare the inputs for the main financial advice prompt
        # and then run the rest of the chain.
        # This part effectively recreates the flow that would normally happen *after*
        # manage_state_and_prepare_inputs, but we do it conditionally.

        # Retrieve context here, as it's only needed if we're giving advice
        context = (itemgetter("question") | vectorstore.as_retriever() | format_docs).invoke(inputs)

        # Prepare a new dictionary that matches the financial_advice_prompt's input_variables
        # without the 'missing_info_question' and with 'conditional_guidance'
        prompt_inputs = {
            "goals": inputs["goals"],
            "income_level": inputs["income_level"],
            "time_frame": inputs["time_frame"],
            "blockages": inputs["blockages"],
            "context": context,
            "chat_history_summary": inputs["chat_history_summary"],
            "conditional_guidance": "Now, generate the financial advice based on the provided information and context."
        }
        return (financial_advice_prompt | llm | StrOutputParser()).invoke(prompt_inputs)


# The full chain now has a branching logic
full_rag_chain = (
    RunnableLambda(lambda x: manage_state_and_prepare_inputs(x, conversation_state))
    | RunnableLambda(decide_output) # This is the new conditional step
)
# The chain now needs to be wrapped to manage the state.
# We'll use a `RunnableLambda` to call our state management function,
# and then chain the rest.

# Create a global (or per-session) state instance
# In a real application, this would be managed per user session.
# For demonstration, it's global.
conversation_state = ConversationState()

# The full chain that takes a 'question' and updates 'conversation_state'
full_rag_chain = (
    RunnableLambda(lambda x: manage_state_and_prepare_inputs(x, conversation_state))
    .assign(
        # Context retrieval still depends on the 'question' from the initial input
        context=itemgetter("question") | vectorstore.as_retriever() | format_docs,
        # The other variables (goals, income_level, etc.) are already in the dict
        # returned by manage_state_and_prepare_inputs
    )
    # The financial_advice_prompt expects all the keys
    | financial_advice_prompt
    | llm
    | StrOutputParser()
)

# --- Example Usage ---
if __name__ == "__main__":
    # First turn:
    print("--- User 1 ---")
    result1 = full_rag_chain.invoke({
        "question": "I want to eliminate my debt in 4 months. I have a monthly income of $2000 and my current debt is $5000. I am concerned about my high interest rates."
    })
    print(f"Assistant 1: {result1}")
    print(f"\n--- Current State After Turn 1 ---")
    print(conversation_state.get_summary_for_llm())


    # Second turn: Clarify goals, some details might be missing
    print("\n--- User 2 ---")
    result2 = full_rag_chain.invoke({
        "question": "My main goal is to be debt-free. Are there any specific apps that can help?"
    })
    print(f"Assistant 2: {result2}")
    print(f"\n--- Current State After Turn 2 ---")
    print(conversation_state.get_summary_for_llm())

    # Third turn: Add a new detail
    print("\n--- User 3 ---")
    result3 = full_rag_chain.invoke({
        "question": "I also have some issues with impulsive spending. What's your advice?"
    })
    print(f"Assistant 3: {result3}")
    print(f"\n--- Current State After Turn 3 ---")
    print(conversation_state.get_summary_for_llm())

    # Fourth turn: Check state for asking about missing info if user doesn't provide.
    # The prompt should intelligently ask, based on `not provided` values.
    print("\n--- User 4 ---")
    result4 = full_rag_chain.invoke({
        "question": "Give me some budgeting tips."
    })
    print(f"Assistant 4: {result4}")
    print(f"\n--- Current State After Turn 4 ---")
    print(conversation_state.get_summary_for_llm())

    # Fourth turn: Check state for asking about missing info if user doesn't provide.
    # The prompt should intelligently ask, based on `not provided` values.
    print("\n--- User 5 ---")
    result4 = full_rag_chain.invoke({
        "question": "Give me some strategies to invest my savings with medium risk."
    })
    print(f"Assistant 4: {result4}")
    print(f"\n--- Current State After Turn 4 ---")
    print(conversation_state.get_summary_for_llm())