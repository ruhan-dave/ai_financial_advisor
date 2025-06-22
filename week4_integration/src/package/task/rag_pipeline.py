import os
import json
import boto3
from typing import Optional, List, Dict, Any
# from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_aws import BedrockLLM
from langchain_pinecone import PineconeVectorStore
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from operator import itemgetter
from pinecone import Pinecone
from botocore.exceptions import ClientError


# Initialize all required components (should be outside handler for cold start)
# load_dotenv()

# Initialize AWS, Bedrock and HuggingFace
aws_region = os.environ.get("AWS_REGION", "us-east-1")  # aws_region = os.getenv("AWS_REGION", "us-east-1") Ensure AWS_REGION is set
model_id = "meta.llama3-8b-instruct-v1:0"

# 1. Bedrock LLM
bedrock = boto3.client('bedrock-runtime', region_name=aws_region)

# Custom LLM wrapper for Bedrock
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize DynamoDB
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ['DYNAMODB_TABLE'])

class BedrockWrapper:
    def invoke(self, input: str) -> str:
        body = {
            "prompt": input,
            "max_gen_len": 2050,
            "temperature": 0.5
        }
        try:
            response = bedrock.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
                contentType="application/json"
            )
            raw = response['body'].read()
            data = json.loads(raw)
            # Validate response structure
            if 'generation' in data and isinstance(data['generation'], str):
                return data['generation']
            else:
                logger.error(f"Unexpected Bedrock response: {data}")
                return "Sorry, I couldn't generate a response at this time."
        except Exception as e:
            logger.error(f"Bedrock invocation failed: {e}")
            return "Sorry, there was an error generating a response."
        
# Initialize LLM
llm = BedrockWrapper()

# 2. HuggingFace Embeddings (Sentence Transformers)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": 32  # Better for Lambda
    },
    cache_folder="/tmp/hf_models"  # Lambda ephemeral storage
)

# 3. Pinecone Vector Store

# Usage at cold start:
def get_secret(secret_name):
    client = boto3.client('secretsmanager')
    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        secret = get_secret_value_response['SecretString']
        return json.loads(secret)
    except Exception as e:
        logger.error(f"Secrets Manager error: {e}")
        raise e
    
secrets = get_secret(os.environ['SECRETS_MANAGER_SECRET_NAME'])
pinecone_api_key = secrets['PINECONE_API_KEY']

index_name = "financial-advisor-index"
vectorstore = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings,
    pinecone_api_key=pinecone_api_key
)

# 4. Define FinancialDetails class (from your original code)
class FinancialDetails(BaseModel):
    goals: Optional[str] = Field(None, description="The user's primary financial goals")
    income_level: Optional[str] = Field(None, description="The user's current income level")
    time_frame: Optional[str] = Field(None, description="The desired time frame to achieve the goals")
    blockages: Optional[str] = Field(None, description="Any perceived roadblocks")
    
# 5. Create conversation state manager
class ConversationState(BaseModel):
    """
    Pydantic model that holds the user's financial details and chat history, and 
    provides methods to update and summarize them.
    You use it for state management and persistence in DynamoDB.
    """
    financial_details: FinancialDetails = Field(default_factory=FinancialDetails)
    history: List[Dict[str, str]] = Field(default_factory=list)

    def update_financial_details(self, new_details: FinancialDetails):
        for field in new_details.model_fields:
            new_value = getattr(new_details, field)
            if new_value is not None:
                setattr(self.financial_details, field, new_value)
        return self

    def get_summary_for_llm(self) -> str:
        summary_parts = []
        details = self.financial_details.model_dump(exclude_none=True)
        for key, value in details.items():
            if value:
                summary_parts.append(f"{key.replace('_', ' ').capitalize()}: {value}")
        return "Current financial details:\\n- " + "\\n- ".join(summary_parts) if summary_parts else "No financial details established yet."

def load_conversation_state(user_id: str) -> ConversationState:
    try:
        response = table.get_item(Key={'user_id': user_id})
        if 'Item' in response:
            state_data = response['Item']['state']
            return ConversationState(**json.loads(state_data))
        else:
            return ConversationState()
    except ClientError as e:
        logger.error(f"DynamoDB get_item failed: {e}")
        return ConversationState()

def save_conversation_state(user_id: str, state: ConversationState):
    try:
        table.put_item(Item={
            'user_id': user_id,
            'state': state.model_dump_json()
        })
    except ClientError as e:
        logger.error(f"DynamoDB put_item failed: {e}")

# 6. Define the extraction chain components
parser = PydanticOutputParser(pydantic_object=FinancialDetails)

# Define the extraction chain components
extraction_prompt = ChatPromptTemplate.from_template("""
    Extract financial details from this query:
    {query}

    Return JSON with:
    - goals (string)
    - income_level (string)
    - time_frame (string)
    - blockages (string)
    Use 'unknown' for missing fields.
    """)

extraction_chain = (
    extraction_prompt.partial(format_instructions=parser.get_format_instructions())
    | llm
    | parser
)

# 7. Define the main financial advice prompt
financial_advice_prompt = ChatPromptTemplate.from_template("""
You are a financial assistant providing constructive advice based on:
{chat_history_summary}

User's Goals: {goals}
Income Level: {income_level}
Time Frame: {time_frame}
Blockages: {blockages}

Relevant Context:
{context}

Generate personalized advice addressing all these aspects:
""")

# 8. Document formatting function and state management function
def format_docs(docs):
    return "\\n\\n".join(doc.page_content for doc in docs)

def manage_state_and_prepare_inputs(input_dict: Dict[str, Any], conversation_state: ConversationState) -> Dict[str, Any]:
    """
    Updated function that takes conversation_state as a parameter instead of using global state
    """
    user_query = input_dict["question"]
    
    # Extract and update financial details
    new_details = extraction_chain.invoke({"query": user_query})
    conversation_state.update_financial_details(new_details)
    
    # Prepare inputs for RAG
    return {
        "question": user_query,
        "goals": conversation_state.financial_details.goals or "not specified",
        "income_level": conversation_state.financial_details.income_level or "not specified",
        "time_frame": conversation_state.financial_details.time_frame or "not specified",
        "blockages": conversation_state.financial_details.blockages or "not specified",
        "chat_history_summary": conversation_state.get_summary_for_llm()
    }

# 9. Create RAG chain factory function that accepts conversation state
def create_rag_chain(conversation_state: ConversationState):
    """
    Factory function to create RAG chain with conversation state
    """
    return (
        RunnableLambda(lambda input_dict: manage_state_and_prepare_inputs(input_dict, conversation_state))
        .assign(
            context=itemgetter("question") | vectorstore.as_retriever() | format_docs
        )
        | financial_advice_prompt
        | llm  # Now using Bedrock
        | StrOutputParser()
    )