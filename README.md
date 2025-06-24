# ai_financial_advisor
This is an AI agent designed to answer a range of user questions related to personal financial management and basic financial analysis tasks. It is not meant to provide professional advisor assistance or investment recommendations, as it lacks knowledge of advanced topics. However, it is capable of generating practical answers.

For a week by week update, check out: https://www.notion.so/Week-by-Week-21832f83af6680098a51ea47354ebb27

## How it works 

### UI for Chat
![Front End](./image/ui.png)

### Front-End Interface

- **Layout**: The conversation history is displayed in a scrollable container from the top, flowing downward, with the input box fixed at the bottom, mimicking Grok’s style.
- **Styling**: Tailwind CSS provides a modern, responsive design. User messages are blue and right-aligned, while bot messages are gray and left-aligned.
- **Functionality**: The interface fetches conversation history on load, allows users to send messages via a button or Enter key, and disables the send button during API requests.
- **AWS Integration**: The front-end communicates with the back-end via API Gateway endpoints (/history and /ask). A user_id is stored locally for session consistency, with Cognito as an option for secure authentication.
  
### Default behavior 

The app will track the conversation by asking the user to identify the 4 fields: goals, income level, time frame, and perceived blockages. If the user does not provide this, the LLM behind the scenes will ask for clarification. Then, it will refer to a knowledge base, which contains the content from the Personal Financial Management PDF document listed in this repository. After consolidating all these pieces of info, it will then generate a helpful response to help the user achieve the goal in the time frame with the user's concerns and finaincial best practices from that particular document. 

## Architecture
**Key Features:**

- **Financial Detail Extraction:** Utilizes a Pydantic schema (`FinancialDetails`) and a dedicated extraction chain to parse user input and identify crucial financial information such as goals, income level, time frame, and perceived blockages. This structured extraction ensures that the chatbot accurately captures the user's financial situation.
- **Conversation State Management:** Employs a `ConversationState` class to maintain a persistent record of the user's financial details throughout the conversation. This allows the chatbot to remember previously provided information and build upon it in subsequent interactions, leading to more coherent and context-aware advice. The `update_financial_details` method intelligently merges new information with existing details, and `get_summary_for_llm` provides a concise summary for the language model.
- **Contextual Financial Advice:** The core of the chatbot's advice generation is driven by a `financial_advice_prompt`. This prompt is designed to incorporate the extracted financial details, the conversation history summary, and relevant context retrieved from a Pinecone vector store.
- **Retrieval-Augmented Generation (RAG):** The system integrates a RAG approach. It uses `OllamaEmbeddings` to convert user queries into numerical representations and then queries a `PineconeVectorStore` (named "financial-advisor-index") to fetch semantically similar documents (context). This retrieved context enriches the LLM's understanding and allows it to generate more informed and accurate financial advice.
- **Conditional Guidance:** The `manage_state_and_prepare_inputs` function intelligently assesses if critical financial information (goals, income level, time frame) is missing from the conversation. If so, it formulates a targeted follow-up question to gather the necessary details, ensuring the chatbot can provide comprehensive advice. This conditional logic is further managed by the `decide_output` function within the main chain.
- **LangChain Integration:** The entire system is built using the LangChain framework, leveraging its capabilities for prompt templating, output parsing, and creating complex conversational chains.
- **AWS BedRock Foundation Models and Pinecone Integration:** The chatbot utilizes AWS foundational models for its language model (`llama3`) and embeddings (`mxbai-embed-large`), and Pinecone as its vector database for efficient similarity search and retrieval.
- **Environment Variable Management:** It uses `python-dotenv` to manage API keys and other configurations securely, ensuring that sensitive information is not hardcoded directly into the script.

### API Gateway
POST /chat → StartExecutionLambda (starts Step Functions SM)
Step Functions State Machine

ClassifyTask (Lambda)
Choice:
excelBranch → ExcelPipelineTask (Lambda)
searchBranch → RAGTask (Lambda)
Merge & Refine → RefineTask (Lambda)
Return
PresignerLambda for /upload (same as Option A).

### Monitoring & Tracing
CloudWatch Logs & Metrics

LangSmith Interface for Testing

### Benefits
Built-in retry & error handling per step.
Fine-grained metrics on “Classify”, “ExcelPipeline”, “RAG”, “Refine”.
Visual state-machine flow in AWS Console.

### The workflow
1. user types question, with the option to upload an excel file. 
2. the LLM will determine what category of tools to call (in a langgraph paradigm). For instance, if the question is about giving advice, then the search category would be returned but if the question is about calculations or retrieving numbers, then the calculation category would be returned. You decide the number of nodes and edges that accomplish this efficiently. 
3. given the answer of the above step, use the tool_calling_llm with the guidance of prompts. If the user prompt is about retrieval/calculation/comparisons in the excel file, the llm will follow the CLASSIFIER_PROMPT which would return a series of steps in json file for python functions to carry out and return a numerial answer or a list of numbers (depending on the kind of question asked). If the user query is about an inquiry related to finance but does not require anything from the excel file, the llm will simply answer, albeit with the RAG and Chat History implementation. 
4. The above should be then returned to the LLM for double-checking the answer and displayed to the interface.
