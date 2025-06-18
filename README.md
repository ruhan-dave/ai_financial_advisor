# ai_financial_advisor
This is an AI agent designed to answer a range of user questions related to personal financial management and basic financial analysis tasks. It is not meant to provide professional advisor assistance or investment recommendations, as it lacks knowledge of advanced topics. However, it is capable of generating practical answers.

## The workflow
1. user types question, with the option to upload an excel file. 
2. the LLM will determine what category of tools to call (in a langgraph paradigm). For instance, if the question is about giving advice, then the search category would be returned but if the question is about calculations or retrieving numbers, then the calculation category would be returned. You decide the number of nodes and edges that accomplish this efficiently. 
3. given the answer of the above step, use the tool_calling_llm with the guidance of prompts. If the user prompt is about retrieval/calculation/comparisons in the excel file, the llm will follow the CLASSIFIER_PROMPT which would return a series of steps in json file for python functions to carry out and return a numerial answer or a list of numbers (depending on the kind of question asked). If the user query is about an inquiry related to finance but does not require anything from the excel file, the llm will simply answer, albeit with the RAG and Chat History implementation. 
4. The above should be then returned to the LLM for double-checking the answer and displayed to the interface.
