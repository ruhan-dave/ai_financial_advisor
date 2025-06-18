import boto3
import json


client = boto3.client('bedrock')

response = client.get_foundation_model(
    modelIdentifier='meta.llama4-maverick-17b-instruct-v1:0'
)

# Initialize Bedrock runtime client
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1' # Or your chosen region
)

# Initial system prompt (if desired and supported for system role by the model with converse API)
# For Llama 4 with converse, the system prompt should be passed via the `system` parameter of the converse() call.
# So, initialize conversation_history without the system message.
conversation_history = []

system_prompt_text = "You are a financial advisor who helps users with their financial goals with actionable advice in a clear communication style with minimal irrelevant answers."

print(f"AI: Hello! I am your financial advisor. How can I help you today?") # Initial greeting

while True:
    user_input = input("Me: ")
    if user_input.lower() == 'quit':
        print("AI: Goodbye! Let me know if you need financial advice in the future.")
        break

    # Append user input to conversation history - simplified content structure
    conversation_history.append({"role": "user", "content": [{"text": user_input}]})

    try:
        # Send the conversation history to the model
        api_response = bedrock_runtime.converse(
            modelId="meta.llama3-70b-instruct-v1:0", # Using a Llama 3 model as an example, ensure your modelId is correct and accessible
            messages=conversation_history,
            system=[{"text": system_prompt_text}] # Pass system prompt here
        )

        # Extract the response text
        # The structure of the response from the converse API might vary slightly.
        # It's common for the main assistant message to be in output.message.content
        response_text = ""
        if api_response.get("output") and api_response["output"].get("message") and api_response["output"]["message"].get("content"):
            content_list = api_response["output"]["message"]["content"]
            if content_list and isinstance(content_list, list) and content_list[0].get("text"):
                response_text = content_list[0]["text"]

        # Print the current response
        print(f"AI: {response_text}")

        # Append response to conversation history - simplified content structure
        if response_text: # Only append if there's a valid response
            conversation_history.append(
                {"role": "assistant", "content": [{"text": response_text}]}
            )

        # Manage conversation history length
        # Ensure system prompt is not accidentally removed if you re-insert it
        # A common strategy is to keep system prompt separate and only manage user/assistant turns
        if len(conversation_history) > 12:  # Keep the last 6 user/assistant exchanges
            conversation_history = conversation_history[-12:]

    except Exception as e:
        print(f"Error communicating with Bedrock: {e}")
        # Optionally, break the loop or implement more sophisticated error handling
        break
