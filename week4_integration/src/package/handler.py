import json
from rag_pipeline import create_rag_chain, load_conversation_state, save_conversation_state

def lambda_handler(event, context):
    try:
        # Parse the incoming event
        body = json.loads(event['body']) if 'body' in event and isinstance(event['body'], str) else event

        if body.get('next') != 'rag_pipeline':
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Invalid pipeline'})
            }

        user_id = body.get("user_id")
        if not user_id:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Missing user_id'})
            }

        user_question = body.get('question') or body.get('query', '')
        if not user_question:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No question provided'})
            }

        # Load conversation state from DynamoDB
        conversation_state = load_conversation_state(user_id)

        # Create a new RAG chain instance with the loaded state
        rag_chain = create_rag_chain(conversation_state)

        # Run the RAG pipeline
        response = rag_chain.invoke({
            "question": user_question,
            "chat_history": body.get("chat_history", [])
        })

        # Add to conversation history
        conversation_state.history.append({
            "user": user_question,
            "assistant": response
        })

        # Save updated conversation state back to DynamoDB
        save_conversation_state(user_id, conversation_state)

        return {
            'statusCode': 200,
            'body': json.dumps({
                'response': response,
                'pipeline': 'rag_pipeline'
            })
        }
    except Exception as e:
        # Optionally log the error here
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }