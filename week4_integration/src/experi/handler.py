import json
import boto3
from financial_advisor.week4_integration.src.rag_pipeline.rag_pipeline import full_rag_chain  # From layer

"""Update the handler to load and save state per user/session"""
def lambda_handler(event, context):
    try:
        body = json.loads(event['body']) if 'body' in event else event

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

        # Load conversation state from DynamoDB
        conversation_state = load_conversation_state(user_id)

        # Inject the loaded state into your pipeline (if needed)
        # If your pipeline uses a global conversation_state, you may need to refactor
        # For example, pass conversation_state as an argument or set it globally

        response = full_rag_chain.invoke({
            "question": body["question"],
            "chat_history": body.get("chat_history", [])
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
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }