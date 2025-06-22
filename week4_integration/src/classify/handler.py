import boto3
import json

def lambda_handler(event, context):
    query = event.get("query", "")
    if not query:
        return {"error": "No query provided"}

    aws_region = "us-east-1"
    model_id = "meta.llama3-8b-instruct-v1:0"

    bedrock = boto3.client('bedrock-runtime', region_name=aws_region)

    prompt = """
    Analyze this query and classify its intent:
    - Use "excel_pipeline" if it requires spreadsheet/data processing
    - Use "rag_pipeline" for general knowledge questions

    Respond ONLY with valid JSON format like:
    {{"next": "excel_pipeline"}} or {{"next": "rag_pipeline"}}

    Query: %s
    """ % json.dumps(query)

    try:
        body = {
            "prompt": prompt,
            "max_gen_len": 512,
            "temperature": 0.1
        }

        response = bedrock.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType="application/json"
        )

        raw_body = response['body'].read().decode()
        print("RAW BEDROCK RESPONSE:", raw_body)  # For debugging

        # Try to parse as JSON
        try:
            result = json.loads(raw_body)
        except Exception:
            return {"error": f"Bedrock returned non-JSON: {raw_body}"}

        # Try to extract the answer
        if "generation" in result:
            try:
                output = json.loads(result["generation"])
                if output.get("next") in ("excel_pipeline", "rag_pipeline"):
                    return output
            except Exception:
                # If not valid JSON, try to extract with regex
                import re
                match = re.search(r'"next"\s*:\s*"(\w+)"', result["generation"])
                if match:
                    return {"next": match.group(1)}
                return {"error": f"Could not parse 'generation': {result['generation']}"}
        else:
            return {"error": f"Unexpected response format: {result}"}

        return {"next": "rag_pipeline"}  # Default fallback

    except Exception as e:
        return {"error": f"Bedrock invocation failed: {str(e)}"}