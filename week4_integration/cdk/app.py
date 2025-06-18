# CDK entrypoint—you never upload this to S3 or Lambda. 
# Running cdk deploy reads it and spins up the resources.
import os
import aws_cdk as cdk
from langgraph_stack import LangGraphStepFunctionsStack

# Use environment vars or defaults
env = cdk.Environment(
    account=os.getenv("CDK_DEFAULT_ACCOUNT"),
    region=os.getenv("CDK_DEFAULT_REGION"),
)

app = cdk.App()
LangGraphStepFunctionsStack(
    app,
    "LangGraphStepFunctionsStack",
    env=env
)
app.synth()

# When you run cdk deploy, CDK reads this file, instantiates your stack, and 
# synthesizes/deploys the CloudFormation.