# Defines your AWS infrastructure in Python (S3 bucket, 4 Lambdas, 
# Step Functions state machine with retry & catch).

from aws_cdk import (
    Stack,
    Duration,
    aws_s3 as s3,
    aws_logs as logs,
    aws_lambda as _lambda,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as tasks,
)
from aws_cdk.aws_lambda_python_alpha import PythonFunction, Tracing
from constructs import Construct
import os

class LangGraphStepFunctionsStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        # 1) S3 bucket for uploads
        bucket = s3.Bucket(self, "UploadsBucket")

        # 2) Define each of your four Lambdas

        classify_fn = PythonFunction(
            self, "ClassifyTaskFn",
            entry="src/classify",
            index="handler.py",
            handler="lambda_handler",
            runtime=_lambda.Runtime.PYTHON_3_9,
            tracing=Tracing.ACTIVE,
            environment={
                "COHERE_KEY": os.getenv("COHERE_KEY", "")
            }
        )

        excel_fn = PythonFunction(
            self, "ExcelPipelineFn",
            entry="src/excel_pipeline",
            index="handler.py",
            handler="lambda_handler",
            runtime=_lambda.Runtime.PYTHON_3_9,
            tracing=Tracing.ACTIVE,
            environment={
                "BUCKET_NAME": bucket.bucket_name,
                "COHERE_KEY": os.getenv("COHERE_KEY", "")
            }
        )
        bucket.grant_read(excel_fn)

        rag_fn = PythonFunction(
            self, "RAGTaskFn",
            entry="src/rag_task",
            index="handler.py",
            handler="lambda_handler",
            runtime=_lambda.Runtime.PYTHON_3_9,
            tracing=Tracing.ACTIVE
        )

        refine_fn = PythonFunction(
            self, "RefineTaskFn",
            entry="src/refine",
            index="handler.py",
            handler="lambda_handler",
            runtime=_lambda.Runtime.PYTHON_3_9,
            tracing=Tracing.ACTIVE,
            environment={
                "COHERE_KEY": os.getenv("COHERE_KEY", "")
            }
        )

        # 3) Create Step Functions tasks with retry & catch

        classify_task = tasks.LambdaInvoke(
            self, "ClassifyTask",
            lambda_function=classify_fn,
            payload=sfn.TaskInput.from_object({"query.$": "$.query"}),
            output_path="$.Payload"
        )
        classify_task.add_retry(
            errors=["States.ALL"],
            interval=Duration.seconds(2),
            max_attempts=3,
            backoff_rate=2.0
        )
        classify_task.add_catch(
            handler=sfn.Fail(self, "ClassifyFailed",
                             error="ClassifyError",
                             cause="Failed to classify query"),
            errors=["States.ALL"]
        )

        excel_task = tasks.LambdaInvoke(
            self, "ExcelPipelineTask",
            lambda_function=excel_fn,
            payload=sfn.TaskInput.from_object({
                "query.$":     "$.query",
                "s3_key.$":    "$.df_pointer.s3_key"
            }),
            output_path="$.Payload"
        )
        excel_task.add_retry(
            errors=["States.ALL"],
            interval=Duration.seconds(5),
            max_attempts=2,
            backoff_rate=1.5
        )
        excel_task.add_catch(
            handler=sfn.Fail(self, "ExcelFailed",
                             error="ExcelError",
                             cause="Excel pipeline failed"),
            errors=["States.ALL"]
        )

        rag_task = tasks.LambdaInvoke(
            self, "RAGTask",
            lambda_function=rag_fn,
            payload=sfn.TaskInput.from_object({"query.$": "$.query"}),
            output_path="$.Payload"
        )
        rag_task.add_retry(
            errors=["States.ALL"],
            interval=Duration.seconds(3),
            max_attempts=2,
            backoff_rate=2.0
        )
        rag_task.add_catch(
            handler=sfn.Fail(self, "RAGFailed",
                             error="RAGError",
                             cause="RAG pipeline failed"),
            errors=["States.ALL"]
        )

        refine_task = tasks.LambdaInvoke(
            self, "RefineTask",
            lambda_function=refine_fn,
            payload=sfn.TaskInput.from_object({
                "tool.$":        "$.tool",
                "excelAns.$":    "$.excel_answer",
                "ragAns.$":      "$.rag_answer",
                "query.$":       "$.query"
            }),
            output_path="$.Payload"
        )
        refine_task.add_retry(
            errors=["States.ALL"],
            interval=Duration.seconds(2),
            max_attempts=2,
            backoff_rate=1.5
        )
        refine_task.add_catch(
            handler=sfn.Fail(self, "RefineFailed",
                             error="RefineError",
                             cause="Failed to merge/refine answer"),
            errors=["States.ALL"]
        )

        # 4) Choice: route based on $.tool
        choice = sfn.Choice(self, "Excel or Search?")
        is_excel = sfn.Condition.string_equals("$.tool", "excel")

        # 5) Wire steps together
        definition = (
            classify_task
            .next(choice.when(is_excel, excel_task).otherwise(rag_task))
            .next(refine_task)
            .next(sfn.Succeed(self, "Done"))
        )

        # 6) Log Group for Step Functions
        log_group = logs.LogGroup(
            self, "LangGraphLogGroup",
            retention=logs.RetentionDays.ONE_WEEK
        )

        # 7) State Machine
        sfn.StateMachine(
            self, "LangGraphStateMachine",
            definition=definition,
            logs=sfn.LogOptions(
                destination=log_group,
                level=sfn.LogLevel.ALL
            ),
            tracing_enabled=True  # X-Ray end-to-end
        )