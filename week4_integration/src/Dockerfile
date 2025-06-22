# Use an AWS Lambda Python base image for your target runtime
FROM public.ecr.aws/lambda/python:3.12

# Install zip utility (optional, if zipping inside container)
RUN dnf install -y zip

# Set the working directory inside the container
WORKDIR /var/task

# Copy only necessary files from rag_pipeline
COPY rag_pipeline/requirements.txt .
COPY rag_pipeline/handler.py .
COPY rag_pipeline/rag_pipeline.py .

# Install dependencies directly into /var/task
RUN pip install -r requirements.txt -t .

# (Optional) Zip the contents if you want to do it inside the container
# RUN zip -r /var/task/lambda_package.zip .