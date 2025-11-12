FROM gpu-ai-base
WORKDIR /workspace
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /workspace
