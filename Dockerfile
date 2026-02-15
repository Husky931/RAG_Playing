# 1. Use a stable Python base (3.10 or 3.11 is safer for AI libs than 3.12 right now)
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system dependencies needed for ChromaDB and Torch
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Install the EXACT versions that work together
RUN pip install --no-cache-dir \
    "numpy<2" \
    "langchain>=0.3" \
    "langchain-text-splitters" \
    "chromadb" \
    "sentence-transformers" \
    "torch"


# 6. Copy your script and data into the container
COPY script.py .
COPY Tesla2.txt .

# 7. Run your script
CMD ["python", "script.py"]