#base image
FROM python:3.11-slim

WORKDIR /app

# install dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt /app/requirements.txt

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

#Copy directories
COPY config/ ./config/
#   /app/config/common_settings.py

COPY yt_neo4j_etl/src/ ./yt_neo4j_etl/src/ 
#    /app/yt_neo4j_etl/src/load_to_neo4j.py

# run
CMD ["python", "-m", "yt_neo4j_etl.src.load_to_neo4j"]
