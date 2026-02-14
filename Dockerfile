# Flask chatbot server with Gemini file search
FROM python:3.12-slim


WORKDIR /app

# Install dependencies
COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

# Copy application code
COPY chatbot_server.py .
COPY templates/ templates/

# Store config (from vector_generation.py). Mount over it if you use a different path:
#   -v ./store_config.json:/app/store_config.json
COPY store_config.json ./

EXPOSE 5000

# GEMINI_API_KEY and other env vars should be set at runtime (e.g. docker run -e GEMINI_API_KEY=...)
# or via a .env file mounted into /app
CMD ["python", "chatbot_server.py"]
