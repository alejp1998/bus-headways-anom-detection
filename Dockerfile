FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .
RUN chmod +x run_server.sh

EXPOSE 8050

ENTRYPOINT ["/app/run_server.sh"]
