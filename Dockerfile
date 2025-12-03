# 1. Start with a lightweight Python base
FROM python:3.9-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy requirements first (to cache dependencies)
# We need to create a requirements.txt first!
COPY requirements.txt .

# 4. Install dependencies
# We install build-essential for XGBoost compilation
RUN apt-get update && apt-get install -y build-essential
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the code
COPY . .

# 6. Expose the port
EXPOSE 8000

# 7. Run the command to start the API
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]