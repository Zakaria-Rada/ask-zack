# Using ubuntu as a builder
FROM ubuntu:22.04 AS builder
ARG MODEL=llama-2-13b-chat.ggmlv3.q4_0.bin
RUN apt-get update && apt-get upgrade -y && apt-get install -y build-essential git wget
RUN git clone https://github.com/ggerganov/llama.cpp.git && cd llama.cpp && make
WORKDIR /llama.cpp
RUN wget "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve/main/${MODEL}"

# The actual image
FROM ubuntu:22.04 AS final
ARG MODEL

# Install Python, pip, and python3-venv
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv

# Create a virtual environment and activate it
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install FastAPI and Uvicorn
RUN pip install fastapi uvicorn

ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser
USER appuser

# Copy the compiled llama.cpp and the model to the new image
COPY --from=builder /llama.cpp /llama.cpp

# Copy the FastAPI app to the new image
COPY ./app /app

# Set the working directory to /app so that our code can execute
WORKDIR /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "2023"]
