# Use the builder stage to build the llama.cpp executable
FROM ubuntu:22.04 AS builder
ARG MODEL=llama-2-13b-chat.ggmlv3.q4_0.bin

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y build-essential git wget

RUN git clone https://github.com/ggerganov/llama.cpp.git && cd llama.cpp && make

WORKDIR /llama.cpp
RUN wget "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve/main/${MODEL}"

# Final image
FROM ubuntu:22.04 AS final
ARG MODEL

# Install Python, pip, and python3-venv
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv

# Create a virtual environment and activate it
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install FastAPI and Uvicorn
RUN pip install fastapi uvicorn

# Create a user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Switch to non-root user
USER appuser

# Copy llama.cpp and model from the builder
COPY --from=builder /llama.cpp /llama.cpp

# Copy FastAPI app
COPY ./app /app

WORKDIR /llama.cpp

# Run Uvicorn on port 2023
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "2023"]
