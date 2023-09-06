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

# Install Python and pip as root user
RUN apt-get update && apt-get install -y python3 python3-pip

# Install FastAPI and Uvicorn as root user
RUN pip3 install fastapi uvicorn

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

# Copy llama.cpp and model from the builder as root user
COPY --from=builder /llama.cpp /llama.cpp

# Switch to non-root user
USER appuser

# Copy FastAPI app
COPY ./app /app

WORKDIR /llama.cpp

# Run Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "2023"]
