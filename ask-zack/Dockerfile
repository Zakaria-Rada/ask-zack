# The builder stage
FROM ubuntu:22.04 AS builder

# Arguments
ARG MODEL=llama-2-13b-chat.ggmlv3.q4_0.bin

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y build-essential git wget

RUN git clone https://github.com/ggerganov/llama.cpp.git && cd llama.cpp && make

WORKDIR /llama.cpp

RUN wget "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve/main/${MODEL}"

# The final stage
FROM ubuntu:22.04 AS final

# Arguments
ARG MODEL
ARG UID=10001

# Add non-root user
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/home/appuser" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

# Correcting Permissions
USER root
RUN chown -R appuser:appuser /home/appuser && \
    chmod 755 /home/appuser

# Switch to non-root user
USER appuser

# Copy from builder
COPY --from=builder --chown=appuser:appuser /llama.cpp /home/appuser/llama.cpp

# Install Python and other dependencies
USER root
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv

# Create a virtual environment
RUN python3 -m venv /home/appuser/venv
ENV PATH="/home/appuser/venv/bin:$PATH"

# Install FastAPI and Uvicorn
RUN pip install fastapi uvicorn python-multipart flask-cors

# Switch back to non-root user
USER appuser

# Copy your FastAPI app into the container
COPY --chown=appuser:appuser ./app /home/appuser/app

# Set working directory
WORKDIR /home/appuser/app

# Command to run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "2023"]
