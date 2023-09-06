# Stage 1: Builder
ARG MODEL=llama-2-13b-chat.ggmlv3.q4_0.bin
FROM ubuntu:22.04 AS builder
ARG MODEL

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y build-essential git wget

RUN git clone https://github.com/ggerganov/llama.cpp.git && cd llama.cpp && make

WORKDIR /llama.cpp

ENV MODEL=$MODEL
RUN wget "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve/main/${MODEL}"

# Stage 2: Final
FROM ubuntu:22.04 AS final

ARG UID=10001
ARG MODEL

RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv

RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

USER appuser

COPY --from=builder /llama.cpp /llama.cpp

# Create virtual environment and activate it
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install FastAPI and Uvicorn
RUN pip install fastapi uvicorn python-multipart

COPY ./app /app
WORKDIR /app

ENV MODEL=$MODEL

# Expose port
EXPOSE 2023

# Run Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "2023"]
