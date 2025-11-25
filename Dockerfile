FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime
WORKDIR /app


# Install Python deps first for better caching
COPY pyproject.toml README.md MANIFEST.in ./ 
COPY sam3 ./sam3
COPY assets/bpe_simple_vocab_16e6.txt.gz ./assets/

RUN pip install -e . && \
    pip install flask pillow einops decord pydicom apscheduler pycocotools psutil

# Copy the rest of the app (server, scripts, assets)
COPY . /app
EXPOSE 5000

CMD ["python", "server.py"]
