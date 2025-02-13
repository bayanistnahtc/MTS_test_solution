FROM bitnami/pytorch

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# # Install bitsandbytes with CUDA 11.8 support
# RUN git clone https://github.com/TimDettmers/bitsandbytes.git
# RUN cd bitsandbytes && CUDA_VERSION=118 make cuda11x
# RUN pip install -e bitsandbytes

COPY labeling.py .
COPY data ./data

CMD ["python", "labeling.py"]