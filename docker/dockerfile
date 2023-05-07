ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:22.07-py3
FROM $BASE_IMAGE

ENV TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"

COPY . .

RUN echo "GO BRRR!" \
    && conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath \
    && pip install "git+https://github.com/facebookresearch/pytorch3d.git" \
    && apt-get update \
    && apt-get install -y libasound2-dev portaudio19-dev \
    && pip install -r requirements.txt \
    && pip install tensorflow==2.12.0 "opencv-python-headless<4.3" protobuf==3.20.3 \
    && conda install -y ffmpeg \
    && bash install_ext.sh
