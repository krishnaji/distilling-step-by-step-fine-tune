FROM europe-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-12.py310:latest

WORKDIR /

LABEL com.nvidia.volumes.needed=nvidia_driver

ENV PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64

COPY configs /configs

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY dataset.jsonl dataset.jsonl
COPY trainer.py trainer.py

ENTRYPOINT ["deepspeed", "--num_gpus=8", "trainer.py"]

