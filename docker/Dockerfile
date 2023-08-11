FROM continuumio/miniconda3


ENV JAVA_HOME="/usr"
ENV PATH=$PATH:/opt/google-cloud-sdk/bin:/opt/conda/envs/use/bin
ENV DEBIAN_FRONTEND=noninteractive
ENV PYSPARK_PYTHON="/opt/conda/envs/use/bin/python"
ENV PYSPARK_DRIVER_PYTHON="/opt/conda/envs/use/bin/python"
ENV DEBIAN_FRONTEND=noninteractive

## Add foo user
RUN apt update && apt install \
    curl \
    sudo \
    rsync \
    git \
    nano \
    wget \
    telnet gcc g++ openjdk-11-jdk -y

# Reinstall certificates
RUN apt reinstall ca-certificates -y && update-ca-certificates

# Foo
RUN adduser --disabled-password --gecos '' foo && \
    adduser foo sudo&& \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    sudo chown -R foo:foo /home/foo

RUN echo "jupyter server list" >> /home/foo/.bashrc

# Create the virtual environment
RUN conda create -n use python=3.9.13 -y
RUN echo ". /opt/conda/etc/profile.d/conda.sh && conda activate use" >> /home/foo/.bashrc
SHELL ["conda", "run", "-n", "use", "/bin/bash", "-c"]
RUN conda install nb_conda -y

# For ressources monitoring
RUN pip install glances

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Move back to the workspace
WORKDIR /workspace

USER foo
WORKDIR /workspace

ENV USE_URL="https://tfhub.dev/google/universal-sentence-encoder/3"
ENV NLLB_ID="facebook/nllb-200-distilled-600M"
ENV DEVICE="cpu"

# Download the models
RUN python -c 'from transformers import AutoModelForSeq2SeqLM, AutoTokenizer; AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="ja", tgt_lang="en")'
RUN python -c 'from transformers import AutoModelForSeq2SeqLM, AutoTokenizer; AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")'
RUN python -c 'import tensorflow_hub as hub; module_url = "https://tfhub.dev/google/universal-sentence-encoder/3";hub.load(module_url)'

COPY server /workspace/server
COPY docker/static/entrypoint /entrypoint
ENTRYPOINT [ "/entrypoint" ]