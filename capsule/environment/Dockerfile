# hash:sha256:b0bf888c26fd5b8eeb36514a55449be0dedbecb26d1b2883092e720b90cf0f0f
FROM registry.codeocean.com/codeocean/lightning-jupyterlab-py39:1.0

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install -U \
    hydra-core==1.3.2 \
    torch==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html \
    torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html \
    torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html \
    torch-sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html \
    torch-spline-conv==1.2.2 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html \
    torch_geometric==2.4.0 \
    lightning==2.1.4 \
    torchaudio==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html \
    torchmetrics==1.3.0.post0 \
    torchvision==0.16.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html \
    matminer==0.9.0 \
    pymatgen==2023.8.10 \
    SMACT==2.5.5 \
    fastdtw==0.3.4 \
    tqdm==4.66.1 \
    p_tqdm==1.4.0 \
    chemparse==0.3.1

COPY postInstall /
RUN /postInstall
