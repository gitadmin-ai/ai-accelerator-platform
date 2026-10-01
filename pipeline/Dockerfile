# GPU node only. Build from the repo root: docker build -f pipeline/Dockerfile .
# (needs shared/ in the build context alongside pipeline/)
#
# The ddl_client native extension (--checkpoint-storage ddl) comes from a
# prebuilt NebulaR artifact image supplied as the `ddlwheel` named build
# context -- /wheels/*.whl plus the UCX it was built against at /opt/ucx. Build
# that image once from the NebulaR repo (see its client/python/Dockerfile.wheel):
#   docker build -f client/python/Dockerfile.wheel -t nebula-ddl-client:0.1.0 .
# then build this one with:
#   docker build -f pipeline/Dockerfile \
#       --build-context ddlwheel=docker-image://nebula-ddl-client:0.1.0 .
# (docker-compose.yml wires this up via additional_contexts.)
#
# `-base-` (not `-runtime-`): torch's own pip wheel bundles its own copy of
# the CUDA math libraries it needs (see the explicit --index-url install
# below), so apt-installing a second, full CUDA runtime here would just
# duplicate ~2GB nothing ends up using. `-base-` still carries the same
# nvidia_entrypoint.sh driver-detection/compatibility warning at startup.
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# libnuma1/libibverbs1/librdmacm1: runtime deps of the source-built UCX
# copied in from the ddlwheel context below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 curl ca-certificates libnuma1 libibverbs1 librdmacm1 \
    && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3.11 /usr/bin/python
# ensurepip is disabled for Debian/Ubuntu's apt-installed Python, and
# apt's python3-pip binds to whatever /usr/bin/python3 defaults to (3.10
# here), not python3.11 -- get-pip.py installs pip for python3.11 directly.
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && python3.11 /tmp/get-pip.py \
    && rm /tmp/get-pip.py

WORKDIR /srv

COPY shared/nebula-events ./shared/nebula-events
COPY shared/nebula-ddl-storage ./shared/nebula-ddl-storage
COPY pipeline ./pipeline

# Installed from PyTorch's own CUDA 12.4 wheel index, matching the target
# GPU node's toolkit exactly (`nvcc --version` -> release 12.4, V12.4.131)
# -- must stay in sync with the torch==2.6.0 pin in pipeline/pyproject.toml.
# Installing this first means the next line's `torch==2.6.0` requirement is
# already satisfied and pip won't reach PyPI's default index for it, which
# only carries a much newer CUDA-13-bundled build.
RUN python3.11 -m pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Native ddl_client: UCX first (the wheel's extension links against it), then
# the wheel. The import below fails the image build if the wheel, UCX and this
# base image's Python/glibc don't actually fit together.
COPY --from=ddlwheel /opt/ucx /opt/ucx
COPY --from=ddlwheel /wheels /tmp/wheels
RUN echo /opt/ucx/lib > /etc/ld.so.conf.d/ucx.conf && ldconfig \
    && python3.11 -m pip install --no-cache-dir /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels \
    && python3.11 -c "import ddl_client; print('ddl_client', ddl_client.__version__)"

RUN python3.11 -m pip install --no-cache-dir ./shared/nebula-events ./shared/nebula-ddl-storage ./pipeline

EXPOSE 8001

# Runs pipeline/service/app.py -- the HTTP trigger backend.job_manager's
# HttpTrainingClient calls, which in turn subprocess.Popen()s
# pipeline.train_lora_with_gpu_stats for each job, same as local dev does.
CMD ["uvicorn", "pipeline.service.app:app", "--host", "0.0.0.0", "--port", "8001"]
