# Build from the repo root: docker build -f backend/Dockerfile .
# (needs shared/ in the build context alongside backend/)
FROM python:3.11-slim

WORKDIR /srv

# Third-party dependencies first, keyed only on backend/pyproject.toml, so editing
# code doesn't re-download them (same reasoning as pipeline/Dockerfile).
COPY backend/pyproject.toml /tmp/backend-pyproject.toml
RUN --mount=type=cache,target=/root/.cache/pip \
    python3.11 -c "import tomllib; print('\\n'.join(tomllib.load(open('/tmp/backend-pyproject.toml', 'rb'))['project']['dependencies']))" > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt

COPY shared/nebula-events ./shared/nebula-events
COPY shared/nebula-ddl-storage ./shared/nebula-ddl-storage
COPY backend ./backend

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps ./shared/nebula-events ./shared/nebula-ddl-storage ./backend

# TODO(nebula storage): the above gives the backend the nebula-ddl-storage
# *package*, but not the native ddl_client extension it optionally imports
# (see shared/nebula-ddl-storage/nebula_ddl_storage/connection.py's build
# hint) -- Settings storage backend "local" works out of the box; "ddl"
# additionally needs ddl_client built from the NebulaR repo's client/
# directory and placed on PYTHONPATH in this image. Add that build stage
# here once you're ready to exercise DDL storage from this container.

ENV NEBULA_TRAINING_BACKEND=http
EXPOSE 8000

CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
