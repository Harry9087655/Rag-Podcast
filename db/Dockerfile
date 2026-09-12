# Postgres 16 + pgvector, with pg_jieba (Chinese full-text search) compiled in.
#
# Builds on the official pgvector image (Debian bookworm), which already carries
# pg_config on PATH and the PGDG apt repo, so postgresql-server-dev-16 is
# installable. pg_jieba is a plain extension — no shared_preload_libraries
# needed for basic tokenization (custom dictionaries are not used here).
#
# Build deps are installed and purged in ONE RUN layer so the purge actually
# shrinks the final image (split layers would keep the toolchain around).
FROM pgvector/pgvector:pg16

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake git libpq-dev postgresql-server-dev-16; \
    git clone --depth 1 https://github.com/jaiminpan/pg_jieba.git /tmp/pg_jieba; \
    cd /tmp/pg_jieba; \
    git submodule update --init --recursive; \
    mkdir -p build; \
    cd build; \
    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DPostgreSQL_TYPE_INCLUDE_DIR="$(pg_config --includedir-server)" \
        -DPostgreSQL_LIBRARY_DIR="$(pg_config --libdir)" \
        -DPostgreSQL_INCLUDE_DIR="$(pg_config --includedir)" \
        ..; \
    make; \
    make install; \
    cd /; \
    rm -rf /tmp/pg_jieba; \
    apt-get purge -y --auto-remove \
        build-essential cmake git libpq-dev postgresql-server-dev-16; \
    rm -rf /var/lib/apt/lists/*
