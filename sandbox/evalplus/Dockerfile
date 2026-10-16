FROM python@sha256:6e13e65c55e33adf203d77ee371cf8bf5d81bd4902ef07565721f46bf44917af
COPY requirements.txt /tmp/runtime-requirements.txt
RUN python -c 'import sys; assert sys.version_info[:2] == (3, 12)' \
    && python -m pip install --no-cache-dir --only-binary=:all: --no-deps --require-hashes -r /tmp/runtime-requirements.txt
