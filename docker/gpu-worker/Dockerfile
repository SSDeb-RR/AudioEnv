ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG AUDIOENV_WHEEL
ARG VERL_OMNI_COMMIT=49dedb6ec0691588494b629fc1b610a86612966b

RUN test -n "$AUDIOENV_WHEEL" && test -n "$VERL_OMNI_COMMIT"
COPY ${AUDIOENV_WHEEL} /tmp/audioenv.whl
RUN python -m pip install --no-cache-dir /tmp/audioenv.whl \
    && python -m pip install --no-cache-dir \
      "git+https://github.com/verl-project/verl-omni.git@${VERL_OMNI_COMMIT}"

RUN useradd --create-home --uid 10001 audioenv \
    && mkdir -p /workspace \
    && chown -R audioenv:audioenv /workspace
USER audioenv
WORKDIR /workspace
ENTRYPOINT ["audioenv-gpu-worker"]
