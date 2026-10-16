ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG AUDIOENV_WHEEL
ARG VERL_OMNI_COMMIT=49dedb6ec0691588494b629fc1b610a86612966b
ARG VERL_COMMIT=8a694930275061f52ebd538c906ef8819af56dbd
ARG VLLM_OMNI_COMMIT=fe478a95ab39e80085da06e3401c3ed471898a57

RUN test -n "$AUDIOENV_WHEEL" && test -n "$VERL_OMNI_COMMIT" \
    && test -n "$VERL_COMMIT" && test -n "$VLLM_OMNI_COMMIT"
COPY ${AUDIOENV_WHEEL} /tmp/audioenv.whl
COPY docker/gpu-worker/dependencies.lock.json /opt/audioenv/dependencies.lock.json
RUN python -m pip install --no-cache-dir \
      "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@${VLLM_OMNI_COMMIT}" \
    && python -m pip install --no-cache-dir \
      "verl @ git+https://github.com/verl-project/verl.git@${VERL_COMMIT}" \
    && python -m pip install --no-cache-dir \
      "verl-omni[audio] @ git+https://github.com/verl-project/verl-omni.git@${VERL_OMNI_COMMIT}" \
    && python -m pip install --no-cache-dir /tmp/audioenv.whl

ENV AUDIOENV_DEPENDENCY_LOCK=/opt/audioenv/dependencies.lock.json

RUN useradd --create-home --uid 10001 audioenv \
    && mkdir -p /workspace \
    && chown -R audioenv:audioenv /workspace
USER audioenv
WORKDIR /workspace
ENTRYPOINT ["audioenv-gpu-worker"]
