ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG AUDIOENV_WHEEL
ARG VERL_OMNI_COMMIT=49dedb6ec0691588494b629fc1b610a86612966b
ARG VERL_COMMIT=8a694930275061f52ebd538c906ef8819af56dbd
ARG VLLM_OMNI_COMMIT=fe478a95ab39e80085da06e3401c3ed471898a57
ARG TAU2_COMMIT=fc0055dc4e0a316c3f83133267fbd6faaa770992

RUN test -n "$AUDIOENV_WHEEL" && test -n "$VERL_OMNI_COMMIT" \
    && test -n "$VERL_COMMIT" && test -n "$VLLM_OMNI_COMMIT" && test -n "$TAU2_COMMIT"
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
      ca-certificates \
      git \
    && rm -rf /var/lib/apt/lists/* \
    && git --version
COPY ${AUDIOENV_WHEEL} /tmp/audioenv.whl
COPY docker/gpu-worker/dependencies.lock.json /opt/audioenv/dependencies.lock.json
RUN mkdir -p /opt/audioenv/checkouts/verl-omni /opt/audioenv/checkouts/tau2 \
    && git -C /opt/audioenv/checkouts/verl-omni init \
    && git -C /opt/audioenv/checkouts/verl-omni remote add origin https://github.com/verl-project/verl-omni.git \
    && git -C /opt/audioenv/checkouts/verl-omni fetch --depth 1 origin "$VERL_OMNI_COMMIT" \
    && git -C /opt/audioenv/checkouts/verl-omni checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/audioenv/checkouts/verl-omni rev-parse HEAD)" = "$VERL_OMNI_COMMIT" \
    && git -C /opt/audioenv/checkouts/tau2 init \
    && git -C /opt/audioenv/checkouts/tau2 remote add origin https://github.com/sierra-research/tau2-bench.git \
    && git -C /opt/audioenv/checkouts/tau2 fetch --depth 1 origin "$TAU2_COMMIT" \
    && git -C /opt/audioenv/checkouts/tau2 checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/audioenv/checkouts/tau2 rev-parse HEAD)" = "$TAU2_COMMIT"
RUN python -m pip install --no-cache-dir \
      "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@${VLLM_OMNI_COMMIT}" \
    && python -m pip install --no-cache-dir \
      "verl @ git+https://github.com/verl-project/verl.git@${VERL_COMMIT}" \
    && python -m pip install --no-cache-dir \
      "verl-omni[audio] @ git+https://github.com/verl-project/verl-omni.git@${VERL_OMNI_COMMIT}" \
      "tau2 @ git+https://github.com/sierra-research/tau2-bench.git@${TAU2_COMMIT}" \
    && python -m pip install --no-cache-dir /tmp/audioenv.whl \
    && python -m pip freeze --all | LC_ALL=C sort > /opt/audioenv/python-environment.lock \
    && test -s /opt/audioenv/python-environment.lock

# Fail the immutable image build if either pinned upstream changes an API that
# AudioEnv's agent-loop, multimodal output, or protected reward path relies on.
RUN audioenv-verl-contract-check

ENV AUDIOENV_DEPENDENCY_LOCK=/opt/audioenv/dependencies.lock.json
ENV AUDIOENV_PYTHON_ENVIRONMENT_LOCK=/opt/audioenv/python-environment.lock

RUN useradd --create-home --uid 10001 audioenv \
    && mkdir -p /workspace \
    && chown -R audioenv:audioenv /workspace
USER audioenv
WORKDIR /workspace
# Providers always supply the full worker command. The idle default keeps a
# freshly provisioned SSH-managed Pod alive without accidentally starting an
# unconfigured or duplicate worker process.
CMD ["sleep", "infinity"]
