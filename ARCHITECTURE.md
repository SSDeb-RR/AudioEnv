# Architecture and trust boundaries

```text
private task truth ──> caller renderer ──audio──> trainable policy
       │                                      │
       │                               speech or tool call
       │                                      │
       └─> verifier <── state diff <── tool executor
                  │
            reward events ──> trace ──> existing trainer
```

## Boundaries

1. The policy receives audio, structured tool results, rewards, and public
   metrics. It never receives caller source text, hidden facts, expected state,
   oracle actions, or verifier code.
2. The caller renderer receives private utterance text. A production renderer
   must return native audio without placing text in the policy observation.
3. Tool handlers receive only current isolated state and structured arguments.
4. The verifier reads evaluator state after each action and at termination.
5. Public traces exclude semantic observability sidecars by default.
6. Trainer metadata contains policy token IDs and log-probabilities, never
   private benchmark truth.

## Deployment separation

For a managed deployment, place environment execution and verifier truth in an
isolated worker. Give the model worker only the serialized agent view. Mount the
private verifier directory only in the environment worker, never in a model or
harness container.

Each parallel episode owns a deep copy of initial state. A deployment backed by
a database must provide the same property using a transaction, snapshot, or
per-episode database namespace.

## Audio profiles

Audio profiles are versioned environment inputs. A profile records language,
speaker, sample rate, codec, speaking rate, accent, prosody, and noise level.
Changing a profile changes the evaluation distribution and must not silently
reuse an earlier result label.

The built-in fixture renderer uses the profile and seed to generate a stable
WAV. It exists only for contract and determinism tests. Native model or recorded
fixture renderers must be validated independently for license, provenance, and
speaker consent.

