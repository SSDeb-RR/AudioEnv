from __future__ import annotations

from types import SimpleNamespace

import torch

from audioenv.models.qwen3_omni import Qwen3OmniAgentAdapter
from audioenv.spec import AudioObservation, SpeakAction


class _Inputs(dict):
    def to(self, *_args, **_kwargs):
        return self


class _Processor:
    def apply_chat_template(self, *_args, **_kwargs):
        return "prompt"

    def __call__(self, **_kwargs):
        return _Inputs(input_ids=torch.tensor([[10, 11]]))

    def batch_decode(self, *_args, **_kwargs):
        return ["hello"]


class _Model:
    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self, *, diverge: bool = False):
        self.calls = []
        self.diverge = diverge

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        ids = torch.tensor([[10, 11, 20, 21]])
        if kwargs["return_audio"] is False:
            return SimpleNamespace(sequences=ids, scores=(object(), object()))
        if self.diverge:
            ids = torch.tensor([[10, 11, 20, 22]])
        return ids, torch.ones((1, 1, 240))

    def compute_transition_scores(self, sequences, scores, normalize_logits):
        assert sequences.tolist() == [[10, 11, 20, 21]]
        assert len(scores) == 2
        assert normalize_logits
        return torch.tensor([[-0.25, -0.5]])


def _adapter(*, diverge: bool = False) -> tuple[Qwen3OmniAgentAdapter, _Model]:
    adapter = Qwen3OmniAgentAdapter("unused")
    model = _Model(diverge=diverge)
    adapter._model = model
    adapter._processor = _Processor()
    return adapter, model


def test_qwen_adapter_scores_thinker_then_renders_matching_audio(monkeypatch):
    monkeypatch.setattr("qwen_omni_utils.process_mm_info", lambda *_args, **_kwargs: ([], [], []))
    adapter, model = _adapter()

    action, record = adapter.generate_step(AudioObservation())

    assert isinstance(action, SpeakAction)
    assert action.audio.startswith(b"RIFF")
    assert record.generated_ids == (20, 21)
    assert record.old_logprobs == (-0.25, -0.5)
    assert [call["return_audio"] for call in model.calls] == [False, True]
    assert model.calls[0]["return_dict_in_generate"] is True
    assert model.calls[0]["output_scores"] is True


def test_qwen_adapter_rejects_scored_and_rendered_thinker_divergence(monkeypatch):
    monkeypatch.setattr("qwen_omni_utils.process_mm_info", lambda *_args, **_kwargs: ([], [], []))
    adapter, _model = _adapter(diverge=True)

    try:
        adapter.generate_step(AudioObservation())
    except RuntimeError as exc:
        assert "changed between scored and rendered" in str(exc)
    else:  # pragma: no cover - assertion aid
        raise AssertionError("divergent generations must fail closed")
