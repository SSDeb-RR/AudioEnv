from __future__ import annotations

from pathlib import Path

import pytest
import torch

from audioenv.training.gradient_audit import install_gradient_audit, validate_gradient_evidence


class _AllowedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj_lora_A = torch.nn.Parameter(torch.tensor([1.0]))
        self.frozen = torch.nn.Parameter(torch.tensor([2.0]), requires_grad=False)


class _Engine:
    def __init__(self, module):
        self.module = module
        self.optimizer = torch.optim.SGD((p for p in module.parameters() if p.requires_grad), lr=0.1)

    def optimizer_step(self):
        self.optimizer.step()
        return 1.0


def test_gradient_audit_proves_lora_only_parameter_update(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AUDIOENV_GRADIENT_EVIDENCE_DIR", str(tmp_path))
    install_gradient_audit(_Engine)
    engine = _Engine(_AllowedModule())
    (engine.module.q_proj_lora_A.square().sum()).backward()
    engine.optimizer_step()
    summary = validate_gradient_evidence(tmp_path, expected_ranks=1)
    assert summary["all_lora_only"] and summary["all_steps_changed"]
    assert summary["ranks"] == (0,)


def test_gradient_audit_fails_on_undeclared_trainable_parameter(tmp_path: Path, monkeypatch):
    class UnsafeModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.talker_weight = torch.nn.Parameter(torch.tensor([1.0]))

    class UnsafeEngine(_Engine):
        pass

    monkeypatch.setenv("AUDIOENV_GRADIENT_EVIDENCE_DIR", str(tmp_path))
    install_gradient_audit(UnsafeEngine)
    engine = UnsafeEngine(UnsafeModule())
    engine.module.talker_weight.square().sum().backward()
    with pytest.raises(RuntimeError, match="outside declared Thinker LoRA"):
        engine.optimizer_step()
    with pytest.raises(ValueError, match="outside Thinker LoRA"):
        validate_gradient_evidence(tmp_path)
