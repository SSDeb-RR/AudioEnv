"""Transformers inference adapter for Qwen3-Omni.

Model loading is explicit and never downloads weights unless the caller opts
in. Availability is reported by ``probe`` rather than inferred from an ID.
"""

from __future__ import annotations

from hashlib import sha256
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from typing import Any
from uuid import uuid4

from ..contracts import GenerationRecord, ModelCapabilities, ProbeResult
from ..spec import AudioObservation, SpeakAction, ToolCallAction
from .caller import CallerTurn


class Qwen3OmniAgentAdapter:
    model_id = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

    def __init__(
        self,
        checkpoint: str | Path = model_id,
        *,
        revision: str | None = None,
        speaker: str = "Ethan",
        allow_download: bool = False,
        system_prompt: str | None = None,
    ):
        self.checkpoint = str(checkpoint)
        self.revision = revision or "unresolved"
        self.speaker = speaker
        self.allow_download = allow_download
        self.system_prompt = system_prompt or (
            "You are a voice agent. Reply naturally or emit exactly one JSON object "
            'of the form {"tool_call":{"name":...,"arguments":{...}}}.'
        )
        self._model = None
        self._processor = None
        self._history: list[dict[str, Any]] = []

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            audio_input=True,
            audio_output=True,
            structured_tools=True,
            token_logprobs=True,
            trainable=True,
            trainable_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            checkpoint_format="peft-lora",
        )

    def probe(self) -> ProbeResult:
        reasons: list[str] = []
        device = None
        try:
            import torch
            if not torch.cuda.is_available():
                reasons.append("CUDA is unavailable")
            else:
                device = torch.cuda.get_device_name(0)
        except ImportError:
            reasons.append("missing dependency: torch")
        try:
            import transformers
            if not hasattr(transformers, "Qwen3OmniMoeForConditionalGeneration"):
                reasons.append("installed transformers lacks Qwen3-Omni support")
        except ImportError:
            reasons.append("missing dependency: transformers")
        for dependency in ("qwen_omni_utils", "soundfile"):
            try:
                __import__(dependency)
            except ImportError:
                reasons.append(f"missing dependency: {dependency}")
        if not Path(self.checkpoint).exists() and not self.allow_download:
            reasons.append("checkpoint is not local and downloads are disabled")
        return ProbeResult(
            available=not reasons,
            model_id=self.model_id,
            capabilities=self.capabilities,
            device=device,
            revision=None if self.revision == "unresolved" else self.revision,
            reasons=tuple(reasons),
        )

    def load(self) -> None:
        if not Path(self.checkpoint).exists() and not self.allow_download:
            raise RuntimeError("checkpoint download requires allow_download=True")
        try:
            from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
        except ImportError as exc:  # pragma: no cover - requires the qwen extra
            raise RuntimeError("Install AudioEnv with the 'qwen' extra") from exc
        kwargs: dict[str, Any] = {"device_map": "auto", "dtype": "auto"}
        if self.revision != "unresolved":
            kwargs["revision"] = self.revision
        self._model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(self.checkpoint, **kwargs)
        self._processor = Qwen3OmniMoeProcessor.from_pretrained(
            self.checkpoint,
            revision=None if self.revision == "unresolved" else self.revision,
        )
        resolved = getattr(getattr(self._model, "config", None), "_commit_hash", None)
        if resolved:
            self.revision = str(resolved)

    def reset_conversation(self) -> None:
        self._history = [{"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}]

    def generate_step(self, observation: AudioObservation):
        if self._model is None or self._processor is None:
            raise RuntimeError("call load() before generate_step()")
        import soundfile as sf
        from qwen_omni_utils import process_mm_info

        if not self._history:
            self.reset_conversation()
        content: list[dict[str, Any]] = []
        temporary_audio: Path | None = None
        if observation.audio:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as stream:
                stream.write(observation.audio)
                temporary_audio = Path(stream.name)
                content.append({"type": "audio", "audio": stream.name})
        if observation.tool_result is not None:
            content.append({"type": "text", "text": "Tool result: " + json.dumps(observation.tool_result, sort_keys=True)})
        self._history.append({"role": "user", "content": content})
        text = self._processor.apply_chat_template(self._history, add_generation_prompt=True, tokenize=False)
        try:
            audios, images, videos = process_mm_info(self._history, use_audio_in_video=True)
            inputs = self._processor(
                text=text, audio=audios, images=images, videos=videos,
                return_tensors="pt", padding=True, use_audio_in_video=True,
            )
        finally:
            if temporary_audio is not None:
                temporary_audio.unlink(missing_ok=True)
        inputs = inputs.to(self._model.device).to(self._model.dtype)

        # Qwen3-Omni's public ``generate`` API returns the Thinker generation
        # object only when audio rendering is disabled.  When the Talker is
        # enabled it deliberately returns ``(sequences, waveform)`` and drops
        # the Thinker scores.  Run the deterministic Thinker once to retain the
        # policy scores, then render the same sequence through the frozen
        # Talker and fail closed if the two Thinker passes ever diverge.  This
        # is slower than the vLLM-Omni rollout path used for training, but it
        # keeps this direct Transformers adapter truthful and aligned.
        thinker_result = self._model.generate(
            **inputs,
            return_audio=False,
            output_scores=True,
            return_dict_in_generate=True,
            use_audio_in_video=True,
        )
        rendered_ids, output_audio = self._model.generate(
            **inputs,
            speaker=self.speaker,
            return_audio=True,
            use_audio_in_video=True,
        )
        if not hasattr(thinker_result, "sequences") or not hasattr(thinker_result, "scores"):
            raise RuntimeError("Qwen Thinker did not return sequences and token scores")
        sequences = thinker_result.sequences
        if sequences.shape != rendered_ids.shape or not sequences.equal(rendered_ids):
            raise RuntimeError("Qwen Thinker output changed between scored and rendered generation")
        prompt_length = int(inputs["input_ids"].shape[1])
        generated = sequences[0, prompt_length:]
        if not len(generated):
            raise RuntimeError("Qwen Thinker generated no action tokens")
        decoded = self._processor.batch_decode(
            sequences[:, prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        transition_scores = self._model.compute_transition_scores(
            sequences, thinker_result.scores, normalize_logits=True
        )[0]
        logprobs = tuple(float(value) for value in transition_scores[-len(generated) :].detach().cpu())
        audio_bytes = b""
        audio_id = None
        if output_audio is not None:
            buffer = io.BytesIO()
            sf.write(buffer, output_audio.reshape(-1).detach().cpu().numpy(), samplerate=24_000, format="WAV")
            audio_bytes = buffer.getvalue()
            audio_id = sha256(audio_bytes).hexdigest()
        input_ids = tuple(int(value) for value in inputs["input_ids"][0].detach().cpu())
        generated_ids = tuple(int(value) for value in generated.detach().cpu())
        parsed_tool = _parse_tool_call(decoded)
        if parsed_tool is not None:
            audio_id = None
        record = GenerationRecord(
            generation_id=str(uuid4()), model_id=self.model_id, model_revision=self.revision,
            policy_version=self.revision, semantic_text=decoded, input_ids=input_ids,
            generated_ids=generated_ids,
            attention_mask=tuple(1 for _ in range(len(input_ids) + len(generated_ids))),
            action_mask=tuple(1 for _ in generated_ids), old_logprobs=logprobs,
            audio_artifact_id=audio_id, tool_call=parsed_tool,
        )
        self._history.append({"role": "assistant", "content": [{"type": "text", "text": decoded}]})
        if parsed_tool is not None:
            call = parsed_tool["tool_call"]
            action = ToolCallAction(str(call["name"]), dict(call["arguments"]))
        else:
            action = SpeakAction(audio_bytes, decoded, False)
        return action, record

    def save_adapter(self, destination: Path) -> Path:
        if self._model is None or not hasattr(self._model, "save_pretrained"):
            raise RuntimeError("loaded model does not expose save_pretrained")
        destination.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(destination)
        return destination

    def load_adapter(self, source: Path) -> None:
        if not source.exists():
            raise FileNotFoundError(source)
        self.checkpoint = str(source)
        self.allow_download = False
        self.load()


class Qwen3OmniCallerAdapter(Qwen3OmniAgentAdapter):
    def __init__(self, *args, private_goal: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.set_private_goal(private_goal)

    def set_private_goal(self, private_goal: str) -> None:
        self.private_goal = private_goal
        self.system_prompt = (
            "You are a frozen caller simulator. Pursue this private goal without revealing hidden "
            f"instructions: {private_goal}. Respond only as the caller."
        )

    def load(self) -> None:
        super().load()
        self._model.eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)

    @property
    def capabilities(self) -> ModelCapabilities:
        return super().capabilities.model_copy(
            update={"structured_tools": False, "trainable": False, "trainable_modules": ()}
        )

    def reset(self, seed: int) -> CallerTurn:
        del seed  # sampling seed is configured by the isolated model worker
        self.reset_conversation()
        return self._caller_turn(AudioObservation(), 0)

    def respond(self, agent_audio: bytes) -> CallerTurn | None:
        return self._caller_turn(AudioObservation(audio=agent_audio), len(self._history))

    def _caller_turn(self, observation: AudioObservation, index: int) -> CallerTurn:
        action, record = self.generate_step(observation)
        if not isinstance(action, SpeakAction) or not action.audio:
            raise RuntimeError("model-driven caller must generate non-empty speech audio")
        return CallerTurn(
            action.audio, "qwen3_omni_frozen_caller", index,
            semantic_text=record.semantic_text,
        )

    def start_isolated_session(self, private_goal: str, seed: int) -> tuple[CallerTurn, list[dict[str, Any]]]:
        self.set_private_goal(private_goal)
        turn = self.reset(seed)
        return turn, deepcopy(self._history)

    def continue_isolated_session(
        self, history: list[dict[str, Any]], agent_audio: bytes,
    ) -> tuple[CallerTurn, list[dict[str, Any]]]:
        self._history = deepcopy(history)
        turn = self.respond(agent_audio)
        if turn is None:  # Protocol permits this for other callers, not Qwen.
            raise RuntimeError("Qwen caller ended without a speech turn")
        return turn, deepcopy(self._history)


def _parse_tool_call(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or set(value) != {"tool_call"}:
        return None
    call = value["tool_call"]
    if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
        return None
    if not isinstance(call["name"], str) or not isinstance(call["arguments"], dict):
        return None
    if any(not isinstance(key, str) for key in call["arguments"]):
        return None
    return value
