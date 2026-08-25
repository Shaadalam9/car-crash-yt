"""Text model screening for real road crash footage and compilations."""

from __future__ import annotations

from typing import Any, Dict

from . import settings
from .model_loading import load_model_with_fallback
from .shared import (
    clamp_float,
    clean_text,
    log,
    normalise_bool,
    optional_text,
    recover_json,
    save_state,
    unload_model,
)


class TextMetadataJudge:
    def __init__(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log(f"Loading metadata model: {settings.TEXT_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(settings.TEXT_MODEL)
        self.model = load_model_with_fallback(
            AutoModelForCausalLM.from_pretrained,
            settings.TEXT_MODEL,
            device=settings.TEXT_DEVICE,
            load_in_4bit=settings.TEXT_MODEL_4BIT,
            label="metadata model",
        ).eval()

    def judge(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        messages = [{"role": "user", "content": self._prompt(metadata)}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=settings.TEXT_MAX_NEW_TOKENS,
            )
        prompt_length = inputs["input_ids"].shape[1]
        answer = self.tokenizer.decode(
            generated[0, prompt_length:], skip_special_tokens=True
        )
        data = recover_json(answer)
        if data is None:
            return {
                "include": False,
                "confidence": 0.0,
                "short_reason": "The metadata model did not return valid JSON.",
                "locality": None,
                "state": None,
                "country": None,
                "raw_response": answer,
                "error": "json_recovery_failed",
            }

        confidence = clamp_float(data.get("confidence"))
        include = normalise_bool(data.get("include")) and confidence >= settings.MIN_TEXT_CONFIDENCE
        return {
            "include": include,
            "confidence": confidence,
            "short_reason": clean_text(data.get("short_reason")),
            "locality": optional_text(data.get("locality")),
            "state": optional_text(data.get("state")),
            "country": optional_text(data.get("country")),
            "raw_response": answer,
            "error": None,
        }

    @staticmethod
    def _prompt(metadata: Dict[str, Any]) -> str:
        title = clean_text(metadata.get("title"))
        description = clean_text(metadata.get("description"))[:5000]
        return f"""
You screen YouTube metadata for a research dataset of real road traffic crash
footage. Include individual crashes and compilations that probably contain
genuine dashcam, CCTV, phone, action camera, or broadcast footage of collisions
or near collision events involving road users.

Reject video games, simulations, films, television drama, animation, crash
tests, motorsport-only footage, toy vehicles, commentary with no original
footage, emergency response footage with no crash event, and unrelated uses of
the word crash. Do not reject a candidate because it is very short or very
long. A video may be included even when no location is available.

Extract locality, state, and country only when directly supported by the title
or description. Do not infer them from appearance.

Title: {title}
Description: {description}

Return JSON only:
{{
  "include": true,
  "confidence": 0.0,
  "short_reason": "one factual sentence",
  "locality": null,
  "state": null,
  "country": null
}}
""".strip()


def pending_records(state: Dict[str, Any]) -> list[tuple[str, Dict[str, Any]]]:
    return [
        (video_id, record)
        for video_id, record in state.get("videos", {}).items()
        if isinstance(record, dict) and not isinstance(record.get("text_decision"), dict)
    ]


def run_text_stage(state: Dict[str, Any]) -> int:
    pending = pending_records(state)
    if not pending:
        return 0
    judge = TextMetadataJudge()
    processed = 0
    try:
        for video_id, record in pending:
            try:
                decision = judge.judge(record.get("metadata", {}))
                record["text_decision"] = decision
                record["status"] = "text_accepted" if decision["include"] else "text_rejected"
                record["error"] = decision.get("error")
            except KeyboardInterrupt:
                save_state(settings.STATE_JSON, state)
                raise
            except Exception as exc:
                record["status"] = "text_error"
                record["error"] = str(exc)
                log(f"Metadata filtering failed for {video_id}: {exc}")
            save_state(settings.STATE_JSON, state)
            processed += 1
    finally:
        unload_model(judge)
    return processed

