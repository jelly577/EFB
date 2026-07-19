"""Hugging Face implementation of likelihood scoring and suffix resampling."""

from __future__ import annotations

from typing import Any

from .backend import GeneratedText


class HuggingFaceBackend:
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.8,
        seed: int = 42,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.temperature = temperature
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.model.eval()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @property
    def device(self) -> Any:
        return next(self.model.parameters()).device

    def _prompt_ids(self, prompt: str) -> Any:
        return self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids.to(self.device)

    def _continuation_ids(self, text: str) -> Any:
        return self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(self.device)

    def token_count(self, text: str) -> int:
        return int(self._continuation_ids(text).shape[1])

    def generate_initial(self, prompt: str, max_new_tokens: int) -> GeneratedText:
        prompt_ids = self._prompt_ids(prompt)
        output = self.model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        continuation_ids = output[0, prompt_ids.shape[1] :]
        text = self.tokenizer.decode(continuation_ids, skip_special_tokens=True)
        return GeneratedText(text=text, token_count=int(continuation_ids.shape[0]))

    def score(self, prompt: str, continuation: str) -> float:
        prompt_ids = self._prompt_ids(prompt)
        continuation_ids = self._continuation_ids(continuation)
        if continuation_ids.shape[1] == 0:
            return float("-inf")

        input_ids = self.torch.cat((prompt_ids, continuation_ids), dim=1)
        with self.torch.inference_mode():
            logits = self.model(input_ids).logits[:, :-1, :].float()
        targets = input_ids[:, 1:]
        log_probabilities = self.torch.log_softmax(logits, dim=-1)
        token_log_probabilities = log_probabilities.gather(
            2,
            targets.unsqueeze(-1),
        ).squeeze(-1)
        continuation_start = prompt_ids.shape[1] - 1
        return float(token_log_probabilities[:, continuation_start:].sum().item())

    def resample_suffix(
        self,
        prompt: str,
        continuation: str,
        split_token_index: int,
        max_new_tokens: int,
    ) -> GeneratedText:
        prompt_ids = self._prompt_ids(prompt)
        continuation_ids = self._continuation_ids(continuation)
        if not 0 <= split_token_index < continuation_ids.shape[1]:
            raise ValueError("split_token_index is outside the continuation")

        kept_prefix = continuation_ids[:, :split_token_index]
        model_input = self.torch.cat((prompt_ids, kept_prefix), dim=1)
        output = self.model.generate(
            model_input,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        new_suffix = output[:, model_input.shape[1] :]
        proposed_ids = self.torch.cat((kept_prefix, new_suffix), dim=1)[0]
        text = self.tokenizer.decode(proposed_ids, skip_special_tokens=True)
        return GeneratedText(text=text, token_count=int(new_suffix.shape[1]))

