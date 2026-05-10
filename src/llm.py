"""
Language model integration — generates answers from retrieved context.

Default model: Qwen/Qwen3-0.6B (via HuggingFace transformers).
Supports swapping to any HuggingFace model or a local vLLM server (bonus).
Context is trimmed to fit within the model's token limit before each call.
"""

# tokenizer and model loading from HuggingFace
from transformers import AutoTokenizer, AutoModelForCausalLM
# tensors, device picking, no_grad for inference
import torch
# HTTP calls to the vLLM server (bonus)
import requests
from typing import Any, cast

DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_CONTEXT_CHARACTERS = 3000
MAX_NEW_TOKENS = 256


def load_model(
    model_name: str = DEFAULT_MODEL_NAME, force_cpu: bool = False
) -> tuple:
    """
    Load a HuggingFace model and tokenizer into memory.

    Args:
        model_name: HuggingFace model identifier.
        force_cpu: If True, skip CUDA and load on CPU. Used as a fallback
            when CUDA is detected but the GPU is incompatible at runtime.

    Returns:
        A tuple of (tokenizer, model).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    use_cuda = torch.cuda.is_available() and not force_cpu
    device_map = "cuda" if use_cuda else "cpu"
    dtype = torch.float16 if use_cuda else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, device_map=device_map
    )
    return tokenizer, model


def build_prompt(question: str, context_chunks: list[str]) -> list[dict]:
    """
    Build a chat message list from the question and retrieved chunks,
    trimmed to MAX_CONTEXT_CHARACTERS.

    Args:
        question: The question to answer.
        context_chunks: List of retrieved text chunks.

    Returns:
        A list of chat messages for apply_chat_template.
    """
    context = ''
    for chunk in context_chunks:
        if len(context) + len(chunk) > MAX_CONTEXT_CHARACTERS:
            break
        context += chunk + '\n\n'
    return [
        {"role": "system", "content": (
            "You are a technical assistant for the vLLM project. "
            "Answer concisely using only the provided context. "
            "Quote exact values, commands, and identifiers verbatim."
        )},
        {"role": "user", "content": (
            f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )},
    ]


def generate_answer(
    prompt: list[dict], model: Any, tokenizer: Any
) -> str:
    """
    Run inference and return the model's answer as a plain string.

    Args:
        prompt: Chat messages from build_prompt.
        model: The loaded HuggingFace model.
        tokenizer: The loaded HuggingFace tokenizer.

    Returns:
        The generated answer as a plain string.
    """
    # enable_thinking=False disables Qwen3's chain-of-thought
    text = tokenizer.apply_chat_template(
        prompt, tokenize=False,
        add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors='pt')
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            repetition_penalty=1.2,
            do_sample=False,
        )
    new_tokens = output_ids[0][inputs['input_ids'].shape[1]:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return cast(str, decoded)


def answer_question(
    question: str,
    retrieved_chunks: list[dict],
    model: Any,
    tokenizer: Any
) -> str:
    """
    End-to-end: build prompt, run model, return answer.

    Args:
        question: The question to answer.
        retrieved_chunks: List of chunk dicts with a 'text' key.
        model: The loaded HuggingFace model.
        tokenizer: The loaded HuggingFace tokenizer.

    Returns:
        The generated answer as a plain string.
    """
    context_chunks = [chunk['text'] for chunk in retrieved_chunks]
    prompt = build_prompt(question, context_chunks)
    return generate_answer(prompt, model, tokenizer)


def load_vllm_model(model_name: str = DEFAULT_MODEL_NAME) -> dict:
    """
    Load a model via a local vLLM server for faster inference (bonus).

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        A dict with 'model' and 'base_url' for use with
        generate_answer_with_vllm.
    """
    # the vLLM server speaks the OpenAI API, we just keep the URL around
    return {'model': model_name, 'base_url': 'http://localhost:8000/v1'}


def generate_answer_with_vllm(prompt: list[dict], vllm_client: dict) -> str:
    """
    Generate an answer using a local vLLM server (bonus).

    Args:
        prompt: Chat messages from build_prompt.
        vllm_client: The dict returned by load_vllm_model.

    Returns:
        The generated answer as a plain string.
    """
    response = requests.post(
        f"{vllm_client['base_url']}/chat/completions",
        json={
            'model': vllm_client['model'],
            'messages': prompt,
            'max_tokens': MAX_NEW_TOKENS,
        }
    )
    return cast(str, response.json()['choices'][0]['message']['content'])
