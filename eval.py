import torch
import math
import json

def eval_perplexity(model, dataloader, device, max_steps=100):
    """Calculates validation perplexity across a dataset."""
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for batch in dataloader:
            if steps >= max_steps:
                break
            batch = batch.to(device)
            out = model(batch, labels=batch)
            total_loss += out.loss.item()
            steps += 1

    avg_loss = total_loss / max(1, steps)
    ppl = math.exp(min(avg_loss, 20.0))
    return avg_loss, ppl

def eval_structured_output(model, tokenizer, device, num_prompts=5):
    """Sanity prompt to check structured JSON emission capability."""
    prompts = [
        'Generate a JSON object representing a user:\n```json\n{"name":',
        'Provide a JSON configuration:\n```json\n{"setting":',
        'Create a JSON for a book:\n```json\n{"title":',
        'Write JSON for a car:\n```json\n{"brand":',
        'Output a JSON coordinate:\n```json\n{"x":'
    ]
    prompts = prompts[:num_prompts]

    valid_count = 0
    results = []

    model.eval()

    # Force deterministic decoding
    old_state = torch.get_rng_state()
    if torch.cuda.is_available():
        old_cuda_state = torch.cuda.get_rng_state()
    torch.manual_seed(42)

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            generated = inputs
            for _ in range(20):
                out = model(generated)
                # Deterministic greedy decoding
                next_token = out.logits[0, -1].argmax(dim=-1).unsqueeze(0).unsqueeze(0)
                generated = torch.cat([generated, next_token], dim=-1)

            text = tokenizer.decode(generated[0])
            valid_json = False
            try:
                if '}' in text and '```json\n' in text:
                    chunk = text.split('```json\n')[1].split('}')[0] + '}'
                    json.loads(chunk)
                    valid_json = True
                    valid_count += 1
            except Exception:
                pass
            results.append({"prompt": prompt, "structured_text": text, "valid_json": valid_json})

    torch.set_rng_state(old_state)
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(old_cuda_state)

    validity_rate = valid_count / max(1, len(prompts))
    return {"validity_rate": validity_rate, "results": results}

def eval_needle_in_haystack(model, tokenizer, device, context_len=1024, insertion_pos="middle"):
    """Long-context retrieval probe."""
    needle = "The secret password is 'arch_a_rules'."
    haystack = "The quick brown fox jumps over the lazy dog. " * (context_len // 10)

    if insertion_pos == "start":
        prompt = needle + haystack
    elif insertion_pos == "end":
        prompt = haystack + needle
    else:  # middle
        midpoint = len(haystack) // 2
        prompt = haystack[:midpoint] + needle + haystack[midpoint:]

    question = "\nWhat is the secret password? The secret password is '"
    full_prompt = prompt + question
    inputs = tokenizer(full_prompt, return_tensors="pt").input_ids.to(device)

    model.eval()
    with torch.no_grad():
        out = model(inputs)
        # Check if the next highest probability token matches 'arch'
        next_token_id = out.logits[0, -1].argmax(dim=-1).item()
        predicted_word = tokenizer.decode([next_token_id]).strip()

    exact_match = predicted_word == "arch_a_rules"
    contains_match = "arch" in predicted_word.lower()

    return {
        "exact_match": exact_match,
        "contains_match": contains_match,
        "predicted_word": predicted_word,
        "context_len": context_len,
        "insertion_pos": insertion_pos
    }

if __name__ == "__main__":
    print("Evaluation harness ready. To use, import these functions into the trainer.")