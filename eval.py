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

def eval_structured_output(model, tokenizer, device):
    """Sanity prompt to check structured JSON emission capability."""
    prompt = 'Generate a JSON object representing a user:\n```json\n{"name":'
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Very basic greedy generation stub
    model.eval()
    with torch.no_grad():
        generated = inputs
        for _ in range(20):
            out = model(generated)
            next_token = out.logits[0, -1].argmax(dim=-1).unsqueeze(0).unsqueeze(0)
            generated = torch.cat([generated, next_token], dim=-1)

    text = tokenizer.decode(generated[0])
    valid_json = False
    try:
        # Check if the model closed the braces logically
        if '}' in text:
            chunk = text.split('```json\n')[1].split('}')[0] + '}'
            json.loads(chunk)
            valid_json = True
    except Exception:
        pass

    return {"structured_text": text, "valid_json": valid_json}

def eval_needle_in_haystack(model, tokenizer, device, context_len=1024):
    """Basic retrieval probe."""
    needle = "The secret password is 'arch_a_rules'."
    haystack = "The quick brown fox jumps over the lazy dog. " * (context_len // 10)

    # Insert needle in the middle
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

    success = predicted_word.startswith("arch")
    return {"retrieval_success": success, "predicted_word": predicted_word}

if __name__ == "__main__":
    print("Evaluation harness ready. To use, import these functions into the trainer.")