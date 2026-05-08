import torch
import time
from arch_a import ArchAConfig, ArchAForCausalLM
from arch_a.kernels.dispatcher import KernelDispatcher

def run_sanity(device_name: str, use_compile: bool):
    try:
        device = torch.device(device_name)
        # Verify device availability (torch.device doesn't error out on creation if cuda not present but accessed later)
        if device.type == "cuda" and not torch.cuda.is_available():
            print(f"[{device_name}] Hardware unavailable, skipping.")
            return
        if device.type == "mps" and not torch.backends.mps.is_available():
             print(f"[{device_name}] Hardware unavailable, skipping.")
             return
    except RuntimeError:
        print(f"[{device_name}] Hardware unavailable, skipping.")
        return

    print(f"\n--- Testing Backend: {device_name} (Compile: {use_compile}) ---")
    config = ArchAConfig.for_debug()
    model = ArchAForCausalLM(config)
    model.to(device)

    dispatcher = KernelDispatcher(verbose=True)
    if use_compile:
        model = dispatcher.maybe_compile(model)
        print(f"Compile Status: {dispatcher.compile_status}")

    # Dummy inputs
    bsz, seq_len = 2, 64
    x = torch.randint(0, config.vocab_size, (bsz, seq_len), device=device)

    # Warmup
    try:
        model(x, labels=x)
    except Exception as e:
        print(f"[{device_name}] Warmup FAILED: {e}")
        return

    # Benchmark Forward
    start = time.time()
    try:
        out = model(x, labels=x)
        fw_time = time.time() - start
        print(f"[{device_name}] Forward Pass: {fw_time:.4f}s | Loss: {out.loss.item():.4f}")
    except Exception as e:
        print(f"[{device_name}] Forward FAILED: {e}")
        return

    # Benchmark Backward
    start = time.time()
    try:
        out.loss.backward()
        bw_time = time.time() - start
        print(f"[{device_name}] Backward Pass: {bw_time:.4f}s")
    except Exception as e:
        print(f"[{device_name}] Backward FAILED: {e}")
        return

    print(f"[{device_name}] Status: PASSED")

if __name__ == "__main__":
    print("Starting Runtime Sanity Matrix...")
    backends = ["cpu", "cuda", "mps"]
    for backend in backends:
        run_sanity(backend, use_compile=False)
        run_sanity(backend, use_compile=True)
    print("\nSanity Matrix Complete.")