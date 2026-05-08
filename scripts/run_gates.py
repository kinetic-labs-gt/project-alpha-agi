import os
import torch
import shutil
from arch_a import ArchAConfig, ArchAForCausalLM
from arch_a.training.checkpointing import CheckpointManager

def test_smoke_nan(device):
    """1k-step (mocked as 50 steps for fast CI gating) smoke test for NaN/Loss trend."""
    print("--- Running Smoke Test (No NaNs, Loss Trending Down) ---")
    config = ArchAConfig.for_debug()
    model = ArchAForCausalLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    bsz, seq_len = 2, 64
    x = torch.randint(0, config.vocab_size, (bsz, seq_len), device=device)

    initial_loss = None
    final_loss = None

    for step in range(50):
        optimizer.zero_grad()
        out = model(x, labels=x)
        loss = out.loss

        assert not torch.isnan(loss), f"NaN encountered at step {step}"

        if step == 0:
            initial_loss = loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = loss.item()

    assert final_loss < initial_loss, f"Loss did not trend down: {initial_loss} -> {final_loss}"
    print(f"Smoke Test PASSED: Loss {initial_loss:.4f} -> {final_loss:.4f}")

def test_checkpoint_resume(device):
    """Checkpoint resume test: Ensures identical continuation behavior."""
    print("\n--- Running Checkpoint Resume Test ---")
    config = ArchAConfig.for_debug()
    model1 = ArchAForCausalLM(config).to(device)
    optimizer1 = torch.optim.AdamW(model1.parameters(), lr=1e-3)

    save_dir = "tmp_ckpt_test"
    os.makedirs(save_dir, exist_ok=True)
    manager = CheckpointManager(save_dir=save_dir)

    bsz, seq_len = 2, 64
    x = torch.randint(0, config.vocab_size, (bsz, seq_len), device=device)

    # Train model1 for 5 steps
    for _ in range(5):
        out = model1(x, labels=x)
        out.loss.backward()
        optimizer1.step()

    # Save checkpoint
    manager.save_checkpoint(model1, optimizer1, None, 5, config.to_dict(), 0.0)
    ckpt_path = os.path.join(save_dir, "step_5_loss_0.0000.pt")

    # Generate gold standard step 6 from model 1
    optimizer1.zero_grad()
    gold_out = model1(x, labels=x)

    # Load model2 from checkpoint
    model2 = ArchAForCausalLM(config).to(device)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    manager.load_checkpoint(ckpt_path, model2, optimizer2, None)

    # Generate test step 6 from model 2
    optimizer2.zero_grad()
    test_out = model2(x, labels=x)

    # Verify exact loss match
    diff = abs(gold_out.loss.item() - test_out.loss.item())
    assert diff < 1e-6, f"Resume divergence detected! Loss difference: {diff}"
    print("Checkpoint Resume Test PASSED: Resumed loss identically matches continuous loss.")

    shutil.rmtree(save_dir, ignore_errors=True)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Pre-train Gates on {device}...")

    test_smoke_nan(device)
    test_checkpoint_resume(device)

    print("\nAll Go/No-Go Gates PASSED. Safe for 10k-step pilot and full pre-training.")