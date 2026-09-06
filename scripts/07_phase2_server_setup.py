import os
import json
import torch
import multiprocessing

print("==================================================")
print("  PHASE 2: REMOTE SERVER ENVIRONMENT & SIZING CHECK")
print("==================================================")

cpu_count = multiprocessing.cpu_count()
ram_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
ram_gb = ram_bytes / (1024**3)

device_type = "CPU"
if torch.cuda.is_available():
    device_type = f"GPU ({torch.cuda.get_device_name(0)})"

config_recommendations = {
    "server_hostname": "panini (172.29.9.154)",
    "cpu_cores_available": cpu_count,
    "recommended_num_workers": min(8, cpu_count - 2),
    "system_ram_gb": round(ram_gb, 2),
    "device_selected": device_type,
    "batch_size_recommendation": {
        "training_batch_size": 16,
        "validation_batch_size": 32,
        "gradient_accumulation_steps": 2,
        "effective_batch_size": 32
    },
    "data_loader_pin_memory": False,
    "precision": "float32 (torch.float32) for E(3) Equivariant Tensor Clebsch-Gordan operations"
}

print("Server Resource Analysis:")
print(f"  - CPU Cores: {config_recommendations['cpu_cores_available']}")
print(f"  - Recommended PyTorch DataLoader Workers: {config_recommendations['recommended_num_workers']}")
print(f"  - System Memory (RAM): {config_recommendations['system_ram_gb']} GB")
print(f"  - Compute Acceleration Device: {config_recommendations['device_selected']}")
print("\nTraining Batch Sizing & Control Flow Recommendations:")
for k, v in config_recommendations['batch_size_recommendation'].items():
    print(f"  - {k}: {v}")

out_path = "/home/vibhan23/data/processed/phase2_environment_config.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(config_recommendations, f, indent=2)

print(f"\nSaved environment configuration to: {out_path}")
print("==================================================")
