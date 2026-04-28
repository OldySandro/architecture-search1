import json

INPUT_FILE = "nas_results.json"
OUTPUT_FILE = "selected_arc.json"

# 🔹 mode:
# "full"  → ambil semua field
# "clean" → hanya field penting (arsitektur)
MODE = "clean"

KEEP_FIELDS = {
    "arch_id", "arch_name", "arch_family",
    "vocab_size", "hidden_dim", "num_layers", "seq_len", "batch_size",
    "attn_type", "num_heads", "num_kv_heads", "head_dim",
    "window_size", "global_attn_layers",
    "ffn_type", "ffn_multiplier",
    "num_experts", "top_k_experts", "expert_capacity_factor",
    "norm_type", "pos_enc", "tie_embeddings",
    "optimizer_type",
    "use_flash_attn", "use_gradient_checkpointing",
    "use_mixed_precision", "use_torch_compile",
    "dropout",
    "param_count"
}


def clean_arch(arch):
    return {k: v for k, v in arch.items() if k in KEEP_FIELDS}


def find_arch_by_id(data, target_id):
    for arch in data.get("architectures", []):
        if arch.get("arch_id") == target_id:
            return arch
    return None


def main():
    target_id = input("Masukkan arch_id (contoh: ARC-1621): ").strip()

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    arch = find_arch_by_id(data, target_id)

    if not arch:
        print(f"❌ arch_id '{target_id}' tidak ditemukan.")
        return

    if MODE == "clean":
        arch = clean_arch(arch)

    output = {
        "architectures": [arch]
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"✅ Berhasil! Disimpan ke: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
