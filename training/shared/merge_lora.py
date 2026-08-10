import argparse
from unsloth import FastLanguageModel

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into base model for vLLM deployment.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained LoRA adapter (e.g., training/layer1_router/checkpoints/final)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the merged model (e.g., export/layer1_merged)")
    parser.add_argument("--format", type=str, choices=["safetensors", "gguf_16bit", "gguf_4bit"], default="safetensors", help="Export format")
    args = parser.parse_args()

    print(f"Loading LoRA model from {args.model_path}...")
    # Unsloth handles loading the base model automatically based on the adapter's config
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False, # We want to load in 16bit to merge
    )

    print(f"Merging and exporting model to {args.output_path} in {args.format} format...")
    
    if args.format == "safetensors":
        # Merges to 16-bit safetensors which is optimal for vLLM
        model.save_pretrained_merged(args.output_path, tokenizer, save_method="merged_16bit")
    elif args.format == "gguf_16bit":
        # Export to GGUF in 16-bit
        model.save_pretrained_gguf(args.output_path, tokenizer, quantization_method="f16")
    elif args.format == "gguf_4bit":
        # Export to GGUF in 4-bit (Q4_K_M)
        model.save_pretrained_gguf(args.output_path, tokenizer, quantization_method="q4_k_m")
        
    print(f"Merge complete! Model saved at {args.output_path}")

if __name__ == "__main__":
    main()
