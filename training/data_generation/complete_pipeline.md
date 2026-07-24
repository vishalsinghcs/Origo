```bash
# Step 1: Generate raw synthetic data (runs overnight)
python generate_layer1.py --target 7000 --output ../../datasets/raw/layer1
python generate_layer2.py --target 10000 --output ../../datasets/raw/layer2

# Step 2: Filter quality samples only
python ./filters/quality_filter.py --input ../../datasets/raw/layer1 --output ../../datasets/filtered/layer1
python ./filters/quality_filter.py --input ../../datasets/raw/layer2 --output ../../datasets/filtered/layer2

# Step 3: Convert to training format
python convert_layer1.py --input ../../datasets/filtered/layer1 --output ../../datasets/processed/layer1
python convert_layer2.py --input ../../datasets/filtered/layer2 --output ../../datasets/processed/layer2

# Step 4: Upload to Hugging Face
huggingface-cli upload origo/layer1-router-dataset ../../datasets/processed/layer1
huggingface-cli upload origo/layer2-specialist-dataset ../../datasets/processed/layer2
```