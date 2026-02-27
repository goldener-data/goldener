# Goldener Tools

This directory contains utility tools for working with Goldener.

## optimize_gold_descriptor.py

A comprehensive benchmarking tool to optimize the throughput and memory usage of GoldDescriptor with a Vectorizer.

### Features

- **Model Layer Listing**: List all layers of a Timm model to help select which layers to use for feature extraction
- **Feature Extractor Benchmarking**: Measure memory usage (RAM and GPU VRAM) and inference time for TorchGoldFeatureExtractor
- **GoldDescriptor Benchmarking**: End-to-end benchmarking including vectorization
- **Memory Tracking**: Real-time monitoring of RAM and GPU VRAM usage
- **Performance Metrics**: Inference time per batch and throughput (samples/second)
- **Optimal Settings Recommendation**: Based on available memory, the tool recommends optimal batch sizes and estimates performance using linear interpolation

### Prerequisites

Install the required dependencies:

```bash
pip install timm psutil pynvml
```

Or add them to your environment:

```bash
uv pip install timm psutil pynvml
```

### Usage

#### Basic Usage

List all layers of a model:

```bash
python tools/optimize_gold_descriptor.py --model resnet18 --list-layers-only
```

Benchmark a model with default settings:

```bash
python tools/optimize_gold_descriptor.py --model resnet18 --input-size 224 --batch-size 32
```

#### Advanced Usage

Benchmark with specific layers:

```bash
python tools/optimize_gold_descriptor.py \
    --model resnet50 \
    --input-size 224 \
    --batch-size 16 \
    --layers layer3 layer4
```

Benchmark with vectorizer:

```bash
python tools/optimize_gold_descriptor.py \
    --model resnet18 \
    --input-size 224 \
    --batch-size 32 \
    --vectorize \
    --vector-count 100
```

Full end-to-end benchmark with custom settings:

```bash
python tools/optimize_gold_descriptor.py \
    --model efficientnet_b0 \
    --input-size 224 \
    --batch-size 64 \
    --num-samples 500 \
    --min-pxt-insert-size 200 \
    --vectorize \
    --vector-count 200
```

Skip specific benchmarks:

```bash
# Skip feature extractor benchmark, only run descriptor benchmark
python tools/optimize_gold_descriptor.py \
    --model resnet18 \
    --input-size 224 \
    --batch-size 32 \
    --skip-extractor-benchmark

# Skip descriptor benchmark, only run feature extractor benchmark
python tools/optimize_gold_descriptor.py \
    --model resnet18 \
    --input-size 224 \
    --batch-size 32 \
    --skip-descriptor-benchmark
```

### Command-Line Arguments

#### Model Configuration

- `--model`: Timm model name (required, e.g., `resnet18`, `efficientnet_b0`, `vit_base_patch16_224`)
- `--input-size`: Input image size in pixels (default: 224)
- `--batch-size`: Batch size for benchmarking (default: 32)
- `--channels`: Number of input channels (default: 3)

#### Feature Extractor Configuration

- `--layers`: Space-separated list of layers to extract features from (default: last layer)
- `--layer-fusion`: Strategy for fusing features from multiple layers (default: `concat`)
  - Choices: `concat`, `add`, `average`, `max`

#### Vectorizer Configuration

- `--vectorize`: Enable vectorizer in the benchmark
- `--vector-count`: Number of vectors to keep when vectorizing (default: 100)
- `--min-pxt-insert-size`: Minimum number of rows before inserting into PixelTable (default: 100)

#### Benchmark Configuration

- `--num-samples`: Number of samples for GoldDescriptor benchmark (default: 100)
- `--num-warmup`: Number of warmup iterations for feature extractor (default: 5)
- `--num-iterations`: Number of benchmark iterations for feature extractor (default: 10)

#### Device Configuration

- `--device`: Device to use (default: `auto`)
  - Choices: `auto`, `cpu`, `cuda`

#### Actions

- `--list-layers-only`: Only list model layers and exit
- `--skip-extractor-benchmark`: Skip feature extractor benchmark
- `--skip-descriptor-benchmark`: Skip GoldDescriptor benchmark

### Output

The tool provides detailed output including:

1. **Model Information**: List of all layers in the model
2. **Feature Extractor Benchmark**:
   - RAM usage (before, after, peak, increase)
   - GPU VRAM usage (before, after, peak, increase)
   - Inference time per batch
   - Throughput (samples/second)
3. **GoldDescriptor Benchmark**:
   - End-to-end memory usage
   - Processing time per sample
   - Throughput
4. **Optimal Settings Recommendation**:
   - Per-sample memory usage
   - Recommended batch size based on available memory
   - Estimated performance at optimal settings
   - Warnings or suggestions for improvement

### Examples

#### Example 1: Explore Model Layers

```bash
python tools/optimize_gold_descriptor.py --model resnet50 --list-layers-only
```

Output will show all layers like:
```
Model: resnet50
================================================================================

Model Layers:
Layer Name                                          Layer Type                    
--------------------------------------------------------------------------------
conv1                                               Conv2d                        
bn1                                                 BatchNorm2d                   
relu                                                ReLU                          
maxpool                                             MaxPool2d                     
layer1                                              Sequential                    
layer1.0                                            Bottleneck                    
...
```

#### Example 2: Basic Benchmarking

```bash
python tools/optimize_gold_descriptor.py \
    --model resnet18 \
    --input-size 224 \
    --batch-size 32
```

This will benchmark both the feature extractor and GoldDescriptor, showing memory usage and performance metrics.

#### Example 3: Optimize for Your Hardware

```bash
python tools/optimize_gold_descriptor.py \
    --model efficientnet_b0 \
    --input-size 224 \
    --batch-size 16 \
    --num-samples 200
```

The tool will analyze your system's available memory and recommend an optimal batch size.

### Tips

1. **Start Small**: Begin with a small batch size and increase it based on recommendations
2. **Check Layers**: Use `--list-layers-only` to explore which layers might be useful for feature extraction
3. **Monitor Memory**: Watch the memory usage to understand your hardware limits
4. **Vectorization**: Enable vectorization (`--vectorize`) when you need to process large feature maps
5. **Custom Layers**: Experiment with different layer combinations to find the best trade-off between feature quality and performance

### Troubleshooting

- **Out of Memory**: Reduce `--batch-size` or disable vectorization
- **Slow Performance**: Check if you're using GPU (`--device cuda`) and ensure CUDA is properly installed
- **Missing Dependencies**: Install `timm`, `psutil`, and `pynvml` as shown in Prerequisites
- **Model Not Found**: Verify the model name using timm's model list: `python -c "import timm; print(timm.list_models())"`
