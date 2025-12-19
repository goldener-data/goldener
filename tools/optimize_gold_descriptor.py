#!/usr/bin/env python3
"""
GoldDescriptor Optimization Tool

This tool helps optimize the throughput and memory usage of GoldDescriptor with a Vectorizer.
It allows users to:
1. Define a Timm model and see its layers
2. Configure TorchGoldFeatureExtractor and benchmark it
3. Benchmark memory usage (RAM and GPU VRAM) and inference time
4. Configure GoldDescriptor with vectorizer and benchmark end-to-end performance
5. Get optimal settings recommendations based on available memory

Usage:
    python optimize_gold_descriptor.py --model resnet18 --input-size 224 --batch-size 32
"""

import argparse
import gc
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Add parent directory to path to import goldener modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from goldener.extract import (
    TorchGoldFeatureExtractor,
    TorchGoldFeatureExtractorConfig,
    FeatureFusionStrategy,
)
from goldener.describe import GoldDescriptor
from goldener.vectorize import TensorVectorizer, Filter2DWithCount, FilterLocation

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm is not installed. Install with: pip install timm")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil is not installed. Install with: pip install psutil")

try:
    import pynvml
    PYNVML_AVAILABLE = True
    try:
        pynvml.nvmlInit()
    except Exception:
        PYNVML_AVAILABLE = False
except ImportError:
    PYNVML_AVAILABLE = False


@dataclass
class MemoryUsage:
    """Memory usage statistics."""
    ram_mb: float
    gpu_vram_mb: float = 0.0


@dataclass
class BenchmarkResult:
    """Benchmark result with memory and timing information."""
    memory_before: MemoryUsage
    memory_after: MemoryUsage
    memory_peak: MemoryUsage
    inference_time_ms: float
    throughput_samples_per_sec: float


class DummyDataset(Dataset):
    """Dummy dataset for benchmarking."""
    
    def __init__(self, size: int, channels: int, height: int, width: int, num_samples: int = 1000):
        """Initialize dummy dataset.
        
        Args:
            size: Batch size
            channels: Number of channels
            height: Image height
            width: Image width
            num_samples: Total number of samples in the dataset
        """
        self.size = size
        self.channels = channels
        self.height = height
        self.width = width
        self.num_samples = num_samples
        
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary with 'data' and 'idx' keys
        """
        return {
            "data": torch.randn(self.channels, self.height, self.width),
            "idx": idx,
        }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate function for DataLoader.
    
    Args:
        batch: List of samples
        
    Returns:
        Batched data
    """
    return {
        "data": torch.stack([item["data"] for item in batch]),
        "idx": [item["idx"] for item in batch],
    }


def get_memory_usage() -> MemoryUsage:
    """Get current memory usage.
    
    Returns:
        MemoryUsage object with RAM and GPU VRAM usage
    """
    ram_mb = 0.0
    if PSUTIL_AVAILABLE:
        process = psutil.Process()
        ram_mb = process.memory_info().rss / 1024 / 1024
    
    gpu_vram_mb = 0.0
    if PYNVML_AVAILABLE and torch.cuda.is_available():
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_vram_mb = info.used / 1024 / 1024
        except Exception:
            pass
    
    return MemoryUsage(ram_mb=ram_mb, gpu_vram_mb=gpu_vram_mb)


def print_model_layers(model: torch.nn.Module, model_name: str) -> None:
    """Print all layers of a model.
    
    Args:
        model: PyTorch model
        model_name: Name of the model
    """
    print(f"\n{'='*80}")
    print(f"Model: {model_name}")
    print(f"{'='*80}")
    print("\nModel Layers:")
    print(f"{'Layer Name':<50} {'Layer Type':<30}")
    print("-" * 80)
    
    for name, module in model.named_modules():
        if name:  # Skip the root module
            print(f"{name:<50} {type(module).__name__:<30}")
    
    print(f"\nTotal layers: {sum(1 for _ in model.named_modules()) - 1}")
    print(f"{'='*80}\n")


def benchmark_feature_extractor(
    extractor: TorchGoldFeatureExtractor,
    input_size: tuple[int, int, int, int],
    device: torch.device,
    num_warmup: int = 5,
    num_iterations: int = 10,
) -> BenchmarkResult:
    """Benchmark feature extractor performance.
    
    Args:
        extractor: Feature extractor to benchmark
        input_size: Input tensor size (batch, channels, height, width)
        device: Device to run on
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations
        
    Returns:
        BenchmarkResult with memory and timing information
    """
    # Cleanup before benchmark
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Get baseline memory
    memory_before = get_memory_usage()
    
    # Create dummy input
    dummy_input = torch.randn(input_size).to(device)
    
    # Warmup
    for _ in range(num_warmup):
        _ = extractor.extract_and_fuse(dummy_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    # Benchmark
    memory_peak = get_memory_usage()
    start_time = time.time()
    
    for _ in range(num_iterations):
        _ = extractor.extract_and_fuse(dummy_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Track peak memory
        current_memory = get_memory_usage()
        memory_peak = MemoryUsage(
            ram_mb=max(memory_peak.ram_mb, current_memory.ram_mb),
            gpu_vram_mb=max(memory_peak.gpu_vram_mb, current_memory.gpu_vram_mb),
        )
    
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    avg_time_ms = (total_time / num_iterations) * 1000
    throughput = (input_size[0] * num_iterations) / total_time
    
    memory_after = get_memory_usage()
    
    return BenchmarkResult(
        memory_before=memory_before,
        memory_after=memory_after,
        memory_peak=memory_peak,
        inference_time_ms=avg_time_ms,
        throughput_samples_per_sec=throughput,
    )


def benchmark_gold_descriptor(
    descriptor: GoldDescriptor,
    dataset: Dataset,
    device: torch.device,
    num_samples: int = 100,
) -> BenchmarkResult:
    """Benchmark GoldDescriptor performance.
    
    Args:
        descriptor: GoldDescriptor to benchmark
        dataset: Dataset to process
        device: Device to run on
        num_samples: Number of samples to process
        
    Returns:
        BenchmarkResult with memory and timing information
    """
    # Cleanup before benchmark
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Get baseline memory
    memory_before = get_memory_usage()
    
    # Create a limited dataset
    limited_dataset = torch.utils.data.Subset(dataset, range(num_samples))
    
    # Benchmark
    memory_peak = memory_before
    start_time = time.time()
    
    # We need to use a temporary table path for benchmarking
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        table_path = os.path.join(tmpdir, "benchmark_table")
        descriptor.table_path = table_path
        
        try:
            _ = descriptor.describe_in_dataset(limited_dataset)
        except Exception as e:
            print(f"Warning: Error during benchmarking: {e}")
            # Continue with partial results
        
        # Track peak memory
        memory_peak = get_memory_usage()
    
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    avg_time_ms = (total_time / num_samples) * 1000
    throughput = num_samples / total_time
    
    memory_after = get_memory_usage()
    
    return BenchmarkResult(
        memory_before=memory_before,
        memory_after=memory_after,
        memory_peak=memory_peak,
        inference_time_ms=avg_time_ms,
        throughput_samples_per_sec=throughput,
    )


def print_benchmark_results(results: BenchmarkResult, title: str) -> None:
    """Print benchmark results in a formatted way.
    
    Args:
        results: Benchmark results to print
        title: Title for the results section
    """
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    
    print(f"\nMemory Usage:")
    print(f"  RAM Before:     {results.memory_before.ram_mb:.2f} MB")
    print(f"  RAM After:      {results.memory_after.ram_mb:.2f} MB")
    print(f"  RAM Peak:       {results.memory_peak.ram_mb:.2f} MB")
    print(f"  RAM Increase:   {results.memory_after.ram_mb - results.memory_before.ram_mb:.2f} MB")
    
    if torch.cuda.is_available():
        print(f"\n  GPU VRAM Before: {results.memory_before.gpu_vram_mb:.2f} MB")
        print(f"  GPU VRAM After:  {results.memory_after.gpu_vram_mb:.2f} MB")
        print(f"  GPU VRAM Peak:   {results.memory_peak.gpu_vram_mb:.2f} MB")
        print(f"  GPU VRAM Increase: {results.memory_after.gpu_vram_mb - results.memory_before.gpu_vram_mb:.2f} MB")
    
    print(f"\nPerformance:")
    print(f"  Inference Time: {results.inference_time_ms:.2f} ms")
    print(f"  Throughput:     {results.throughput_samples_per_sec:.2f} samples/sec")
    
    print(f"{'='*80}\n")


def recommend_optimal_settings(
    available_ram_mb: float,
    available_vram_mb: float,
    results: BenchmarkResult,
    current_batch_size: int,
) -> dict[str, Any]:
    """Recommend optimal settings based on available memory and benchmark results.
    
    Uses linear interpolation to estimate optimal batch size.
    
    Args:
        available_ram_mb: Available RAM in MB
        available_vram_mb: Available GPU VRAM in MB
        results: Benchmark results from current settings
        current_batch_size: Current batch size used in benchmark
        
    Returns:
        Dictionary with recommended settings
    """
    print(f"\n{'='*80}")
    print("Optimal Settings Recommendation")
    print(f"{'='*80}")
    
    # Calculate memory usage per sample
    ram_per_sample = (results.memory_peak.ram_mb - results.memory_before.ram_mb) / current_batch_size
    vram_per_sample = (results.memory_peak.gpu_vram_mb - results.memory_before.gpu_vram_mb) / current_batch_size
    
    # Calculate optimal batch size based on available memory
    # Use 80% of available memory as a safety margin
    safety_factor = 0.8
    
    optimal_batch_size_ram = int((available_ram_mb * safety_factor) / ram_per_sample) if ram_per_sample > 0 else float('inf')
    optimal_batch_size_vram = int((available_vram_mb * safety_factor) / vram_per_sample) if vram_per_sample > 0 else float('inf')
    
    # Take the minimum of the two
    optimal_batch_size = max(1, min(optimal_batch_size_ram, optimal_batch_size_vram))
    
    # Estimate performance at optimal batch size
    estimated_throughput = results.throughput_samples_per_sec * (optimal_batch_size / current_batch_size)
    estimated_time_ms = results.inference_time_ms * (current_batch_size / optimal_batch_size)
    
    recommendations = {
        "optimal_batch_size": optimal_batch_size,
        "estimated_throughput": estimated_throughput,
        "estimated_inference_time_ms": estimated_time_ms,
        "ram_per_sample_mb": ram_per_sample,
        "vram_per_sample_mb": vram_per_sample,
    }
    
    print(f"\nCurrent Settings:")
    print(f"  Batch Size: {current_batch_size}")
    print(f"  RAM Usage:  {results.memory_peak.ram_mb - results.memory_before.ram_mb:.2f} MB")
    if torch.cuda.is_available():
        print(f"  VRAM Usage: {results.memory_peak.gpu_vram_mb - results.memory_before.gpu_vram_mb:.2f} MB")
    
    print(f"\nAvailable Resources:")
    print(f"  Available RAM:  {available_ram_mb:.2f} MB")
    if torch.cuda.is_available():
        print(f"  Available VRAM: {available_vram_mb:.2f} MB")
    
    print(f"\nPer-Sample Memory Usage:")
    print(f"  RAM per sample:  {ram_per_sample:.2f} MB")
    if torch.cuda.is_available():
        print(f"  VRAM per sample: {vram_per_sample:.2f} MB")
    
    print(f"\nRecommended Settings:")
    print(f"  Optimal Batch Size: {optimal_batch_size}")
    print(f"  Estimated Throughput: {estimated_throughput:.2f} samples/sec")
    print(f"  Estimated Inference Time: {estimated_time_ms:.2f} ms")
    
    if optimal_batch_size < current_batch_size:
        print(f"\n⚠️  Warning: Recommended batch size is smaller than current batch size.")
        print(f"   Consider reducing batch size to avoid out-of-memory errors.")
    elif optimal_batch_size > current_batch_size * 2:
        print(f"\n✓ Good news: You can increase batch size for better throughput!")
        print(f"   Try increasing to {optimal_batch_size} for optimal performance.")
    else:
        print(f"\n✓ Current batch size is reasonably close to optimal.")
    
    print(f"{'='*80}\n")
    
    return recommendations


def main():
    """Main function to run the optimization tool."""
    parser = argparse.ArgumentParser(
        description="Optimize GoldDescriptor throughput and memory usage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark a ResNet18 model
  python optimize_gold_descriptor.py --model resnet18 --input-size 224 --batch-size 32

  # Benchmark with specific layers
  python optimize_gold_descriptor.py --model resnet50 --input-size 224 --batch-size 16 --layers layer4

  # Benchmark with vectorizer
  python optimize_gold_descriptor.py --model resnet18 --input-size 224 --batch-size 32 --vectorize --vector-count 100

  # Full end-to-end benchmark
  python optimize_gold_descriptor.py --model efficientnet_b0 --input-size 224 --batch-size 64 --num-samples 500
        """,
    )
    
    # Model configuration
    parser.add_argument("--model", type=str, required=True, help="Timm model name (e.g., resnet18, efficientnet_b0)")
    parser.add_argument("--input-size", type=int, default=224, help="Input image size (default: 224)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for benchmarking (default: 32)")
    parser.add_argument("--channels", type=int, default=3, help="Number of input channels (default: 3)")
    
    # Feature extractor configuration
    parser.add_argument("--layers", type=str, nargs="+", help="Layers to extract features from (default: last layer)")
    parser.add_argument("--layer-fusion", type=str, default="concat", choices=["concat", "add", "average", "max"],
                        help="Layer fusion strategy (default: concat)")
    
    # Vectorizer configuration
    parser.add_argument("--vectorize", action="store_true", help="Include vectorizer in benchmark")
    parser.add_argument("--vector-count", type=int, default=100, help="Number of vectors to keep when vectorizing (default: 100)")
    parser.add_argument("--min-pxt-insert-size", type=int, default=100, 
                        help="Minimum number of rows to accumulate before inserting into PixelTable (default: 100)")
    
    # Benchmark configuration
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples for GoldDescriptor benchmark (default: 100)")
    parser.add_argument("--num-warmup", type=int, default=5, help="Number of warmup iterations (default: 5)")
    parser.add_argument("--num-iterations", type=int, default=10, help="Number of benchmark iterations (default: 10)")
    
    # Device configuration
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device to use (default: auto)")
    
    # Actions
    parser.add_argument("--skip-extractor-benchmark", action="store_true", help="Skip feature extractor benchmark")
    parser.add_argument("--skip-descriptor-benchmark", action="store_true", help="Skip GoldDescriptor benchmark")
    parser.add_argument("--list-layers-only", action="store_true", help="Only list model layers and exit")
    
    args = parser.parse_args()
    
    # Check dependencies
    if not TIMM_AVAILABLE:
        print("Error: timm is required. Install with: pip install timm")
        sys.exit(1)
    
    if not PSUTIL_AVAILABLE:
        print("Warning: psutil is not installed. Memory tracking will be limited.")
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"\nUsing device: {device}")
    
    # Load model
    print(f"\nLoading model: {args.model}")
    try:
        model = timm.create_model(args.model, pretrained=False)
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Print model layers
    print_model_layers(model, args.model)
    
    if args.list_layers_only:
        return
    
    # Configure feature extractor
    layer_fusion_map = {
        "concat": FeatureFusionStrategy.CONCAT,
        "add": FeatureFusionStrategy.ADD,
        "average": FeatureFusionStrategy.AVERAGE,
        "max": FeatureFusionStrategy.MAX,
    }
    
    config = TorchGoldFeatureExtractorConfig(
        model=model,
        layers=args.layers,
        layer_fusion=layer_fusion_map[args.layer_fusion],
    )
    
    extractor = TorchGoldFeatureExtractor(config)
    
    print(f"\nFeature Extractor Configuration:")
    print(f"  Layers: {extractor.layers}")
    print(f"  Layer Fusion: {args.layer_fusion}")
    
    # Benchmark feature extractor
    if not args.skip_extractor_benchmark:
        print(f"\nBenchmarking Feature Extractor...")
        input_size = (args.batch_size, args.channels, args.input_size, args.input_size)
        extractor_results = benchmark_feature_extractor(
            extractor, input_size, device, args.num_warmup, args.num_iterations
        )
        print_benchmark_results(extractor_results, "Feature Extractor Benchmark Results")
    
    # Benchmark GoldDescriptor
    if not args.skip_descriptor_benchmark:
        print(f"\nBenchmarking GoldDescriptor...")
        
        # Create dummy dataset
        dataset = DummyDataset(
            size=args.batch_size,
            channels=args.channels,
            height=args.input_size,
            width=args.input_size,
            num_samples=args.num_samples,
        )
        
        # Configure vectorizer if requested
        vectorizer = None
        if args.vectorize:
            vectorizer = TensorVectorizer(
                keep=Filter2DWithCount(
                    filter_count=args.vector_count,
                    filter_location=FilterLocation.START,
                    keep=True,
                )
            )
            print(f"\nVectorizer Configuration:")
            print(f"  Vector Count: {args.vector_count}")
        
        # Create temporary directory for benchmark
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            table_path = os.path.join(tmpdir, "benchmark_table")
            
            descriptor = GoldDescriptor(
                table_path=table_path,
                extractor=extractor,
                vectorizer=vectorizer,
                collate_fn=collate_fn,
                min_pxt_insert_size=args.min_pxt_insert_size,
                batch_size=args.batch_size,
                device=device,
                max_batches=args.num_samples // args.batch_size,
            )
            
            descriptor_results = benchmark_gold_descriptor(
                descriptor, dataset, device, args.num_samples
            )
            print_benchmark_results(descriptor_results, "GoldDescriptor Benchmark Results")
    
    # Recommend optimal settings
    if not args.skip_extractor_benchmark or not args.skip_descriptor_benchmark:
        # Get available memory
        available_ram_mb = 0.0
        if PSUTIL_AVAILABLE:
            available_ram_mb = psutil.virtual_memory().available / 1024 / 1024
        
        available_vram_mb = 0.0
        if PYNVML_AVAILABLE and torch.cuda.is_available():
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                available_vram_mb = info.free / 1024 / 1024
            except Exception:
                pass
        
        # Use the most relevant benchmark results
        if not args.skip_descriptor_benchmark:
            results = descriptor_results
        else:
            results = extractor_results
        
        recommendations = recommend_optimal_settings(
            available_ram_mb, available_vram_mb, results, args.batch_size
        )


if __name__ == "__main__":
    main()
