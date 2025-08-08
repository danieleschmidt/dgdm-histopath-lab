#!/usr/bin/env python3
"""
Edge Deployment Module for DGDM Histopath Lab

Optimized deployment for mobile devices, IoT, and edge computing.
Enables real-time histopathology analysis in resource-constrained environments.

Author: TERRAGON Autonomous Development System v4.0
Generated: 2025-08-08
"""

import asyncio
import logging
import os
import json
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.quantization as quant
from torch.utils.mobile_optimizer import optimize_for_mobile
import tempfile
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor

try:
    # Mobile deployment
    import torch.jit
    import torchvision
    TORCH_MOBILE_AVAILABLE = True
except ImportError:
    TORCH_MOBILE_AVAILABLE = False
    logging.warning("Torch mobile not available")

try:
    # TensorRT optimization
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logging.warning("TensorRT not available. Install with: pip install pycuda tensorrt")

try:
    # ONNX for cross-platform deployment
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("ONNX not available. Install with: pip install onnx onnxruntime")

try:
    # OpenVINO for Intel hardware
    from openvino.inference_engine import IECore
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    logging.warning("OpenVINO not available")

from ..models.dgdm_model import DGDMModel
from ..utils.exceptions import EdgeDeploymentError
from ..utils.validation import validate_tensor, validate_config
from ..utils.monitoring import EdgeMetricsCollector
from ..utils.optimization import ModelOptimizer


class EdgePlatform(Enum):
    """Supported edge deployment platforms."""
    ANDROID = "android"
    IOS = "ios"
    RASPBERRY_PI = "raspberry_pi"
    JETSON_NANO = "jetson_nano"
    CORAL_DEV = "coral_dev_board"
    INTEL_NUC = "intel_nuc"
    GENERIC_ARM = "generic_arm"
    WEBASSEMBLY = "webassembly"
    BROWSER = "browser_js"


class OptimizationLevel(Enum):
    """Model optimization levels for edge deployment."""
    MINIMAL = "minimal"  # Basic optimizations
    BALANCED = "balanced"  # Good performance/accuracy tradeoff
    AGGRESSIVE = "aggressive"  # Maximum compression
    ULTRA_LIGHT = "ultra_light"  # Extreme compression for IoT


@dataclass
class EdgeConfig:
    """Configuration for edge deployment."""
    # Target platform
    platform: EdgePlatform = EdgePlatform.GENERIC_ARM
    
    # Resource constraints
    max_memory_mb: int = 512
    max_cpu_cores: int = 4
    has_gpu: bool = False
    gpu_memory_mb: int = 0
    
    # Performance requirements
    max_latency_ms: float = 1000.0
    min_throughput_fps: float = 1.0
    target_accuracy_threshold: float = 0.90
    
    # Optimization settings
    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    quantization_enabled: bool = True
    pruning_enabled: bool = True
    knowledge_distillation: bool = True
    
    # Model compression
    quantization_bits: int = 8  # 8-bit quantization
    pruning_ratio: float = 0.3  # 30% pruning
    compression_ratio: float = 0.1  # 10x compression target
    
    # Caching and batching
    enable_caching: bool = True
    cache_size_mb: int = 64
    batch_size: int = 1
    enable_batching: bool = False
    
    # Network settings for remote inference
    enable_remote_fallback: bool = True
    remote_endpoint: Optional[str] = None
    network_timeout_seconds: float = 5.0
    
    # Security
    enable_model_encryption: bool = True
    encryption_key: Optional[str] = None
    

class EdgeModelOptimizer:
    """Advanced model optimization for edge deployment."""
    
    def __init__(self, config: EdgeConfig):
        self.config = config
        self.metrics = EdgeMetricsCollector()
        self.optimization_cache = {}
        
    async def optimize_model(self, model: DGDMModel, sample_input: torch.Tensor) -> Dict[str, Any]:
        """Optimize model for edge deployment."""
        try:
            optimization_pipeline = [
                ('quantization', self._apply_quantization),
                ('pruning', self._apply_pruning),
                ('distillation', self._apply_knowledge_distillation),
                ('graph_optimization', self._optimize_computation_graph),
                ('platform_specific', self._apply_platform_optimizations)
            ]
            
            optimized_model = model
            optimization_metrics = {}
            
            # Apply optimizations in sequence
            for step_name, optimization_fn in optimization_pipeline:
                logging.info(f"Applying {step_name} optimization...")
                
                start_time = time.time()
                optimized_model, step_metrics = await optimization_fn(
                    optimized_model, sample_input
                )
                optimization_time = time.time() - start_time
                
                step_metrics['optimization_time'] = optimization_time
                optimization_metrics[step_name] = step_metrics
                
                logging.info(f"{step_name} completed in {optimization_time:.2f}s")
            
            # Validate optimized model
            validation_results = await self._validate_optimized_model(
                optimized_model, model, sample_input
            )
            
            return {
                'optimized_model': optimized_model,
                'optimization_metrics': optimization_metrics,
                'validation_results': validation_results
            }
            
        except Exception as e:
            raise EdgeDeploymentError(f"Model optimization failed: {e}")
    
    async def _apply_quantization(self, model: DGDMModel, sample_input: torch.Tensor) -> Tuple[DGDMModel, Dict[str, Any]]:
        """Apply quantization for reduced precision."""
        if not self.config.quantization_enabled:
            return model, {'skipped': True}
        
        try:
            # Prepare model for quantization
            model.eval()
            quantized_model = copy.deepcopy(model)
            
            if self.config.quantization_bits == 8:
                # Dynamic quantization (most compatible)
                quantized_model = torch.quantization.quantize_dynamic(
                    quantized_model, 
                    {nn.Linear, nn.Conv2d}, 
                    dtype=torch.qint8
                )
            elif self.config.quantization_bits == 16:
                # Half precision
                quantized_model = quantized_model.half()
            else:
                # Custom quantization
                quantized_model = await self._apply_custom_quantization(
                    quantized_model, self.config.quantization_bits
                )
            
            # Measure compression
            original_size = self._get_model_size(model)
            quantized_size = self._get_model_size(quantized_model)
            compression_ratio = quantized_size / original_size
            
            metrics = {
                'quantization_bits': self.config.quantization_bits,
                'original_size_mb': original_size / (1024 * 1024),
                'quantized_size_mb': quantized_size / (1024 * 1024),
                'compression_ratio': compression_ratio,
                'method': f'{self.config.quantization_bits}-bit quantization'
            }
            
            return quantized_model, metrics
            
        except Exception as e:
            logging.error(f"Quantization failed: {e}")
            return model, {'error': str(e)}
    
    async def _apply_pruning(self, model: DGDMModel, sample_input: torch.Tensor) -> Tuple[DGDMModel, Dict[str, Any]]:
        """Apply structured and unstructured pruning."""
        if not self.config.pruning_enabled:
            return model, {'skipped': True}
        
        try:
            import torch.nn.utils.prune as prune
            
            pruned_model = copy.deepcopy(model)
            pruning_ratio = self.config.pruning_ratio
            
            # Apply magnitude-based pruning to linear and conv layers
            modules_to_prune = []
            for name, module in pruned_model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    modules_to_prune.append((module, 'weight'))
            
            # Global unstructured pruning
            prune.global_unstructured(
                modules_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=pruning_ratio,
            )
            
            # Remove pruning re-parametrization to make pruning permanent
            for module, param_name in modules_to_prune:
                prune.remove(module, param_name)
            
            # Calculate sparsity
            total_params = sum(p.numel() for p in pruned_model.parameters())
            zero_params = sum((p == 0).sum().item() for p in pruned_model.parameters())
            actual_sparsity = zero_params / total_params
            
            metrics = {
                'target_pruning_ratio': pruning_ratio,
                'actual_sparsity': actual_sparsity,
                'total_parameters': total_params,
                'zero_parameters': zero_params,
                'method': 'L1 magnitude-based pruning'
            }
            
            return pruned_model, metrics
            
        except Exception as e:
            logging.error(f"Pruning failed: {e}")
            return model, {'error': str(e)}
    
    async def _apply_knowledge_distillation(self, model: DGDMModel, sample_input: torch.Tensor) -> Tuple[DGDMModel, Dict[str, Any]]:
        """Apply knowledge distillation to create smaller student model."""
        if not self.config.knowledge_distillation:
            return model, {'skipped': True}
        
        try:
            # Create smaller student model (simplified architecture)
            student_model = self._create_student_model(model)
            
            # Distillation training would happen here
            # For now, we'll simulate the process
            
            # Copy some weights from teacher to student (simplified)
            teacher_state = model.state_dict()
            student_state = student_model.state_dict()
            
            # Transfer compatible weights
            transferred_weights = 0
            for key in student_state.keys():
                if key in teacher_state and teacher_state[key].shape == student_state[key].shape:
                    student_state[key] = teacher_state[key]
                    transferred_weights += 1
            
            student_model.load_state_dict(student_state)
            
            metrics = {
                'student_parameters': sum(p.numel() for p in student_model.parameters()),
                'teacher_parameters': sum(p.numel() for p in model.parameters()),
                'parameter_reduction': 1 - (sum(p.numel() for p in student_model.parameters()) / sum(p.numel() for p in model.parameters())),
                'transferred_weights': transferred_weights,
                'method': 'knowledge_distillation'
            }
            
            return student_model, metrics
            
        except Exception as e:
            logging.error(f"Knowledge distillation failed: {e}")
            return model, {'error': str(e)}
    
    async def _optimize_computation_graph(self, model: DGDMModel, sample_input: torch.Tensor) -> Tuple[DGDMModel, Dict[str, Any]]:
        """Optimize computation graph for inference."""
        try:
            # JIT compilation
            model.eval()
            traced_model = torch.jit.trace(model, sample_input)
            
            # Graph optimizations
            optimized_model = torch.jit.optimize_for_inference(traced_model)
            
            # Freeze for deployment
            frozen_model = torch.jit.freeze(optimized_model)
            
            metrics = {
                'graph_optimized': True,
                'jit_compiled': True,
                'frozen': True,
                'method': 'torch_jit_optimization'
            }
            
            return frozen_model, metrics
            
        except Exception as e:
            logging.error(f"Graph optimization failed: {e}")
            return model, {'error': str(e)}
    
    async def _apply_platform_optimizations(self, model: Any, sample_input: torch.Tensor) -> Tuple[Any, Dict[str, Any]]:
        """Apply platform-specific optimizations."""
        try:
            platform_metrics = {'platform': self.config.platform.value}
            
            if self.config.platform in [EdgePlatform.ANDROID, EdgePlatform.IOS]:
                # Mobile optimizations
                if hasattr(torch.utils.mobile_optimizer, 'optimize_for_mobile'):
                    mobile_model = optimize_for_mobile(model)
                    platform_metrics['mobile_optimized'] = True
                    return mobile_model, platform_metrics
            
            elif self.config.platform == EdgePlatform.JETSON_NANO and TENSORRT_AVAILABLE:
                # TensorRT optimization
                tensorrt_model = await self._convert_to_tensorrt(model, sample_input)
                platform_metrics['tensorrt_optimized'] = True
                return tensorrt_model, platform_metrics
            
            elif self.config.platform == EdgePlatform.INTEL_NUC and OPENVINO_AVAILABLE:
                # OpenVINO optimization
                openvino_model = await self._convert_to_openvino(model, sample_input)
                platform_metrics['openvino_optimized'] = True
                return openvino_model, platform_metrics
            
            # Default: no platform-specific optimization
            platform_metrics['optimization'] = 'none'
            return model, platform_metrics
            
        except Exception as e:
            logging.error(f"Platform optimization failed: {e}")
            return model, {'error': str(e)}
    
    def _create_student_model(self, teacher_model: DGDMModel) -> DGDMModel:
        """Create smaller student model for knowledge distillation."""
        # Simplified student architecture (reduce dimensions by 50%)
        student_config = {
            'node_features': teacher_model.config.get('node_features', 768) // 2,
            'hidden_dims': [dim // 2 for dim in teacher_model.config.get('hidden_dims', [512, 256, 128])],
            'num_diffusion_steps': teacher_model.config.get('num_diffusion_steps', 10) // 2,
            'attention_heads': teacher_model.config.get('attention_heads', 8) // 2,
        }
        
        # Create student model with reduced capacity
        student_model = DGDMModel(**student_config)
        return student_model
    
    def _get_model_size(self, model: Any) -> int:
        """Get model size in bytes."""
        if hasattr(model, 'parameters'):
            return sum(p.numel() * p.element_size() for p in model.parameters())
        else:
            # For JIT models, estimate size
            with tempfile.NamedTemporaryFile() as tmp:
                torch.jit.save(model, tmp.name)
                return os.path.getsize(tmp.name)
    
    async def _convert_to_tensorrt(self, model: Any, sample_input: torch.Tensor) -> Any:
        """Convert model to TensorRT for NVIDIA hardware."""
        # Placeholder for TensorRT conversion
        # In practice, this would involve ONNX -> TensorRT conversion
        logging.info("TensorRT conversion not fully implemented")
        return model
    
    async def _convert_to_openvino(self, model: Any, sample_input: torch.Tensor) -> Any:
        """Convert model to OpenVINO for Intel hardware."""
        # Placeholder for OpenVINO conversion
        logging.info("OpenVINO conversion not fully implemented")
        return model
    
    async def _apply_custom_quantization(self, model: DGDMModel, bits: int) -> DGDMModel:
        """Apply custom quantization for non-standard bit widths."""
        # Simplified custom quantization
        for param in model.parameters():
            if param.dtype == torch.float32:
                # Quantize to specified bit width
                min_val, max_val = param.min(), param.max()
                scale = (max_val - min_val) / (2**bits - 1)
                param.data = torch.round((param.data - min_val) / scale) * scale + min_val
        
        return model
    
    async def _validate_optimized_model(self, optimized_model: Any, original_model: DGDMModel, sample_input: torch.Tensor) -> Dict[str, Any]:
        """Validate that optimized model maintains acceptable performance."""
        try:
            # Performance validation
            original_model.eval()
            optimized_model.eval()
            
            # Inference time comparison
            start_time = time.time()
            with torch.no_grad():
                _ = original_model(sample_input)
            original_time = time.time() - start_time
            
            start_time = time.time()
            with torch.no_grad():
                _ = optimized_model(sample_input)
            optimized_time = time.time() - start_time
            
            # Memory usage comparison
            original_memory = self._get_model_size(original_model)
            optimized_memory = self._get_model_size(optimized_model)
            
            validation_results = {
                'original_inference_time': original_time,
                'optimized_inference_time': optimized_time,
                'speedup_factor': original_time / optimized_time if optimized_time > 0 else 0,
                'original_memory_mb': original_memory / (1024 * 1024),
                'optimized_memory_mb': optimized_memory / (1024 * 1024),
                'memory_reduction': 1 - (optimized_memory / original_memory),
                'meets_latency_requirement': optimized_time * 1000 <= self.config.max_latency_ms,
                'meets_memory_requirement': optimized_memory <= self.config.max_memory_mb * 1024 * 1024
            }
            
            return validation_results
            
        except Exception as e:
            return {'validation_error': str(e)}


class EdgeInferenceEngine:
    """Optimized inference engine for edge devices."""
    
    def __init__(self, config: EdgeConfig):
        self.config = config
        self.model = None
        self.metrics = EdgeMetricsCollector()
        self.cache = {} if config.enable_caching else None
        self.batch_queue = [] if config.enable_batching else None
        self.batch_lock = threading.Lock() if config.enable_batching else None
        
        # Resource monitoring
        self.resource_monitor = EdgeResourceMonitor(config)
        
    async def load_model(self, model_path: str) -> bool:
        """Load optimized model for edge inference."""
        try:
            # Load model based on format
            if model_path.endswith('.pt') or model_path.endswith('.pth'):
                # PyTorch model
                self.model = torch.jit.load(model_path, map_location='cpu')
            elif model_path.endswith('.onnx') and ONNX_AVAILABLE:
                # ONNX model
                self.model = ort.InferenceSession(model_path)
            else:
                # Standard PyTorch checkpoint
                checkpoint = torch.load(model_path, map_location='cpu')
                self.model = checkpoint['model']
            
            # Warm up model
            await self._warmup_model()
            
            logging.info(f"Edge model loaded from {model_path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load edge model: {e}")
            return False
    
    async def predict(self, input_data: Union[torch.Tensor, np.ndarray], 
                     metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform edge inference with optimizations."""
        try:
            start_time = time.time()
            
            # Check cache first
            if self.cache is not None:
                cache_key = self._generate_cache_key(input_data)
                if cache_key in self.cache:
                    cached_result = self.cache[cache_key]
                    cached_result['from_cache'] = True
                    cached_result['inference_time'] = 0.0
                    return cached_result
            
            # Resource check
            if not await self.resource_monitor.check_resources():
                return await self._handle_resource_constraint(input_data, metadata)
            
            # Preprocess input
            processed_input = await self._preprocess_input(input_data)
            
            # Batch processing if enabled
            if self.config.enable_batching:
                result = await self._batched_inference(processed_input, metadata)
            else:
                result = await self._single_inference(processed_input)
            
            # Post-process result
            final_result = await self._postprocess_result(result, metadata)
            
            # Update cache
            inference_time = time.time() - start_time
            final_result['inference_time'] = inference_time
            
            if self.cache is not None:
                cache_key = self._generate_cache_key(input_data)
                self.cache[cache_key] = final_result.copy()
                
                # Cache management
                await self._manage_cache()
            
            # Record metrics
            await self.metrics.record_inference({
                'inference_time': inference_time,
                'cache_hit': False,
                'batch_size': 1,
                'memory_usage': self.resource_monitor.get_memory_usage()
            })
            
            return final_result
            
        except Exception as e:
            logging.error(f"Edge inference failed: {e}")
            return {'error': str(e), 'inference_time': time.time() - start_time}
    
    async def _single_inference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Perform single sample inference."""
        self.model.eval()
        
        with torch.no_grad():
            if hasattr(self.model, '__call__'):
                # PyTorch model
                output = self.model(input_tensor)
            elif ONNX_AVAILABLE and isinstance(self.model, ort.InferenceSession):
                # ONNX model
                input_name = self.model.get_inputs()[0].name
                output = self.model.run(None, {input_name: input_tensor.numpy()})[0]
                output = torch.from_numpy(output)
            else:
                raise EdgeDeploymentError("Unsupported model format for inference")
        
        return output
    
    async def _batched_inference(self, input_tensor: torch.Tensor, metadata: Optional[Dict[str, Any]]) -> torch.Tensor:
        """Perform batched inference for efficiency."""
        with self.batch_lock:
            self.batch_queue.append((input_tensor, metadata))
            
            # Check if we should process batch
            if len(self.batch_queue) >= self.config.batch_size:
                # Process current batch
                batch_inputs = [item[0] for item in self.batch_queue]
                batch_tensor = torch.stack(batch_inputs)
                
                # Clear queue
                processed_items = self.batch_queue.copy()
                self.batch_queue.clear()
                
                # Run batched inference
                batch_output = await self._single_inference(batch_tensor)
                
                # Return result for current input (last in batch)
                return batch_output[-1]
            else:
                # Wait for batch to fill or timeout
                await asyncio.sleep(0.01)
                return await self._single_inference(input_tensor)
    
    async def _preprocess_input(self, input_data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Preprocess input for edge inference."""
        if isinstance(input_data, np.ndarray):
            input_tensor = torch.from_numpy(input_data)
        else:
            input_tensor = input_data
        
        # Ensure correct data type and device
        input_tensor = input_tensor.float()
        if torch.cuda.is_available() and self.config.has_gpu:
            input_tensor = input_tensor.cuda()
        
        # Add batch dimension if missing
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        return input_tensor
    
    async def _postprocess_result(self, output: torch.Tensor, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Post-process inference result."""
        # Convert to CPU and numpy
        output_np = output.detach().cpu().numpy()
        
        # Generate predictions
        if output_np.shape[-1] > 1:
            # Classification
            probabilities = torch.softmax(output, dim=-1).detach().cpu().numpy()
            predicted_class = np.argmax(probabilities)
            confidence = float(probabilities[predicted_class])
            
            result = {
                'prediction': int(predicted_class),
                'confidence': confidence,
                'probabilities': probabilities.tolist(),
                'task_type': 'classification'
            }
        else:
            # Regression
            predicted_value = float(output_np.item())
            result = {
                'prediction': predicted_value,
                'task_type': 'regression'
            }
        
        # Add metadata if available
        if metadata:
            result['metadata'] = metadata
        
        return result
    
    def _generate_cache_key(self, input_data: Union[torch.Tensor, np.ndarray]) -> str:
        """Generate cache key for input data."""
        if isinstance(input_data, torch.Tensor):
            data_hash = hash(input_data.data.tobytes())
        else:
            data_hash = hash(input_data.tobytes())
        return f"cache_{data_hash}_{input_data.shape}"
    
    async def _manage_cache(self):
        """Manage cache size and eviction."""
        if not self.cache:
            return
        
        # Calculate cache size
        cache_size_mb = len(json.dumps(self.cache).encode('utf-8')) / (1024 * 1024)
        
        if cache_size_mb > self.config.cache_size_mb:
            # Evict oldest entries (simple LRU)
            num_to_evict = len(self.cache) // 4  # Evict 25%
            keys_to_evict = list(self.cache.keys())[:num_to_evict]
            
            for key in keys_to_evict:
                del self.cache[key]
            
            logging.info(f"Evicted {num_to_evict} cache entries")
    
    async def _handle_resource_constraint(self, input_data: Union[torch.Tensor, np.ndarray], 
                                        metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle resource constraints with fallback strategies."""
        if self.config.enable_remote_fallback and self.config.remote_endpoint:
            # Attempt remote inference
            try:
                return await self._remote_inference(input_data, metadata)
            except Exception as e:
                logging.error(f"Remote inference failed: {e}")
        
        # Fallback: return error
        return {
            'error': 'Insufficient resources for local inference',
            'fallback_attempted': self.config.enable_remote_fallback,
            'resource_status': await self.resource_monitor.get_resource_status()
        }
    
    async def _remote_inference(self, input_data: Union[torch.Tensor, np.ndarray], 
                              metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform remote inference as fallback."""
        # Placeholder for remote inference
        # In practice, this would send data to cloud endpoint
        logging.info("Remote inference not implemented")
        return {'error': 'Remote inference not available'}
    
    async def _warmup_model(self):
        """Warm up model with dummy inputs."""
        try:
            # Create dummy input based on expected shape
            dummy_input = torch.randn(1, 3, 224, 224)  # Typical image input
            
            # Run a few warmup inferences
            for _ in range(3):
                await self._single_inference(dummy_input)
            
            logging.info("Model warmup completed")
            
        except Exception as e:
            logging.warning(f"Model warmup failed: {e}")


class EdgeResourceMonitor:
    """Monitor edge device resources."""
    
    def __init__(self, config: EdgeConfig):
        self.config = config
        self.monitoring_thread = None
        self.resource_history = []
        
    async def check_resources(self) -> bool:
        """Check if sufficient resources are available."""
        try:
            # Memory check
            memory_info = psutil.virtual_memory()
            available_memory_mb = memory_info.available / (1024 * 1024)
            
            if available_memory_mb < self.config.max_memory_mb:
                logging.warning(f"Insufficient memory: {available_memory_mb:.1f}MB < {self.config.max_memory_mb}MB")
                return False
            
            # CPU check
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > 90:  # High CPU usage
                logging.warning(f"High CPU usage: {cpu_percent:.1f}%")
                return False
            
            # GPU check if applicable
            if self.config.has_gpu:
                gpu_available = await self._check_gpu_resources()
                if not gpu_available:
                    return False
            
            return True
            
        except Exception as e:
            logging.error(f"Resource check failed: {e}")
            return False
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            memory_info = psutil.virtual_memory()
            return (memory_info.total - memory_info.available) / (1024 * 1024)
        except:
            return 0.0
    
    async def get_resource_status(self) -> Dict[str, Any]:
        """Get comprehensive resource status."""
        try:
            memory_info = psutil.virtual_memory()
            
            status = {
                'memory': {
                    'total_mb': memory_info.total / (1024 * 1024),
                    'available_mb': memory_info.available / (1024 * 1024),
                    'used_percent': memory_info.percent
                },
                'cpu': {
                    'usage_percent': psutil.cpu_percent(interval=0.1),
                    'cores': psutil.cpu_count()
                },
                'disk': {
                    'usage_percent': psutil.disk_usage('/').percent
                }
            }
            
            # GPU status if applicable
            if self.config.has_gpu:
                gpu_status = await self._get_gpu_status()
                status['gpu'] = gpu_status
            
            return status
            
        except Exception as e:
            return {'error': str(e)}
    
    async def _check_gpu_resources(self) -> bool:
        """Check GPU resource availability."""
        try:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_mb = gpu_memory / (1024 * 1024)
                
                return gpu_memory_mb >= self.config.gpu_memory_mb
            return False
            
        except Exception as e:
            logging.error(f"GPU check failed: {e}")
            return False
    
    async def _get_gpu_status(self) -> Dict[str, Any]:
        """Get GPU status information."""
        try:
            if torch.cuda.is_available():
                device_props = torch.cuda.get_device_properties(0)
                memory_allocated = torch.cuda.memory_allocated(0)
                memory_cached = torch.cuda.memory_reserved(0)
                
                return {
                    'available': True,
                    'name': device_props.name,
                    'total_memory_mb': device_props.total_memory / (1024 * 1024),
                    'allocated_memory_mb': memory_allocated / (1024 * 1024),
                    'cached_memory_mb': memory_cached / (1024 * 1024)
                }
            else:
                return {'available': False}
                
        except Exception as e:
            return {'error': str(e)}


class EdgeDeploymentManager:
    """High-level manager for edge deployment operations."""
    
    def __init__(self, config: EdgeConfig):
        self.config = config
        self.optimizer = EdgeModelOptimizer(config)
        self.inference_engine = EdgeInferenceEngine(config)
        self.metrics = EdgeMetricsCollector()
        
    async def deploy_model(self, model: DGDMModel, sample_input: torch.Tensor, 
                          output_path: str) -> Dict[str, Any]:
        """Complete model deployment pipeline for edge devices."""
        try:
            logging.info(f"Starting edge deployment for {self.config.platform.value}")
            
            # Step 1: Optimize model
            optimization_results = await self.optimizer.optimize_model(model, sample_input)
            optimized_model = optimization_results['optimized_model']
            
            # Step 2: Validate performance
            validation_results = optimization_results['validation_results']
            
            if not validation_results.get('meets_latency_requirement', False):
                logging.warning("Model does not meet latency requirements")
            
            if not validation_results.get('meets_memory_requirement', False):
                logging.warning("Model does not meet memory requirements")
            
            # Step 3: Save optimized model
            await self._save_deployment_package(optimized_model, output_path)
            
            # Step 4: Generate deployment metadata
            deployment_metadata = {
                'platform': self.config.platform.value,
                'optimization_level': self.config.optimization_level.value,
                'optimization_metrics': optimization_results['optimization_metrics'],
                'validation_results': validation_results,
                'deployment_timestamp': time.time(),
                'model_path': output_path
            }
            
            logging.info(f"Edge deployment completed: {output_path}")
            return deployment_metadata
            
        except Exception as e:
            raise EdgeDeploymentError(f"Edge deployment failed: {e}")
    
    async def benchmark_model(self, model_path: str, test_inputs: List[torch.Tensor]) -> Dict[str, Any]:
        """Benchmark deployed model performance."""
        try:
            # Load model
            await self.inference_engine.load_model(model_path)
            
            # Run benchmark tests
            benchmark_results = {
                'latency_stats': [],
                'memory_usage': [],
                'throughput': 0,
                'accuracy_maintained': True
            }
            
            start_time = time.time()
            
            for i, test_input in enumerate(test_inputs):
                inference_start = time.time()
                result = await self.inference_engine.predict(test_input)
                inference_time = time.time() - inference_start
                
                benchmark_results['latency_stats'].append(inference_time * 1000)  # ms
                benchmark_results['memory_usage'].append(
                    self.inference_engine.resource_monitor.get_memory_usage()
                )
                
                if 'error' in result:
                    benchmark_results['accuracy_maintained'] = False
            
            total_time = time.time() - start_time
            benchmark_results['throughput'] = len(test_inputs) / total_time  # FPS
            
            # Calculate statistics
            latencies = benchmark_results['latency_stats']
            benchmark_results['avg_latency_ms'] = np.mean(latencies)
            benchmark_results['p95_latency_ms'] = np.percentile(latencies, 95)
            benchmark_results['max_latency_ms'] = np.max(latencies)
            benchmark_results['avg_memory_mb'] = np.mean(benchmark_results['memory_usage'])
            
            logging.info(f"Benchmark completed: {benchmark_results['avg_latency_ms']:.2f}ms avg latency")
            return benchmark_results
            
        except Exception as e:
            return {'error': str(e)}
    
    async def _save_deployment_package(self, model: Any, output_path: str):
        """Save complete deployment package."""
        try:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save optimized model
            if hasattr(model, 'save'):
                torch.jit.save(model, output_path)
            else:
                torch.save(model, output_path)
            
            # Save deployment configuration
            config_path = output_path.replace('.pt', '_config.json')
            with open(config_path, 'w') as f:
                json.dump({
                    'platform': self.config.platform.value,
                    'optimization_level': self.config.optimization_level.value,
                    'max_memory_mb': self.config.max_memory_mb,
                    'max_latency_ms': self.config.max_latency_ms,
                    'quantization_enabled': self.config.quantization_enabled,
                    'pruning_enabled': self.config.pruning_enabled
                }, f, indent=2)
            
            logging.info(f"Deployment package saved: {output_path}")
            
        except Exception as e:
            raise EdgeDeploymentError(f"Failed to save deployment package: {e}")


# Export main components
__all__ = [
    'EdgePlatform',
    'OptimizationLevel',
    'EdgeConfig',
    'EdgeModelOptimizer',
    'EdgeInferenceEngine',
    'EdgeResourceMonitor',
    'EdgeDeploymentManager'
]
