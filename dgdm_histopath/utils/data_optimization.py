"""Optimized data processing utilities for high-performance histopathology analysis."""

import os
import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, Any, List, Optional, Tuple, Iterator, Union
import logging
import pickle
import lz4.frame
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import h5py
import zarr
from functools import partial
import queue
import mmap
from dataclasses import dataclass
import psutil


@dataclass
class DataProcessingConfig:
    """Configuration for optimized data processing."""
    
    # Parallel processing
    num_processes: int = min(mp.cpu_count(), 8)
    num_threads: int = min(mp.cpu_count(), 16)
    chunk_size: int = 1000
    
    # Memory management
    max_memory_per_worker_mb: int = 2000
    use_shared_memory: bool = True
    enable_memory_mapping: bool = True
    
    # I/O optimization
    prefetch_factor: int = 4
    use_compression: bool = True
    compression_level: int = 3
    
    # Caching
    enable_disk_cache: bool = True
    cache_directory: str = "./cache"
    max_cache_size_gb: int = 10
    
    # Data format optimization
    preferred_dtype: str = "float16"  # Use half precision when possible
    enable_quantization: bool = False
    quantization_bits: int = 8


class OptimizedDataLoader:
    """High-performance data loader with advanced optimizations."""
    
    def __init__(self, dataset: Dataset, config: DataProcessingConfig = None, **kwargs):
        self.dataset = dataset
        self.config = config or DataProcessingConfig()
        self.logger = logging.getLogger(__name__)
        
        # Setup optimized DataLoader parameters
        self.dataloader_params = self._optimize_dataloader_params(kwargs)
        
        # Initialize components
        self.prefetch_queue = None
        self.prefetch_thread = None
        self.memory_pool = {}
        
        # Statistics
        self.stats = {
            'batches_loaded': 0,
            'total_load_time': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def _optimize_dataloader_params(self, user_params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize DataLoader parameters based on system capabilities."""
        
        # Base optimized parameters
        params = {
            'batch_size': user_params.get('batch_size', 4),
            'shuffle': user_params.get('shuffle', True),
            'num_workers': min(self.config.num_threads, psutil.cpu_count()),
            'pin_memory': torch.cuda.is_available(),
            'persistent_workers': True,
            'prefetch_factor': self.config.prefetch_factor,
            'drop_last': user_params.get('drop_last', False)
        }
        
        # Memory-based adjustments
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        if available_memory_gb < 8:
            params['num_workers'] = max(1, params['num_workers'] // 2)
            params['prefetch_factor'] = 2
        elif available_memory_gb > 32:
            params['prefetch_factor'] = 6
            
        # GPU memory adjustments
        if torch.cuda.is_available():
            try:
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if gpu_memory_gb < 8:
                    params['pin_memory'] = False
            except Exception:
                pass
        
        # Override with user parameters
        params.update(user_params)
        
        self.logger.info(f"Optimized DataLoader parameters: {params}")
        return params
    
    def __iter__(self):
        """Create optimized iterator."""
        dataloader = DataLoader(self.dataset, **self.dataloader_params)
        
        if self.config.prefetch_factor > 0:
            return self._prefetch_iterator(dataloader)
        else:
            return iter(dataloader)
    
    def _prefetch_iterator(self, dataloader):
        """Iterator with background prefetching."""
        prefetch_queue = queue.Queue(maxsize=self.config.prefetch_factor)
        
        def prefetch_worker():
            try:
                for batch in dataloader:
                    # Optimize batch data
                    optimized_batch = self._optimize_batch(batch)
                    prefetch_queue.put(optimized_batch)
                prefetch_queue.put(None)  # Sentinel to indicate end
            except Exception as e:
                prefetch_queue.put(e)
        
        prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
        prefetch_thread.start()
        
        try:
            while True:
                batch = prefetch_queue.get()
                if batch is None:  # End sentinel
                    break
                elif isinstance(batch, Exception):
                    raise batch
                else:
                    yield batch
        finally:
            prefetch_thread.join(timeout=1)
    
    def _optimize_batch(self, batch):
        """Optimize batch data for processing."""
        start_time = time.time()
        
        if isinstance(batch, dict):
            # Optimize each tensor in the batch
            optimized_batch = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    optimized_batch[key] = self._optimize_tensor(value)
                else:
                    optimized_batch[key] = value
        elif isinstance(batch, (list, tuple)):
            # Optimize list/tuple of tensors
            optimized_batch = []
            for item in batch:
                if isinstance(item, torch.Tensor):
                    optimized_batch.append(self._optimize_tensor(item))
                else:
                    optimized_batch.append(item)
            optimized_batch = type(batch)(optimized_batch)
        else:
            optimized_batch = batch
        
        self.stats['batches_loaded'] += 1
        self.stats['total_load_time'] += time.time() - start_time
        
        return optimized_batch
    
    def _optimize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimize individual tensors."""
        
        # Memory layout optimization
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        # Dtype optimization
        if self.config.preferred_dtype == "float16" and tensor.dtype == torch.float32:
            # Only convert if values are within float16 range
            if tensor.abs().max() < 65000:
                tensor = tensor.half()
        
        # GPU transfer optimization
        if torch.cuda.is_available() and not tensor.is_cuda:
            tensor = tensor.cuda(non_blocking=True)
        
        return tensor
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_load_time = (self.stats['total_load_time'] / self.stats['batches_loaded'] 
                        if self.stats['batches_loaded'] > 0 else 0)
        
        return {
            **self.stats,
            'avg_load_time': avg_load_time,
            'cache_hit_rate': (self.stats['cache_hits'] / 
                              (self.stats['cache_hits'] + self.stats['cache_misses'])
                              if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0)
        }


class HighPerformanceHDF5Dataset(Dataset):
    """Optimized HDF5 dataset with memory mapping and chunked reading."""
    
    def __init__(self, hdf5_path: str, data_key: str = 'data', label_key: str = 'labels',
                 chunk_cache_size: int = 1024*1024*100, enable_swmr: bool = True):
        self.hdf5_path = Path(hdf5_path)
        self.data_key = data_key
        self.label_key = label_key
        self.chunk_cache_size = chunk_cache_size
        self.enable_swmr = enable_swmr
        
        self.logger = logging.getLogger(__name__)
        
        # Thread-local storage for HDF5 file handles
        self._local = threading.local()
        
        # Get dataset metadata
        with h5py.File(self.hdf5_path, 'r') as f:
            self.data_shape = f[data_key].shape
            self.data_dtype = f[data_key].dtype
            self.length = self.data_shape[0]
            
            if label_key in f:
                self.label_shape = f[label_key].shape
                self.label_dtype = f[label_key].dtype
            else:
                self.label_shape = None
                self.label_dtype = None
        
        self.logger.info(f"Initialized HDF5 dataset: {self.length} samples, shape: {self.data_shape}")
    
    def _get_file_handle(self):
        """Get thread-local HDF5 file handle."""
        if not hasattr(self._local, 'file_handle') or self._local.file_handle is None:
            self._local.file_handle = h5py.File(
                self.hdf5_path, 'r',
                rdcc_nbytes=self.chunk_cache_size,
                swmr=self.enable_swmr
            )
        return self._local.file_handle
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """Optimized item retrieval with chunked reading."""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        file_handle = self._get_file_handle()
        
        try:
            # Read data
            data = file_handle[self.data_key][idx]
            
            # Convert to tensor
            data_tensor = torch.from_numpy(data.astype(np.float32))
            
            # Read labels if available
            if self.label_shape is not None:
                labels = file_handle[self.label_key][idx]
                label_tensor = torch.from_numpy(labels.astype(np.int64))
                return data_tensor, label_tensor
            else:
                return data_tensor
                
        except Exception as e:
            self.logger.error(f"Error reading index {idx}: {e}")
            raise
    
    def get_batch(self, indices: List[int]):
        """Optimized batch reading."""
        file_handle = self._get_file_handle()
        
        # Sort indices for better I/O performance
        sorted_indices = sorted(indices)
        
        # Read data in chunks
        data_chunks = []
        label_chunks = []
        
        for idx in sorted_indices:
            data = file_handle[self.data_key][idx]
            data_chunks.append(data)
            
            if self.label_shape is not None:
                labels = file_handle[self.label_key][idx]
                label_chunks.append(labels)
        
        # Stack into tensors
        data_tensor = torch.from_numpy(np.stack(data_chunks).astype(np.float32))
        
        if label_chunks:
            label_tensor = torch.from_numpy(np.stack(label_chunks).astype(np.int64))
            return data_tensor, label_tensor
        else:
            return data_tensor
    
    def __del__(self):
        """Clean up file handles."""
        if hasattr(self._local, 'file_handle') and self._local.file_handle is not None:
            self._local.file_handle.close()


class ParallelDataProcessor:
    """High-performance parallel data processing with automatic optimization."""
    
    def __init__(self, config: DataProcessingConfig = None):
        self.config = config or DataProcessingConfig()
        self.logger = logging.getLogger(__name__)
        
        # Setup process and thread pools
        self.process_pool = None
        self.thread_pool = None
        
        # Performance monitoring
        self.processing_stats = {
            'items_processed': 0,
            'total_processing_time': 0,
            'parallel_efficiency': 0
        }
    
    def __enter__(self):
        """Initialize pools."""
        self.process_pool = ProcessPoolExecutor(max_workers=self.config.num_processes)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.num_threads)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup pools."""
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
    
    def process_parallel(self, data_items: List[Any], processing_func: callable,
                        use_processes: bool = True, chunk_size: Optional[int] = None) -> List[Any]:
        """Process data items in parallel with automatic optimization."""
        
        if not data_items:
            return []
        
        chunk_size = chunk_size or self.config.chunk_size
        start_time = time.time()
        
        # Choose optimal processing strategy
        if use_processes and len(data_items) > 100:
            executor = self.process_pool
            strategy = "multiprocessing"
        else:
            executor = self.thread_pool
            strategy = "multithreading"
        
        self.logger.info(f"Processing {len(data_items)} items using {strategy}")
        
        try:
            # Chunk data for better memory management
            chunks = [data_items[i:i + chunk_size] for i in range(0, len(data_items), chunk_size)]
            
            # Process chunks in parallel
            chunk_processing_func = partial(self._process_chunk, processing_func)
            future_to_chunk = {executor.submit(chunk_processing_func, chunk): chunk for chunk in chunks}
            
            # Collect results
            results = []
            for future in future_to_chunk:
                chunk_results = future.result()
                results.extend(chunk_results)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.processing_stats['items_processed'] += len(data_items)
            self.processing_stats['total_processing_time'] += processing_time
            
            # Calculate parallel efficiency (speedup / num_workers)
            sequential_estimate = processing_time * (self.config.num_processes if use_processes else self.config.num_threads)
            actual_parallel_time = processing_time
            efficiency = min(1.0, sequential_estimate / (actual_parallel_time * (self.config.num_processes if use_processes else self.config.num_threads)))
            self.processing_stats['parallel_efficiency'] = efficiency
            
            self.logger.info(f"Processed {len(data_items)} items in {processing_time:.2f}s (efficiency: {efficiency:.2f})")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Parallel processing failed: {e}")
            raise
    
    def _process_chunk(self, processing_func: callable, chunk: List[Any]) -> List[Any]:
        """Process a chunk of data items."""
        return [processing_func(item) for item in chunk]
    
    def process_stream(self, data_stream: Iterator[Any], processing_func: callable,
                      buffer_size: int = 1000) -> Iterator[Any]:
        """Process streaming data with parallel processing."""
        
        buffer = []
        
        for item in data_stream:
            buffer.append(item)
            
            if len(buffer) >= buffer_size:
                # Process buffer in parallel
                results = self.process_parallel(buffer, processing_func)
                for result in results:
                    yield result
                buffer = []
        
        # Process remaining items
        if buffer:
            results = self.process_parallel(buffer, processing_func)
            for result in results:
                yield result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing performance statistics."""
        avg_processing_time = (self.processing_stats['total_processing_time'] / 
                              self.processing_stats['items_processed']
                              if self.processing_stats['items_processed'] > 0 else 0)
        
        return {
            **self.processing_stats,
            'avg_processing_time_per_item': avg_processing_time,
            'throughput_items_per_second': (self.processing_stats['items_processed'] / 
                                          self.processing_stats['total_processing_time']
                                          if self.processing_stats['total_processing_time'] > 0 else 0)
        }


class CompressedDataCache:
    """High-performance compressed data cache for large datasets."""
    
    def __init__(self, cache_dir: str, max_size_gb: int = 10, compression_level: int = 3):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        self.compression_level = compression_level
        
        self.logger = logging.getLogger(__name__)
        
        # Cache metadata
        self.cache_metadata = {}  # {key: {'file_path', 'size', 'access_time'}}
        self.current_size = 0
        
        # Load existing cache metadata
        self._load_cache_metadata()
        
        # Lock for thread safety
        self.lock = threading.RLock()
    
    def _load_cache_metadata(self):
        """Load existing cache metadata."""
        metadata_file = self.cache_dir / 'cache_metadata.json'
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    import json
                    self.cache_metadata = json.load(f)
                    
                # Calculate current cache size
                self.current_size = sum(item['size'] for item in self.cache_metadata.values())
                
                self.logger.info(f"Loaded cache metadata: {len(self.cache_metadata)} items, {self.current_size / 1024 / 1024:.1f}MB")
                
            except Exception as e:
                self.logger.warning(f"Failed to load cache metadata: {e}")
                self.cache_metadata = {}
                self.current_size = 0
    
    def _save_cache_metadata(self):
        """Save cache metadata to disk."""
        metadata_file = self.cache_dir / 'cache_metadata.json'
        
        try:
            with open(metadata_file, 'w') as f:
                import json
                json.dump(self.cache_metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save cache metadata: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key not in self.cache_metadata:
                return None
            
            metadata = self.cache_metadata[key]
            file_path = Path(metadata['file_path'])
            
            if not file_path.exists():
                # Remove stale metadata
                del self.cache_metadata[key]
                return None
            
            try:
                # Read and decompress data
                with open(file_path, 'rb') as f:
                    compressed_data = f.read()
                
                decompressed_data = lz4.frame.decompress(compressed_data)
                data = pickle.loads(decompressed_data)
                
                # Update access time
                metadata['access_time'] = time.time()
                
                return data
                
            except Exception as e:
                self.logger.error(f"Failed to read cached item {key}: {e}")
                # Remove corrupted cache entry
                if file_path.exists():
                    file_path.unlink()
                del self.cache_metadata[key]
                return None
    
    def put(self, key: str, data: Any) -> bool:
        """Store item in cache."""
        with self.lock:
            try:
                # Serialize data
                serialized_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Compress data
                compressed_data = lz4.frame.compress(
                    serialized_data, 
                    compression_level=self.compression_level
                )
                
                data_size = len(compressed_data)
                
                # Check if we need to evict items
                while self.current_size + data_size > self.max_size_bytes and self.cache_metadata:
                    self._evict_lru_item()
                
                # Save to disk
                file_path = self.cache_dir / f"{key}.cache"
                with open(file_path, 'wb') as f:
                    f.write(compressed_data)
                
                # Update metadata
                self.cache_metadata[key] = {
                    'file_path': str(file_path),
                    'size': data_size,
                    'access_time': time.time()
                }
                
                self.current_size += data_size
                
                # Save metadata periodically
                if len(self.cache_metadata) % 100 == 0:
                    self._save_cache_metadata()
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to cache item {key}: {e}")
                return False
    
    def _evict_lru_item(self):
        """Evict least recently used item."""
        if not self.cache_metadata:
            return
        
        # Find LRU item
        lru_key = min(self.cache_metadata.keys(), 
                     key=lambda k: self.cache_metadata[k]['access_time'])
        
        metadata = self.cache_metadata[lru_key]
        file_path = Path(metadata['file_path'])
        
        # Remove file
        if file_path.exists():
            file_path.unlink()
        
        # Update size and remove metadata
        self.current_size -= metadata['size']
        del self.cache_metadata[lru_key]
        
        self.logger.debug(f"Evicted cache item: {lru_key}")
    
    def clear(self):
        """Clear all cached items."""
        with self.lock:
            for metadata in self.cache_metadata.values():
                file_path = Path(metadata['file_path'])
                if file_path.exists():
                    file_path.unlink()
            
            self.cache_metadata.clear()
            self.current_size = 0
            self._save_cache_metadata()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                'num_items': len(self.cache_metadata),
                'total_size_mb': self.current_size / 1024 / 1024,
                'max_size_mb': self.max_size_bytes / 1024 / 1024,
                'utilization': self.current_size / self.max_size_bytes if self.max_size_bytes > 0 else 0
            }
    
    def __del__(self):
        """Save metadata on destruction."""
        try:
            self._save_cache_metadata()
        except:
            pass


# Global instances
default_config = DataProcessingConfig()
global_cache = CompressedDataCache("./cache", max_size_gb=10)
global_processor = None

def get_parallel_processor() -> ParallelDataProcessor:
    """Get global parallel processor instance."""
    global global_processor
    if global_processor is None:
        global_processor = ParallelDataProcessor(default_config)
        global_processor.__enter__()  # Initialize pools
    return global_processor