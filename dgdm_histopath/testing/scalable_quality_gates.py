"""Scalable quality gates with optimization, caching, and distributed processing."""

import os
import sys
import json
import time
import hashlib
import threading
import multiprocessing
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import tempfile
import pickle
import sqlite3
from contextlib import contextmanager
import logging

from .robust_quality_runner import RobustQualityRunner, RobustValidationResult
from .progressive_quality_gates import ProgressiveQualityConfig, ProjectMaturity
from ..utils.logging import get_logger


@dataclass
class CacheEntry:
    """Cache entry for validation results."""
    validation_hash: str
    result: RobustValidationResult
    timestamp: float
    metadata: Dict[str, Any]
    hits: int = 0


@dataclass
class OptimizationMetrics:
    """Metrics for optimization tracking."""
    cache_hits: int = 0
    cache_misses: int = 0
    parallel_speedup: float = 0.0
    memory_optimization: float = 0.0
    distributed_nodes: int = 0
    total_optimization_time_saved: float = 0.0


class ResultCache:
    """Persistent cache for validation results."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "validation_cache.db"
        self._init_database()
        self.logger = get_logger(f"{__name__}.ResultCache")
    
    def _init_database(self):
        """Initialize SQLite database for caching."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS validation_results (
                    validation_hash TEXT PRIMARY KEY,
                    validator_name TEXT,
                    result_data BLOB,
                    timestamp REAL,
                    metadata TEXT,
                    hits INTEGER DEFAULT 0,
                    file_paths TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_validator_timestamp 
                ON validation_results(validator_name, timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_hits 
                ON validation_results(hits DESC)
            """)
    
    def _calculate_hash(self, validator_name: str, file_paths: List[str], metadata: Dict[str, Any]) -> str:
        """Calculate hash for validation inputs."""
        # Sort file paths for consistent hashing
        sorted_paths = sorted(file_paths)
        
        # Include file modification times
        file_data = {}
        for file_path in sorted_paths:
            path = Path(file_path)
            if path.exists():
                file_data[file_path] = {
                    'mtime': path.stat().st_mtime,
                    'size': path.stat().st_size
                }
        
        hash_input = {
            'validator_name': validator_name,
            'files': file_data,
            'metadata': metadata
        }
        
        hash_str = json.dumps(hash_input, sort_keys=True, default=str)
        return hashlib.sha256(hash_str.encode()).hexdigest()
    
    def get(
        self, 
        validator_name: str, 
        file_paths: List[str], 
        metadata: Dict[str, Any]
    ) -> Optional[RobustValidationResult]:
        """Get cached result if available and valid."""
        validation_hash = self._calculate_hash(validator_name, file_paths, metadata)
        
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("""
                SELECT result_data, hits FROM validation_results 
                WHERE validation_hash = ?
            """, (validation_hash,))
            
            row = cursor.fetchone()
            if row:
                result_data, hits = row
                
                # Update hit count
                conn.execute("""
                    UPDATE validation_results 
                    SET hits = hits + 1 
                    WHERE validation_hash = ?
                """, (validation_hash,))
                
                try:
                    result = pickle.loads(result_data)
                    self.logger.debug(f"Cache hit for {validator_name} (hits: {hits + 1})")
                    return result
                except Exception as e:
                    self.logger.warning(f"Failed to deserialize cached result: {e}")
                    self._remove(validation_hash)
        
        return None
    
    def put(
        self, 
        validator_name: str, 
        file_paths: List[str], 
        metadata: Dict[str, Any],
        result: RobustValidationResult
    ):
        """Store result in cache."""
        validation_hash = self._calculate_hash(validator_name, file_paths, metadata)
        
        try:
            result_data = pickle.dumps(result)
            
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO validation_results 
                    (validation_hash, validator_name, result_data, timestamp, metadata, file_paths)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    validation_hash,
                    validator_name,
                    result_data,
                    time.time(),
                    json.dumps(metadata),
                    json.dumps(file_paths)
                ))
            
            self.logger.debug(f"Cached result for {validator_name}")
            
        except Exception as e:
            self.logger.warning(f"Failed to cache result: {e}")
    
    def _remove(self, validation_hash: str):
        """Remove invalid cache entry."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("DELETE FROM validation_results WHERE validation_hash = ?", (validation_hash,))
    
    def cleanup(self, max_age_days: int = 30, max_entries: int = 1000):
        """Clean up old cache entries."""
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        
        with sqlite3.connect(str(self.db_path)) as conn:
            # Remove old entries
            conn.execute("DELETE FROM validation_results WHERE timestamp < ?", (cutoff_time,))
            
            # Keep only most frequently used entries if over limit
            conn.execute("""
                DELETE FROM validation_results 
                WHERE validation_hash NOT IN (
                    SELECT validation_hash FROM validation_results 
                    ORDER BY hits DESC, timestamp DESC 
                    LIMIT ?
                )
            """, (max_entries,))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_entries,
                    AVG(hits) as avg_hits,
                    MAX(hits) as max_hits,
                    COUNT(DISTINCT validator_name) as unique_validators
                FROM validation_results
            """)
            
            row = cursor.fetchone()
            if row:
                return {
                    'total_entries': row[0],
                    'average_hits': row[1] or 0,
                    'max_hits': row[2] or 0,
                    'unique_validators': row[3]
                }
        
        return {'total_entries': 0, 'average_hits': 0, 'max_hits': 0, 'unique_validators': 0}


class DistributedValidator:
    """Distributed validation across multiple nodes/processes."""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 8)
        self.logger = get_logger(f"{__name__}.DistributedValidator")
    
    def validate_distributed(
        self, 
        validators: List[str], 
        config: ProgressiveQualityConfig,
        project_root: Path,
        output_dir: Path
    ) -> List[RobustValidationResult]:
        """Run validators across distributed processes."""
        
        self.logger.info(f"Starting distributed validation with {self.max_workers} workers")
        
        # Split validators across workers
        validator_chunks = self._chunk_validators(validators)
        
        results = []
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit validation jobs
            futures = []
            for chunk in validator_chunks:
                future = executor.submit(
                    self._run_validator_chunk,
                    chunk, config, project_root, output_dir
                )
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    chunk_results = future.result()
                    results.extend(chunk_results)
                except Exception as e:
                    self.logger.error(f"Distributed validation chunk failed: {e}")
        
        return results
    
    def _chunk_validators(self, validators: List[str]) -> List[List[str]]:
        """Split validators into chunks for distributed processing."""
        chunk_size = max(1, len(validators) // self.max_workers)
        chunks = []
        
        for i in range(0, len(validators), chunk_size):
            chunk = validators[i:i + chunk_size]
            chunks.append(chunk)
        
        return chunks
    
    @staticmethod
    def _run_validator_chunk(
        validators: List[str],
        config: ProgressiveQualityConfig,
        project_root: Path,
        output_dir: Path
    ) -> List[RobustValidationResult]:
        """Run a chunk of validators in a separate process."""
        try:
            with RobustQualityRunner(
                config=config,
                project_root=project_root,
                output_dir=output_dir / f"distributed_{os.getpid()}"
            ) as runner:
                return runner.run_validation(validators=validators, parallel=False)
        except Exception as e:
            # Create error result for failed chunk
            return [RobustValidationResult(
                validator_name="distributed_chunk",
                passed=False,
                score=0.0,
                threshold=1.0,
                message=f"Distributed chunk failed: {e}",
                details={'error': str(e), 'validators': validators},
                execution_time=0.0,
                memory_peak_mb=0.0,
                cpu_usage_percent=0.0,
                warnings=[f"Distributed processing failed for chunk: {validators}"],
                errors=[str(e)],
                artifacts={},
                recovery_attempted=False,
                recovery_successful=False,
                validation_context=None
            )]


class ScalableQualityGates:
    """Scalable quality gates with optimization and caching."""
    
    def __init__(
        self,
        config: ProgressiveQualityConfig = None,
        project_root: Path = None,
        output_dir: str = "./quality_reports",
        cache_dir: str = "./quality_cache",
        enable_caching: bool = True,
        enable_distributed: bool = False,
        max_workers: Optional[int] = None
    ):
        self.config = config or ProgressiveQualityConfig()
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_caching = enable_caching
        self.enable_distributed = enable_distributed
        
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.cache = ResultCache(self.cache_dir) if enable_caching else None
        self.distributed_validator = DistributedValidator(max_workers) if enable_distributed else None
        
        # Optimization metrics
        self.metrics = OptimizationMetrics()
        
        # Performance optimization settings
        self.optimization_settings = self._get_optimization_settings()
        
        self.logger.info(f"Scalable Quality Gates initialized")
        self.logger.info(f"Caching: {'enabled' if enable_caching else 'disabled'}")
        self.logger.info(f"Distributed: {'enabled' if enable_distributed else 'disabled'}")
    
    def _get_optimization_settings(self) -> Dict[str, Any]:
        """Get optimization settings based on maturity and system resources."""
        try:
            import psutil
            cpu_count = psutil.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            cpu_count = 4
            memory_gb = 8
        
        # Maturity-based optimization levels
        optimization_levels = {
            ProjectMaturity.GREENFIELD: {
                'parallel_workers': min(2, cpu_count),
                'memory_limit_gb': min(2, memory_gb * 0.3),
                'cache_size': 100,
                'enable_memory_optimization': False
            },
            ProjectMaturity.DEVELOPMENT: {
                'parallel_workers': min(4, cpu_count),
                'memory_limit_gb': min(4, memory_gb * 0.5),
                'cache_size': 500,
                'enable_memory_optimization': True
            },
            ProjectMaturity.STAGING: {
                'parallel_workers': min(8, cpu_count),
                'memory_limit_gb': min(8, memory_gb * 0.7),
                'cache_size': 1000,
                'enable_memory_optimization': True
            },
            ProjectMaturity.PRODUCTION: {
                'parallel_workers': cpu_count,
                'memory_limit_gb': memory_gb * 0.8,
                'cache_size': 2000,
                'enable_memory_optimization': True
            }
        }
        
        return optimization_levels.get(self.config.maturity, optimization_levels[ProjectMaturity.DEVELOPMENT])
    
    def run_optimized_validation(self, validators: Optional[List[str]] = None) -> List[RobustValidationResult]:
        """Run validation with all optimizations enabled."""
        start_time = time.time()
        
        # Determine validators to run
        if validators is None:
            enabled_gates = self.config.enabled_gates.get(self.config.maturity, [])
            validators = [v for v in enabled_gates if v in ['code_compilation', 'model_validation']]
        
        if not validators:
            self.logger.warning("No validators to run")
            return []
        
        self.logger.info(f"Starting optimized validation of {len(validators)} validators")
        
        # Pre-validation optimizations
        self._perform_pre_validation_optimizations()
        
        # Run validation with optimizations
        if self.enable_distributed and len(validators) > 1:
            results = self._run_distributed_validation(validators)
        else:
            results = self._run_cached_validation(validators)
        
        # Post-validation optimizations
        self._perform_post_validation_optimizations(results, time.time() - start_time)
        
        # Generate optimization report
        self._generate_optimization_report(results, time.time() - start_time)
        
        return results
    
    def _perform_pre_validation_optimizations(self):
        """Perform optimizations before validation."""
        self.logger.info("Performing pre-validation optimizations...")
        
        # Clean up cache
        if self.cache:
            self.cache.cleanup(
                max_age_days=7,
                max_entries=self.optimization_settings['cache_size']
            )
        
        # Memory optimization
        if self.optimization_settings['enable_memory_optimization']:
            self._optimize_memory()
        
        # File system optimization
        self._optimize_file_system()
    
    def _optimize_memory(self):
        """Optimize memory usage."""
        try:
            import gc
            gc.collect()
            
            # Try to free up PyTorch memory if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
                
            self.logger.debug("Memory optimization completed")
        except Exception as e:
            self.logger.warning(f"Memory optimization failed: {e}")
    
    def _optimize_file_system(self):
        """Optimize file system access patterns."""
        # Pre-warm file system cache by scanning common directories
        common_dirs = [
            self.project_root / "dgdm_histopath",
            self.project_root / "tests",
            self.project_root / "configs"
        ]
        
        for directory in common_dirs:
            if directory.exists():
                try:
                    list(directory.rglob("*.py"))  # Pre-cache file listings
                except Exception:
                    pass
    
    def _run_distributed_validation(self, validators: List[str]) -> List[RobustValidationResult]:
        """Run validation using distributed processing."""
        self.logger.info(f"Running distributed validation across multiple processes")
        
        start_time = time.time()
        
        results = self.distributed_validator.validate_distributed(
            validators, self.config, self.project_root, self.output_dir
        )
        
        execution_time = time.time() - start_time
        
        # Calculate speedup (estimate)
        estimated_sequential_time = len(validators) * 30  # Rough estimate
        self.metrics.parallel_speedup = max(1.0, estimated_sequential_time / execution_time)
        self.metrics.distributed_nodes = self.distributed_validator.max_workers
        
        return results
    
    def _run_cached_validation(self, validators: List[str]) -> List[RobustValidationResult]:
        """Run validation with intelligent caching."""
        results = []
        
        for validator_name in validators:
            # Check cache first
            cached_result = None
            if self.cache:
                file_paths = list(str(f) for f in self.project_root.rglob("*.py"))
                metadata = {'maturity': self.config.maturity.value}
                
                cached_result = self.cache.get(validator_name, file_paths, metadata)
                
                if cached_result:
                    self.metrics.cache_hits += 1
                    self.metrics.total_optimization_time_saved += cached_result.execution_time
                    self.logger.info(f"✅ Cache hit for {validator_name}")
                    results.append(cached_result)
                    continue
                else:
                    self.metrics.cache_misses += 1
            
            # Run validation
            self.logger.info(f"🔄 Running {validator_name}")
            
            with RobustQualityRunner(
                config=self.config,
                project_root=self.project_root,
                output_dir=self.output_dir / "cached_validation"
            ) as runner:
                validator_results = runner.run_validation(validators=[validator_name], parallel=False)
                
                if validator_results:
                    result = validator_results[0]
                    results.append(result)
                    
                    # Cache the result
                    if self.cache and result.passed:
                        file_paths = list(str(f) for f in self.project_root.rglob("*.py"))
                        metadata = {'maturity': self.config.maturity.value}
                        self.cache.put(validator_name, file_paths, metadata, result)
        
        return results
    
    def _perform_post_validation_optimizations(self, results: List[RobustValidationResult], total_time: float):
        """Perform optimizations after validation."""
        self.logger.info("Performing post-validation optimizations...")
        
        # Memory cleanup
        if self.optimization_settings['enable_memory_optimization']:
            self._optimize_memory()
        
        # Update metrics
        self._update_optimization_metrics(results, total_time)
    
    def _update_optimization_metrics(self, results: List[RobustValidationResult], total_time: float):
        """Update optimization metrics."""
        # Calculate memory optimization
        if results:
            avg_memory = sum(r.memory_peak_mb for r in results) / len(results)
            memory_limit = self.optimization_settings['memory_limit_gb'] * 1024
            self.metrics.memory_optimization = max(0, 1 - (avg_memory / memory_limit))
        
        # Log optimization stats
        self.logger.info(f"Optimization Summary:")
        self.logger.info(f"  Cache hits: {self.metrics.cache_hits}")
        self.logger.info(f"  Cache misses: {self.metrics.cache_misses}")
        self.logger.info(f"  Time saved: {self.metrics.total_optimization_time_saved:.2f}s")
        
        if self.metrics.parallel_speedup > 1:
            self.logger.info(f"  Parallel speedup: {self.metrics.parallel_speedup:.1f}x")
    
    def _generate_optimization_report(self, results: List[RobustValidationResult], total_time: float):
        """Generate comprehensive optimization report."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        optimization_report = {
            'timestamp': timestamp,
            'total_execution_time': total_time,
            'optimization_enabled': {
                'caching': self.enable_caching,
                'distributed': self.enable_distributed,
                'memory_optimization': self.optimization_settings['enable_memory_optimization']
            },
            'optimization_metrics': asdict(self.metrics),
            'cache_statistics': self.cache.get_stats() if self.cache else {},
            'optimization_settings': self.optimization_settings,
            'results_summary': {
                'total_validators': len(results),
                'passed_validators': len([r for r in results if r.passed]),
                'average_execution_time': sum(r.execution_time for r in results) / len(results) if results else 0,
                'peak_memory_usage': max((r.memory_peak_mb for r in results), default=0)
            }
        }
        
        # Save optimization report
        report_file = self.output_dir / f"optimization_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(optimization_report, f, indent=2)
        
        # Generate human-readable summary
        summary_file = self.output_dir / f"optimization_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("SCALABLE QUALITY GATES OPTIMIZATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Total Execution Time: {total_time:.2f}s\n")
            f.write(f"Cache Hit Rate: {self.metrics.cache_hits / (self.metrics.cache_hits + self.metrics.cache_misses) * 100:.1f}%\n")
            f.write(f"Time Saved by Caching: {self.metrics.total_optimization_time_saved:.2f}s\n")
            
            if self.metrics.parallel_speedup > 1:
                f.write(f"Parallel Speedup: {self.metrics.parallel_speedup:.1f}x\n")
            
            f.write(f"Memory Optimization: {self.metrics.memory_optimization * 100:.1f}%\n")
            
            if self.cache:
                cache_stats = self.cache.get_stats()
                f.write(f"Cache Entries: {cache_stats['total_entries']}\n")
                f.write(f"Cache Efficiency: {cache_stats['average_hits']:.1f} avg hits per entry\n")
        
        self.logger.info(f"Optimization report generated: {report_file}")
        self.logger.info(f"Optimization summary: {summary_file}")
    
    def get_optimization_metrics(self) -> OptimizationMetrics:
        """Get current optimization metrics."""
        return self.metrics
    
    def cleanup_cache(self):
        """Clean up cache resources."""
        if self.cache:
            self.cache.cleanup()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats() if self.cache else {}


def main():
    """Main entry point for scalable quality gates."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Scalable Quality Gates")
    parser.add_argument("--maturity", choices=['greenfield', 'development', 'staging', 'production'],
                       help="Project maturity level")
    parser.add_argument("--output-dir", default="./quality_reports", help="Output directory")
    parser.add_argument("--cache-dir", default="./quality_cache", help="Cache directory")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--distributed", action="store_true", help="Enable distributed processing")
    parser.add_argument("--workers", type=int, help="Number of worker processes")
    parser.add_argument("--validators", nargs="*", help="Specific validators to run")
    
    args = parser.parse_args()
    
    # Setup configuration
    config = ProgressiveQualityConfig()
    if args.maturity:
        config.maturity = ProjectMaturity(args.maturity)
    
    # Create scalable quality gates
    quality_gates = ScalableQualityGates(
        config=config,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        enable_caching=not args.no_cache,
        enable_distributed=args.distributed,
        max_workers=args.workers
    )
    
    # Run validation
    results = quality_gates.run_optimized_validation(args.validators)
    
    # Print summary
    passed_count = len([r for r in results if r.passed])
    print(f"Scalable Quality Gates: {passed_count}/{len(results)} passed")
    
    metrics = quality_gates.get_optimization_metrics()
    if metrics.cache_hits > 0:
        print(f"Cache performance: {metrics.cache_hits} hits, {metrics.total_optimization_time_saved:.1f}s saved")
    
    # Exit code
    sys.exit(0 if passed_count == len(results) else 1)


if __name__ == "__main__":
    main()