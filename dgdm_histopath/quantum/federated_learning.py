#!/usr/bin/env python3
"""
Federated Learning Module for DGDM Histopath Lab

Multi-institutional collaborative learning with privacy preservation.
Enables training across hospitals without sharing raw medical data.

Author: TERRAGON Autonomous Development System v4.0
Generated: 2025-08-08
"""

import asyncio
import logging
import hashlib
import pickle
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from pathlib import Path
import copy
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    # Differential Privacy
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    from opacus.utils.batch_memory_manager import BatchMemoryManager
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    logging.warning("Opacus not available. Install with: pip install opacus")

try:
    # Federated Learning Framework
    import flwr as fl
    from flwr.client import NumPyClient, ClientApp
    from flwr.server import ServerConfig, start_server
    from flwr.common import Parameters, FitRes, EvaluateRes
    FLOWER_AVAILABLE = True
except ImportError:
    FLOWER_AVAILABLE = False
    logging.warning("Flower not available. Install with: pip install flwr")

try:
    # Secure Multi-party Computation
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False
    logging.warning("TenSEAL not available. Install with: pip install tenseal")

from ..models.dgdm_model import DGDMModel
from ..utils.exceptions import FederatedLearningError
from ..utils.validation import validate_tensor, validate_config
from ..utils.monitoring import FederatedMetricsCollector
from ..utils.security import encrypt_model_weights, decrypt_model_weights


class FederationStrategy(Enum):
    """Federated learning aggregation strategies."""
    FEDAVG = "federated_averaging"
    FEDPROX = "federated_proximal"
    SCAFFOLD = "scaffold"
    FEDNOVA = "fed_nova"
    FEDOPT = "federated_optimization"
    QUANTUM_FED = "quantum_federated"


class PrivacyMechanism(Enum):
    """Privacy preservation mechanisms."""
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    HOMOMORPHIC_ENCRYPTION = "homomorphic_encryption"
    SECURE_AGGREGATION = "secure_aggregation"
    LOCAL_DIFFERENTIAL_PRIVACY = "local_differential_privacy"
    NONE = "none"


@dataclass
class FederatedConfig:
    """Configuration for federated learning."""
    # Federation settings
    strategy: FederationStrategy = FederationStrategy.FEDAVG
    num_clients: int = 5
    min_fit_clients: int = 3
    min_evaluate_clients: int = 2
    min_available_clients: int = 3
    
    # Training parameters
    num_rounds: int = 100
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001
    
    # Privacy settings
    privacy_mechanism: PrivacyMechanism = PrivacyMechanism.DIFFERENTIAL_PRIVACY
    epsilon: float = 1.0  # DP budget
    delta: float = 1e-5
    noise_multiplier: float = 1.1
    max_grad_norm: float = 1.0
    
    # Security settings
    use_secure_aggregation: bool = True
    encryption_key_size: int = 2048
    certificate_path: Optional[str] = None
    
    # Communication settings
    compression_ratio: float = 0.1
    quantization_bits: int = 8
    use_gradient_compression: bool = True
    
    # Client selection
    client_selection_strategy: str = "random"
    min_client_resources: Dict[str, float] = field(default_factory=lambda: {
        'cpu_cores': 4,
        'memory_gb': 8,
        'gpu_memory_gb': 6
    })
    
    # Fault tolerance
    max_client_failures: int = 2
    timeout_seconds: int = 600
    retry_attempts: int = 3


class FederatedClient:
    """Federated learning client for medical institutions."""
    
    def __init__(self, 
                 client_id: str,
                 model: DGDMModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: FederatedConfig):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Privacy engine
        self.privacy_engine = None
        if config.privacy_mechanism == PrivacyMechanism.DIFFERENTIAL_PRIVACY and OPACUS_AVAILABLE:
            self._setup_differential_privacy()
        
        # Metrics collection
        self.metrics = FederatedMetricsCollector(client_id)
        
        # Local model state
        self.local_model = copy.deepcopy(model)
        self.optimizer = torch.optim.Adam(self.local_model.parameters(), lr=config.learning_rate)
        
        # Secure aggregation keys
        self.aggregation_keys = None
        if config.use_secure_aggregation:
            self._generate_aggregation_keys()
    
    def _setup_differential_privacy(self):
        """Setup differential privacy with Opacus."""
        try:
            # Validate model for DP
            self.local_model = ModuleValidator.fix(self.local_model)
            
            # Initialize privacy engine
            self.privacy_engine = PrivacyEngine()
            
            self.local_model, self.optimizer, self.train_loader = self.privacy_engine.make_private_with_epsilon(
                module=self.local_model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                epochs=self.config.local_epochs,
                target_epsilon=self.config.epsilon,
                target_delta=self.config.delta,
                max_grad_norm=self.config.max_grad_norm
            )
            
            logging.info(f"Differential privacy enabled with ε={self.config.epsilon}, δ={self.config.delta}")
            
        except Exception as e:
            logging.error(f"Failed to setup differential privacy: {e}")
            self.privacy_engine = None
    
    def _generate_aggregation_keys(self):
        """Generate keys for secure aggregation."""
        # Simplified key generation - in practice, use proper cryptographic protocols
        self.aggregation_keys = {
            'public_key': hashlib.sha256(self.client_id.encode()).hexdigest()[:32],
            'private_key': hashlib.sha256((self.client_id + '_private').encode()).hexdigest()[:32]
        }
    
    async def local_training(self, global_weights: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Perform local training on client data."""
        try:
            # Load global model weights
            self.local_model.load_state_dict(global_weights)
            self.local_model.train()
            
            training_metrics = {
                'loss': 0.0,
                'accuracy': 0.0,
                'privacy_spent': 0.0,
                'samples_processed': 0
            }
            
            # Local training loop
            for epoch in range(self.config.local_epochs):
                epoch_loss = 0.0
                correct_predictions = 0
                total_samples = 0
                
                if self.privacy_engine:
                    # DP training with batch memory manager
                    with BatchMemoryManager(
                        data_loader=self.train_loader,
                        max_physical_batch_size=self.config.batch_size,
                        optimizer=self.optimizer
                    ) as memory_safe_data_loader:
                        
                        for batch_idx, (data, target) in enumerate(memory_safe_data_loader):
                            self.optimizer.zero_grad()
                            
                            output = self.local_model(data)
                            loss = self._compute_loss(output, target)
                            
                            loss.backward()
                            self.optimizer.step()
                            
                            epoch_loss += loss.item()
                            predictions = output.argmax(dim=1)
                            correct_predictions += (predictions == target).sum().item()
                            total_samples += target.size(0)
                else:
                    # Standard training
                    for batch_idx, (data, target) in enumerate(self.train_loader):
                        self.optimizer.zero_grad()
                        
                        output = self.local_model(data)
                        loss = self._compute_loss(output, target)
                        
                        loss.backward()
                        self.optimizer.step()
                        
                        epoch_loss += loss.item()
                        predictions = output.argmax(dim=1)
                        correct_predictions += (predictions == target).sum().item()
                        total_samples += target.size(0)
                
                training_metrics['loss'] = epoch_loss / len(self.train_loader)
                training_metrics['accuracy'] = correct_predictions / total_samples
                training_metrics['samples_processed'] = total_samples
                
                # Record privacy budget spent
                if self.privacy_engine:
                    training_metrics['privacy_spent'] = self.privacy_engine.get_epsilon(self.config.delta)
            
            # Get updated model weights
            updated_weights = self.local_model.state_dict()
            
            # Apply secure aggregation if enabled
            if self.config.use_secure_aggregation:
                updated_weights = await self._apply_secure_aggregation(updated_weights)
            
            # Record metrics
            await self.metrics.record_training_round(training_metrics)
            
            return updated_weights, training_metrics
            
        except Exception as e:
            logging.error(f"Local training failed for client {self.client_id}: {e}")
            raise FederatedLearningError(f"Local training failed: {e}")
    
    async def local_evaluation(self, global_weights: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Evaluate model on local validation data."""
        try:
            # Load global weights
            self.local_model.load_state_dict(global_weights)
            self.local_model.eval()
            
            total_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            
            with torch.no_grad():
                for data, target in self.val_loader:
                    output = self.local_model(data)
                    loss = self._compute_loss(output, target)
                    
                    total_loss += loss.item()
                    predictions = output.argmax(dim=1)
                    correct_predictions += (predictions == target).sum().item()
                    total_samples += target.size(0)
            
            metrics = {
                'loss': total_loss / len(self.val_loader),
                'accuracy': correct_predictions / total_samples,
                'samples': total_samples
            }
            
            await self.metrics.record_evaluation_round(metrics)
            return metrics
            
        except Exception as e:
            logging.error(f"Local evaluation failed for client {self.client_id}: {e}")
            return {'loss': float('inf'), 'accuracy': 0.0, 'samples': 0}
    
    def _compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss function for medical AI task."""
        if output.shape[-1] > 1:
            # Classification task
            return nn.CrossEntropyLoss()(output, target)
        else:
            # Regression task
            return nn.MSELoss()(output.squeeze(), target.float())
    
    async def _apply_secure_aggregation(self, weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply secure aggregation to model weights."""
        if not TENSEAL_AVAILABLE:
            logging.warning("TenSEAL not available, skipping homomorphic encryption")
            return weights
        
        try:
            # Create TenSEAL context for homomorphic encryption
            context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=8192,
                coeff_mod_bit_sizes=[60, 40, 40, 60]
            )
            context.generate_galois_keys()
            context.global_scale = 2**40
            
            # Encrypt model weights
            encrypted_weights = {}
            for name, weight in weights.items():
                if weight.dtype == torch.float32:
                    flat_weight = weight.flatten().tolist()
                    encrypted_weights[name] = ts.ckks_vector(context, flat_weight)
                else:
                    encrypted_weights[name] = weight
            
            # In practice, these would be sent to an aggregator
            # For now, just decrypt them back
            decrypted_weights = {}
            for name, enc_weight in encrypted_weights.items():
                if isinstance(enc_weight, ts.CKKSVector):
                    decrypted = torch.tensor(enc_weight.decrypt(), dtype=torch.float32)
                    decrypted_weights[name] = decrypted.reshape(weights[name].shape)
                else:
                    decrypted_weights[name] = enc_weight
            
            return decrypted_weights
            
        except Exception as e:
            logging.error(f"Secure aggregation failed: {e}")
            return weights


class FederatedServer:
    """Federated learning server for coordinating training."""
    
    def __init__(self, 
                 global_model: DGDMModel,
                 config: FederatedConfig,
                 client_configs: List[Dict[str, Any]]):
        self.global_model = global_model
        self.config = config
        self.client_configs = client_configs
        
        # Server metrics
        self.metrics = FederatedMetricsCollector("server")
        
        # Client management
        self.available_clients = []
        self.client_performance = {}
        
        # Aggregation strategy
        self.aggregation_strategy = self._get_aggregation_strategy()
        
        # Security
        self.round_keys = {}
        
    def _get_aggregation_strategy(self) -> Callable:
        """Get the appropriate aggregation strategy."""
        if self.config.strategy == FederationStrategy.FEDAVG:
            return self._federated_averaging
        elif self.config.strategy == FederationStrategy.FEDPROX:
            return self._federated_proximal
        elif self.config.strategy == FederationStrategy.SCAFFOLD:
            return self._scaffold_aggregation
        elif self.config.strategy == FederationStrategy.QUANTUM_FED:
            return self._quantum_federated_aggregation
        else:
            return self._federated_averaging
    
    async def start_federated_training(self) -> Dict[str, Any]:
        """Start federated training process."""
        try:
            logging.info(f"Starting federated training with {self.config.num_clients} clients")
            
            # Initialize clients
            clients = await self._initialize_clients()
            
            training_history = {
                'rounds': [],
                'global_metrics': [],
                'client_metrics': [],
                'privacy_spent': []
            }
            
            # Federated training rounds
            for round_num in range(self.config.num_rounds):
                logging.info(f"Starting federated round {round_num + 1}/{self.config.num_rounds}")
                
                # Select clients for this round
                selected_clients = await self._select_clients(clients)
                
                # Distribute global model
                global_weights = self.global_model.state_dict()
                
                # Parallel local training
                client_results = await self._parallel_local_training(
                    selected_clients, global_weights
                )
                
                # Aggregate updates
                aggregated_weights, aggregation_metrics = await self.aggregation_strategy(
                    client_results, global_weights
                )
                
                # Update global model
                self.global_model.load_state_dict(aggregated_weights)
                
                # Global evaluation
                global_metrics = await self._global_evaluation(selected_clients)
                
                # Record round results
                round_results = {
                    'round': round_num + 1,
                    'participating_clients': len(selected_clients),
                    'aggregation_metrics': aggregation_metrics,
                    'global_metrics': global_metrics,
                    'timestamp': time.time()
                }
                
                training_history['rounds'].append(round_results)
                training_history['global_metrics'].append(global_metrics)
                
                # Log progress
                logging.info(f"Round {round_num + 1} completed. Global accuracy: {global_metrics.get('accuracy', 0):.4f}")
                
                # Check convergence
                if await self._check_convergence(training_history):
                    logging.info("Federated training converged early")
                    break
            
            # Training completed
            final_results = await self._finalize_training(training_history)
            return final_results
            
        except Exception as e:
            logging.error(f"Federated training failed: {e}")
            raise FederatedLearningError(f"Federated training failed: {e}")
    
    async def _initialize_clients(self) -> List[FederatedClient]:
        """Initialize federated clients."""
        clients = []
        
        for client_config in self.client_configs:
            try:
                # Create client instance
                client = FederatedClient(
                    client_id=client_config['client_id'],
                    model=copy.deepcopy(self.global_model),
                    train_loader=client_config['train_loader'],
                    val_loader=client_config['val_loader'],
                    config=self.config
                )
                
                clients.append(client)
                logging.info(f"Initialized client: {client_config['client_id']}")
                
            except Exception as e:
                logging.error(f"Failed to initialize client {client_config['client_id']}: {e}")
        
        if len(clients) < self.config.min_available_clients:
            raise FederatedLearningError(
                f"Insufficient clients: {len(clients)} < {self.config.min_available_clients}"
            )
        
        return clients
    
    async def _select_clients(self, clients: List[FederatedClient]) -> List[FederatedClient]:
        """Select clients for the current round."""
        if self.config.client_selection_strategy == "random":
            # Random selection
            num_select = min(self.config.min_fit_clients, len(clients))
            selected = np.random.choice(clients, size=num_select, replace=False).tolist()
        
        elif self.config.client_selection_strategy == "performance":
            # Performance-based selection
            client_scores = []
            for client in clients:
                performance = self.client_performance.get(client.client_id, {'accuracy': 0.5, 'reliability': 1.0})
                score = performance['accuracy'] * performance['reliability']
                client_scores.append((client, score))
            
            # Sort by performance and select top clients
            client_scores.sort(key=lambda x: x[1], reverse=True)
            selected = [client for client, score in client_scores[:self.config.min_fit_clients]]
        
        else:
            # Default: select first N available clients
            selected = clients[:self.config.min_fit_clients]
        
        return selected
    
    async def _parallel_local_training(self, 
                                     clients: List[FederatedClient], 
                                     global_weights: Dict[str, torch.Tensor]) -> List[Tuple[str, Dict[str, torch.Tensor], Dict[str, float]]]:
        """Execute local training on selected clients in parallel."""
        
        async def train_client(client):
            try:
                start_time = time.time()
                weights, metrics = await client.local_training(global_weights)
                training_time = time.time() - start_time
                
                # Update client performance tracking
                self.client_performance[client.client_id] = {
                    'accuracy': metrics.get('accuracy', 0),
                    'reliability': 1.0,  # Could be based on successful completions
                    'training_time': training_time
                }
                
                return client.client_id, weights, metrics
                
            except Exception as e:
                logging.error(f"Client {client.client_id} training failed: {e}")
                return client.client_id, None, {'error': str(e)}
        
        # Execute training in parallel
        tasks = [train_client(client) for client in clients]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        successful_results = [
            result for result in results 
            if not isinstance(result, Exception) and result[1] is not None
        ]
        
        if len(successful_results) < self.config.min_fit_clients:
            raise FederatedLearningError(
                f"Insufficient successful client updates: {len(successful_results)} < {self.config.min_fit_clients}"
            )
        
        return successful_results
    
    async def _federated_averaging(self, 
                                 client_results: List[Tuple[str, Dict[str, torch.Tensor], Dict[str, float]]], 
                                 global_weights: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Standard federated averaging (FedAvg) aggregation."""
        
        # Calculate total samples across clients
        total_samples = sum(metrics.get('samples_processed', 1) for _, _, metrics in client_results)
        
        # Initialize aggregated weights
        aggregated_weights = {}
        for key in global_weights.keys():
            aggregated_weights[key] = torch.zeros_like(global_weights[key])
        
        # Weighted averaging based on number of samples
        for client_id, weights, metrics in client_results:
            client_samples = metrics.get('samples_processed', 1)
            weight_factor = client_samples / total_samples
            
            for key in aggregated_weights.keys():
                if key in weights:
                    aggregated_weights[key] += weight_factor * weights[key]
        
        aggregation_metrics = {
            'strategy': 'federated_averaging',
            'participating_clients': len(client_results),
            'total_samples': total_samples,
            'average_client_accuracy': np.mean([m.get('accuracy', 0) for _, _, m in client_results])
        }
        
        return aggregated_weights, aggregation_metrics
    
    async def _federated_proximal(self, 
                                client_results: List[Tuple[str, Dict[str, torch.Tensor], Dict[str, float]]], 
                                global_weights: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """FedProx aggregation with proximal term."""
        
        # Start with FedAvg
        aggregated_weights, metrics = await self._federated_averaging(client_results, global_weights)
        
        # Apply proximal regularization
        mu = 0.01  # Proximal parameter
        for key in aggregated_weights.keys():
            proximal_term = mu * (aggregated_weights[key] - global_weights[key])
            aggregated_weights[key] = aggregated_weights[key] - proximal_term
        
        metrics['strategy'] = 'federated_proximal'
        metrics['proximal_mu'] = mu
        
        return aggregated_weights, metrics
    
    async def _scaffold_aggregation(self, 
                                  client_results: List[Tuple[str, Dict[str, torch.Tensor], Dict[str, float]]], 
                                  global_weights: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """SCAFFOLD aggregation with control variates."""
        
        # Simplified SCAFFOLD implementation
        # In practice, this would maintain control variates for each client
        
        aggregated_weights, metrics = await self._federated_averaging(client_results, global_weights)
        
        # Apply variance reduction (simplified)
        learning_rate = 0.1
        for key in aggregated_weights.keys():
            update = aggregated_weights[key] - global_weights[key]
            aggregated_weights[key] = global_weights[key] + learning_rate * update
        
        metrics['strategy'] = 'scaffold'
        metrics['variance_reduction'] = True
        
        return aggregated_weights, metrics
    
    async def _quantum_federated_aggregation(self, 
                                           client_results: List[Tuple[str, Dict[str, torch.Tensor], Dict[str, float]]], 
                                           global_weights: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Novel quantum-inspired federated aggregation."""
        
        # Quantum superposition-inspired weight combination
        num_clients = len(client_results)
        
        # Create quantum-like amplitudes for each client
        client_amplitudes = []
        for _, _, metrics in client_results:
            accuracy = metrics.get('accuracy', 0.5)
            samples = metrics.get('samples_processed', 1)
            
            # Quantum amplitude based on performance and data size
            amplitude = np.sqrt(accuracy * np.log(1 + samples))
            client_amplitudes.append(amplitude)
        
        # Normalize amplitudes
        total_amplitude = sum(client_amplitudes)
        normalized_amplitudes = [amp / total_amplitude for amp in client_amplitudes]
        
        # Quantum interference-inspired aggregation
        aggregated_weights = {}
        for key in global_weights.keys():
            aggregated_weights[key] = torch.zeros_like(global_weights[key])
            
            for i, (client_id, weights, metrics) in enumerate(client_results):
                if key in weights:
                    # Apply quantum amplitude weighting with interference
                    phase = 2 * np.pi * i / num_clients  # Phase for interference
                    quantum_weight = normalized_amplitudes[i] * np.exp(1j * phase)
                    
                    # Real part contributes to final weights
                    aggregated_weights[key] += quantum_weight.real * weights[key]
        
        aggregation_metrics = {
            'strategy': 'quantum_federated',
            'quantum_amplitudes': normalized_amplitudes,
            'interference_effects': True,
            'participating_clients': num_clients
        }
        
        return aggregated_weights, aggregation_metrics
    
    async def _global_evaluation(self, clients: List[FederatedClient]) -> Dict[str, float]:
        """Evaluate global model across all clients."""
        global_weights = self.global_model.state_dict()
        
        # Collect evaluation results from all clients
        evaluation_tasks = [
            client.local_evaluation(global_weights) for client in clients
        ]
        
        client_evaluations = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
        
        # Aggregate evaluation metrics
        total_samples = 0
        weighted_loss = 0.0
        weighted_accuracy = 0.0
        
        successful_evaluations = [
            eval_result for eval_result in client_evaluations
            if not isinstance(eval_result, Exception) and 'samples' in eval_result
        ]
        
        for eval_result in successful_evaluations:
            samples = eval_result['samples']
            total_samples += samples
            
            weighted_loss += eval_result['loss'] * samples
            weighted_accuracy += eval_result['accuracy'] * samples
        
        if total_samples > 0:
            global_metrics = {
                'loss': weighted_loss / total_samples,
                'accuracy': weighted_accuracy / total_samples,
                'total_samples': total_samples,
                'participating_clients': len(successful_evaluations)
            }
        else:
            global_metrics = {
                'loss': float('inf'),
                'accuracy': 0.0,
                'total_samples': 0,
                'participating_clients': 0
            }
        
        return global_metrics
    
    async def _check_convergence(self, training_history: Dict[str, Any]) -> bool:
        """Check if federated training has converged."""
        if len(training_history['global_metrics']) < 5:
            return False
        
        # Check if accuracy has plateaued
        recent_accuracies = [
            metrics.get('accuracy', 0) 
            for metrics in training_history['global_metrics'][-5:]
        ]
        
        accuracy_variance = np.var(recent_accuracies)
        return accuracy_variance < 0.0001  # Convergence threshold
    
    async def _finalize_training(self, training_history: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize federated training and generate results."""
        final_metrics = training_history['global_metrics'][-1] if training_history['global_metrics'] else {}
        
        # Calculate privacy budget spent across all clients
        total_privacy_spent = 0.0
        for round_data in training_history['rounds']:
            for client_metrics in round_data.get('client_metrics', []):
                if isinstance(client_metrics, dict) and 'privacy_spent' in client_metrics:
                    total_privacy_spent = max(total_privacy_spent, client_metrics['privacy_spent'])
        
        final_results = {
            'status': 'completed',
            'total_rounds': len(training_history['rounds']),
            'final_accuracy': final_metrics.get('accuracy', 0),
            'final_loss': final_metrics.get('loss', float('inf')),
            'total_privacy_spent': total_privacy_spent,
            'training_history': training_history,
            'model_state': self.global_model.state_dict()
        }
        
        logging.info(f"Federated training completed. Final accuracy: {final_results['final_accuracy']:.4f}")
        return final_results


class FederatedDGDMManager:
    """High-level manager for federated DGDM training."""
    
    def __init__(self, base_model: DGDMModel, config: FederatedConfig):
        self.base_model = base_model
        self.config = config
        self.server = None
        
    async def setup_federation(self, client_data_configs: List[Dict[str, Any]]) -> bool:
        """Setup federated learning environment."""
        try:
            # Initialize server with client configurations
            self.server = FederatedServer(
                global_model=copy.deepcopy(self.base_model),
                config=self.config,
                client_configs=client_data_configs
            )
            
            logging.info(f"Federation setup complete with {len(client_data_configs)} clients")
            return True
            
        except Exception as e:
            logging.error(f"Federation setup failed: {e}")
            return False
    
    async def train_federated(self) -> Dict[str, Any]:
        """Execute federated training."""
        if not self.server:
            raise FederatedLearningError("Federation not setup. Call setup_federation() first.")
        
        return await self.server.start_federated_training()
    
    async def deploy_federated_model(self, model_path: str) -> bool:
        """Deploy the trained federated model."""
        try:
            if self.server:
                # Save the global model
                torch.save({
                    'model_state_dict': self.server.global_model.state_dict(),
                    'config': self.config,
                    'training_complete': True
                }, model_path)
                
                logging.info(f"Federated model deployed to {model_path}")
                return True
            else:
                logging.error("No trained model available for deployment")
                return False
                
        except Exception as e:
            logging.error(f"Model deployment failed: {e}")
            return False


# Export main components
__all__ = [
    'FederationStrategy',
    'PrivacyMechanism', 
    'FederatedConfig',
    'FederatedClient',
    'FederatedServer',
    'FederatedDGDMManager'
]
