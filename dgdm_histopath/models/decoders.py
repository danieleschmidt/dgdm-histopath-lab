"""
Task-specific decoders for DGDM model.

Implements classification and regression heads for various
histopathology analysis tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import numpy as np


class ClassificationHead(nn.Module):
    """
    Classification head for histopathology tasks.
    
    Supports multi-class classification with optional class weighting
    and confidence estimation.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_batch_norm: bool = True,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0
    ):
        """
        Initialize classification head.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability
            activation: Activation function
            use_batch_norm: Whether to use batch normalization
            class_weights: Optional class weights for loss computation
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        
        if hidden_dims is None:
            hidden_dims = [input_dim // 2]
            
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()
            
        # Build layers
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
            
        # Final classification layer
        layers.append(nn.Linear(dims[-1], num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        # Store class weights
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classification head.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Classification logits [batch_size, num_classes]
        """
        return self.classifier(x)
        
    def compute_loss(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute classification loss.
        
        Args:
            logits: Model logits [batch_size, num_classes]
            targets: Target labels [batch_size]
            
        Returns:
            Classification loss
        """
        if self.label_smoothing > 0:
            # Label smoothing
            log_probs = F.log_softmax(logits, dim=-1)
            targets_smooth = torch.zeros_like(log_probs).scatter_(
                1, targets.unsqueeze(1), 1 - self.label_smoothing
            )
            targets_smooth += self.label_smoothing / self.num_classes
            loss = -(targets_smooth * log_probs).sum(dim=-1).mean()
        else:
            # Standard cross-entropy
            loss = F.cross_entropy(logits, targets, weight=self.class_weights)
            
        return loss
        
    def predict(self, x: torch.Tensor, return_probs: bool = False) -> torch.Tensor:
        """
        Make predictions.
        
        Args:
            x: Input features
            return_probs: Whether to return probabilities instead of class indices
            
        Returns:
            Predictions (class indices or probabilities)
        """
        with torch.no_grad():
            logits = self.forward(x)
            
            if return_probs:
                return F.softmax(logits, dim=-1)
            else:
                return torch.argmax(logits, dim=-1)


class RegressionHead(nn.Module):
    """
    Regression head for continuous value prediction.
    
    Supports multi-target regression with optional output scaling
    and uncertainty estimation.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_targets: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_batch_norm: bool = True,
        output_activation: Optional[str] = None,
        predict_uncertainty: bool = False
    ):
        """
        Initialize regression head.
        
        Args:
            input_dim: Input feature dimension
            num_targets: Number of regression targets
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability
            activation: Activation function
            use_batch_norm: Whether to use batch normalization
            output_activation: Output activation function
            predict_uncertainty: Whether to predict uncertainty estimates
        """
        super().__init__()
        
        self.num_targets = num_targets
        self.predict_uncertainty = predict_uncertainty
        
        if hidden_dims is None:
            hidden_dims = [input_dim // 2]
            
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()
            
        # Output activation
        if output_activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        elif output_activation == "tanh":
            self.output_activation = nn.Tanh()
        elif output_activation == "softplus":
            self.output_activation = nn.Softplus()
        else:
            self.output_activation = nn.Identity()
            
        # Build layers
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
            
        self.feature_layers = nn.Sequential(*layers)
        
        # Output layers
        self.mean_head = nn.Linear(dims[-1], num_targets)
        
        if predict_uncertainty:
            self.var_head = nn.Linear(dims[-1], num_targets)
        else:
            self.var_head = None
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through regression head.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Regression outputs [batch_size, num_targets]
            If predict_uncertainty=True, returns dict with 'mean' and 'var'
        """
        features = self.feature_layers(x)
        
        # Predict mean values
        mean = self.mean_head(features)
        mean = self.output_activation(mean)
        
        if self.predict_uncertainty:
            # Predict log variance (for numerical stability)
            log_var = self.var_head(features)
            var = torch.exp(log_var)
            
            return {
                'mean': mean,
                'var': var,
                'log_var': log_var
            }
        else:
            return mean
            
    def compute_loss(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        loss_type: str = "mse"
    ) -> torch.Tensor:
        """
        Compute regression loss.
        
        Args:
            predictions: Model predictions
            targets: Target values [batch_size, num_targets]
            loss_type: Type of loss ("mse", "mae", "huber", "gaussian_nll")
            
        Returns:
            Regression loss
        """
        if isinstance(predictions, dict):
            # Uncertainty prediction
            mean = predictions['mean']
            var = predictions['var']
            
            if loss_type == "gaussian_nll":
                # Gaussian negative log-likelihood
                loss = 0.5 * (torch.log(var) + (targets - mean) ** 2 / var)
                return loss.mean()
            else:
                # Use mean for other loss types
                predictions = mean
                
        if loss_type == "mse":
            loss = F.mse_loss(predictions, targets)
        elif loss_type == "mae":
            loss = F.l1_loss(predictions, targets)
        elif loss_type == "huber":
            loss = F.huber_loss(predictions, targets)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
            
        return loss
        
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions.
        
        Args:
            x: Input features
            
        Returns:
            Regression predictions
        """
        with torch.no_grad():
            outputs = self.forward(x)
            
            if isinstance(outputs, dict):
                return outputs['mean']
            else:
                return outputs


class SurvivalHead(nn.Module):
    """
    Survival analysis head for time-to-event prediction.
    
    Implements Cox proportional hazards model and discrete-time survival.
    """
    
    def __init__(
        self,
        input_dim: int,
        time_bins: int = 100,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        survival_type: str = "cox"
    ):
        """
        Initialize survival head.
        
        Args:
            input_dim: Input feature dimension
            time_bins: Number of discrete time bins
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability
            activation: Activation function
            survival_type: Type of survival model ("cox", "discrete")
        """
        super().__init__()
        
        self.time_bins = time_bins
        self.survival_type = survival_type
        
        if hidden_dims is None:
            hidden_dims = [input_dim // 2]
            
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
            
        # Build feature layers
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
            
        self.feature_layers = nn.Sequential(*layers)
        
        # Survival-specific output layers
        if survival_type == "cox":
            # Cox model: single hazard ratio
            self.hazard_head = nn.Linear(dims[-1], 1)
        elif survival_type == "discrete":
            # Discrete-time: probability for each time bin
            self.survival_head = nn.Linear(dims[-1], time_bins)
        else:
            raise ValueError(f"Unknown survival type: {survival_type}")
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through survival head.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Survival predictions
        """
        features = self.feature_layers(x)
        
        if self.survival_type == "cox":
            # Cox hazard ratio
            hazard_ratio = self.hazard_head(features)
            return hazard_ratio
        elif self.survival_type == "discrete":
            # Discrete survival probabilities
            logits = self.survival_head(features)
            survival_probs = torch.sigmoid(logits)
            return survival_probs
            
    def compute_loss(
        self,
        predictions: torch.Tensor,
        times: torch.Tensor,
        events: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute survival loss.
        
        Args:
            predictions: Model predictions
            times: Survival/censoring times
            events: Event indicators (1 = event, 0 = censored)
            
        Returns:
            Survival loss
        """
        if self.survival_type == "cox":
            # Cox partial likelihood
            return self._cox_loss(predictions, times, events)
        elif self.survival_type == "discrete":
            # Discrete-time likelihood
            return self._discrete_survival_loss(predictions, times, events)
            
    def _cox_loss(
        self,
        hazard_ratios: torch.Tensor,
        times: torch.Tensor,
        events: torch.Tensor
    ) -> torch.Tensor:
        """Compute Cox partial likelihood loss."""
        # Sort by survival time (descending)
        sorted_indices = torch.argsort(times, descending=True)
        sorted_hazards = hazard_ratios[sorted_indices]
        sorted_events = events[sorted_indices]
        
        # Compute partial likelihood
        exp_hazards = torch.exp(sorted_hazards)
        cumsum_exp_hazards = torch.cumsum(exp_hazards, dim=0)
        
        # Log partial likelihood for observed events
        log_likelihood = sorted_hazards - torch.log(cumsum_exp_hazards)
        
        # Only include observed events
        observed_mask = sorted_events.bool()
        partial_log_likelihood = log_likelihood[observed_mask].sum()
        
        # Return negative log likelihood (for minimization)
        return -partial_log_likelihood / observed_mask.sum()
        
    def _discrete_survival_loss(
        self,
        survival_probs: torch.Tensor,
        times: torch.Tensor,
        events: torch.Tensor
    ) -> torch.Tensor:
        """Compute discrete-time survival loss."""
        batch_size = survival_probs.size(0)
        
        # Convert continuous times to discrete bins
        max_time = times.max().item()
        time_bins = torch.linspace(0, max_time, self.time_bins + 1)
        discrete_times = torch.bucketize(times, time_bins) - 1
        discrete_times = torch.clamp(discrete_times, 0, self.time_bins - 1)
        
        # Compute likelihood
        log_likelihood = 0
        
        for i in range(batch_size):
            t = discrete_times[i].item()
            event = events[i].item()
            
            # Survival probability up to time t
            survival_t = torch.prod(survival_probs[i, :t+1])
            
            if event == 1:  # Observed event
                # Hazard at time t
                if t < self.time_bins - 1:
                    hazard_t = 1 - survival_probs[i, t]
                    likelihood = survival_t * hazard_t
                else:
                    likelihood = survival_t
            else:  # Censored
                likelihood = survival_t
                
            log_likelihood += torch.log(likelihood + 1e-8)
            
        return -log_likelihood / batch_size


class MultiTaskHead(nn.Module):
    """
    Multi-task head that combines classification and regression.
    
    Allows joint training on multiple histopathology tasks with
    automatic task weighting and gradient balancing.
    """
    
    def __init__(
        self,
        input_dim: int,
        classification_tasks: Optional[List[int]] = None,
        regression_tasks: Optional[List[int]] = None,
        shared_hidden_dims: Optional[List[int]] = None,
        task_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        use_uncertainty_weighting: bool = True
    ):
        """
        Initialize multi-task head.
        
        Args:
            input_dim: Input feature dimension
            classification_tasks: List of class counts for classification tasks
            regression_tasks: List of target counts for regression tasks
            shared_hidden_dims: Hidden dimensions for shared layers
            task_hidden_dims: Hidden dimensions for task-specific layers
            dropout: Dropout probability
            use_uncertainty_weighting: Whether to use uncertainty-based weighting
        """
        super().__init__()
        
        self.classification_tasks = classification_tasks or []
        self.regression_tasks = regression_tasks or []
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        if shared_hidden_dims is None:
            shared_hidden_dims = [input_dim // 2]
            
        if task_hidden_dims is None:
            task_hidden_dims = [shared_hidden_dims[-1] // 2]
            
        # Shared feature layers
        shared_layers = []
        dims = [input_dim] + shared_hidden_dims
        
        for i in range(len(dims) - 1):
            shared_layers.append(nn.Linear(dims[i], dims[i + 1]))
            shared_layers.append(nn.GELU())
            shared_layers.append(nn.Dropout(dropout))
            
        self.shared_layers = nn.Sequential(*shared_layers)
        
        # Task-specific heads
        self.classification_heads = nn.ModuleList()
        for num_classes in self.classification_tasks:
            head = ClassificationHead(
                input_dim=shared_hidden_dims[-1],
                num_classes=num_classes,
                hidden_dims=task_hidden_dims,
                dropout=dropout
            )
            self.classification_heads.append(head)
            
        self.regression_heads = nn.ModuleList()
        for num_targets in self.regression_tasks:
            head = RegressionHead(
                input_dim=shared_hidden_dims[-1],
                num_targets=num_targets,
                hidden_dims=task_hidden_dims,
                dropout=dropout
            )
            self.regression_heads.append(head)
            
        # Uncertainty-based task weighting
        if use_uncertainty_weighting:
            total_tasks = len(self.classification_tasks) + len(self.regression_tasks)
            self.log_vars = nn.Parameter(torch.zeros(total_tasks))
        else:
            self.log_vars = None
            
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-task head.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Dictionary with predictions for each task
        """
        shared_features = self.shared_layers(x)
        outputs = {}
        
        # Classification tasks
        for i, head in enumerate(self.classification_heads):
            logits = head(shared_features)
            outputs[f'classification_{i}'] = logits
            outputs[f'classification_probs_{i}'] = F.softmax(logits, dim=-1)
            
        # Regression tasks
        for i, head in enumerate(self.regression_heads):
            predictions = head(shared_features)
            outputs[f'regression_{i}'] = predictions
            
        return outputs
        
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute multi-task loss with automatic weighting.
        
        Args:
            predictions: Model predictions
            targets: Target values for each task
            
        Returns:
            Weighted multi-task loss
        """
        losses = []
        task_idx = 0
        
        # Classification losses
        for i in range(len(self.classification_tasks)):
            if f'classification_targets_{i}' in targets:
                logits = predictions[f'classification_{i}']
                targets_i = targets[f'classification_targets_{i}']
                loss = self.classification_heads[i].compute_loss(logits, targets_i)
                
                if self.use_uncertainty_weighting:
                    precision = torch.exp(-self.log_vars[task_idx])
                    weighted_loss = precision * loss + self.log_vars[task_idx]
                else:
                    weighted_loss = loss
                    
                losses.append(weighted_loss)
                task_idx += 1
                
        # Regression losses
        for i in range(len(self.regression_tasks)):
            if f'regression_targets_{i}' in targets:
                preds = predictions[f'regression_{i}']
                targets_i = targets[f'regression_targets_{i}']
                loss = self.regression_heads[i].compute_loss(preds, targets_i)
                
                if self.use_uncertainty_weighting:
                    precision = torch.exp(-self.log_vars[task_idx])
                    weighted_loss = precision * loss + self.log_vars[task_idx]
                else:
                    weighted_loss = loss
                    
                losses.append(weighted_loss)
                task_idx += 1
                
        return sum(losses) if losses else torch.tensor(0.0)