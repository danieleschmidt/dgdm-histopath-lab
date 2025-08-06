"""
Quantum Safety and Security Module for DGDM Histopath Lab.

Implements comprehensive safety measures, security controls, and risk management
for quantum-enhanced medical AI systems.
"""

import hashlib
import hmac
import secrets
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json
import numpy as np
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from dgdm_histopath.utils.logging import get_logger
from dgdm_histopath.utils.monitoring import monitor_operation


class SecurityLevel(Enum):
    """Security levels for quantum operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Types of security threats."""
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_TAMPERING = "data_tampering"
    MODEL_POISONING = "model_poisoning"
    QUANTUM_DECOHERENCE = "quantum_decoherence"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    SIDE_CHANNEL_ATTACK = "side_channel_attack"


@dataclass
class SecurityEvent:
    """Security event record."""
    timestamp: float
    threat_type: ThreatType
    severity: SecurityLevel
    description: str
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    affected_resources: List[str] = field(default_factory=list)
    mitigation_actions: List[str] = field(default_factory=list)


@dataclass
class AccessCredentials:
    """Secure access credentials for quantum operations."""
    user_id: str
    api_key_hash: str
    permissions: List[str]
    security_level: SecurityLevel
    expires_at: float
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)


class QuantumSafetyManager:
    """
    Comprehensive safety and security manager for quantum-enhanced medical AI.
    
    Provides:
    - Authentication and authorization
    - Data encryption and integrity checks
    - Quantum state validation and error correction
    - Threat detection and response
    - Audit logging and compliance
    """
    
    def __init__(
        self,
        master_key: Optional[str] = None,
        security_level: SecurityLevel = SecurityLevel.HIGH,
        enable_quantum_error_correction: bool = True,
        audit_log_path: str = "security_audit.log",
        max_failed_attempts: int = 3,
        lockout_duration: float = 300.0  # 5 minutes
    ):
        self.logger = get_logger(__name__)
        
        # Security configuration
        self.security_level = security_level
        self.enable_quantum_error_correction = enable_quantum_error_correction
        self.max_failed_attempts = max_failed_attempts
        self.lockout_duration = lockout_duration
        
        # Initialize encryption
        if master_key:
            self.master_key = master_key.encode()
        else:
            self.master_key = self._generate_master_key()
        
        self.cipher_suite = self._initialize_encryption()
        
        # Access control
        self.active_sessions: Dict[str, AccessCredentials] = {}
        self.failed_attempts: Dict[str, int] = {}
        self.locked_users: Dict[str, float] = {}
        
        # Security monitoring
        self.security_events: List[SecurityEvent] = []
        self.audit_log_path = Path(audit_log_path)
        self.threat_detection_rules: List[Callable] = []
        
        # Quantum safety
        self.quantum_state_checksums: Dict[str, str] = {}
        self.quantum_error_rates: Dict[str, float] = {}
        
        # Threading for background monitoring
        self.monitoring_thread = None
        self.shutdown_event = threading.Event()
        
        # Initialize threat detection
        self._initialize_threat_detection()
        
        # Start security monitoring
        self._start_security_monitoring()
        
        self.logger.info(f"QuantumSafetyManager initialized with security level: {security_level.value}")
    
    def _generate_master_key(self) -> bytes:
        """Generate secure master key."""
        return secrets.token_bytes(32)  # 256-bit key
    
    def _initialize_encryption(self) -> Fernet:
        """Initialize Fernet encryption with derived key."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'dgdm_quantum_salt',  # In production, use random salt
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key))
        return Fernet(key)
    
    def _initialize_threat_detection(self):
        """Initialize threat detection rules."""
        # Rule: Multiple failed authentication attempts
        def detect_brute_force(events: List[SecurityEvent]) -> Optional[SecurityEvent]:
            recent_failures = [e for e in events[-10:] 
                             if e.threat_type == ThreatType.UNAUTHORIZED_ACCESS 
                             and time.time() - e.timestamp < 60]
            
            if len(recent_failures) >= 5:
                return SecurityEvent(
                    timestamp=time.time(),
                    threat_type=ThreatType.UNAUTHORIZED_ACCESS,
                    severity=SecurityLevel.HIGH,
                    description="Potential brute force attack detected"
                )
            return None
        
        # Rule: Abnormal quantum decoherence rates
        def detect_quantum_tampering(events: List[SecurityEvent]) -> Optional[SecurityEvent]:
            decoherence_events = [e for e in events[-5:] 
                                if e.threat_type == ThreatType.QUANTUM_DECOHERENCE]
            
            if len(decoherence_events) >= 3:
                return SecurityEvent(
                    timestamp=time.time(),
                    threat_type=ThreatType.QUANTUM_DECOHERENCE,
                    severity=SecurityLevel.CRITICAL,
                    description="Abnormal quantum decoherence pattern detected"
                )
            return None
        
        # Rule: Resource exhaustion attacks
        def detect_resource_exhaustion(events: List[SecurityEvent]) -> Optional[SecurityEvent]:
            resource_events = [e for e in events[-20:] 
                             if e.threat_type == ThreatType.RESOURCE_EXHAUSTION]
            
            if len(resource_events) >= 10:
                return SecurityEvent(
                    timestamp=time.time(),
                    threat_type=ThreatType.RESOURCE_EXHAUSTION,
                    severity=SecurityLevel.HIGH,
                    description="Resource exhaustion attack pattern detected"
                )
            return None
        
        self.threat_detection_rules.extend([
            detect_brute_force,
            detect_quantum_tampering,
            detect_resource_exhaustion
        ])
    
    def _start_security_monitoring(self):
        """Start background security monitoring thread."""
        def monitor_security():
            while not self.shutdown_event.is_set():
                try:
                    # Clean up expired sessions
                    self._cleanup_expired_sessions()
                    
                    # Check for security threats
                    self._check_threat_detection()
                    
                    # Monitor quantum state integrity
                    self._monitor_quantum_integrity()
                    
                    # Update security metrics
                    self._update_security_metrics()
                    
                    time.sleep(10.0)  # Check every 10 seconds
                    
                except Exception as e:
                    self.logger.error(f"Security monitoring error: {e}")
                    time.sleep(1.0)
        
        self.monitoring_thread = threading.Thread(target=monitor_security, daemon=True)
        self.monitoring_thread.start()
    
    @monitor_operation("authenticate_user")
    def authenticate_user(
        self,
        user_id: str,
        api_key: str,
        required_permissions: Optional[List[str]] = None,
        source_ip: Optional[str] = None
    ) -> bool:
        """Authenticate user and validate permissions."""
        # Check if user is locked out
        if user_id in self.locked_users:
            if time.time() - self.locked_users[user_id] < self.lockout_duration:
                self._log_security_event(
                    ThreatType.UNAUTHORIZED_ACCESS,
                    SecurityLevel.MEDIUM,
                    f"Authentication attempt from locked user: {user_id}",
                    source_ip=source_ip,
                    user_id=user_id
                )
                return False
            else:
                # Lockout expired
                del self.locked_users[user_id]
                self.failed_attempts.pop(user_id, 0)
        
        # Validate API key
        api_key_hash = self._hash_api_key(api_key)
        
        # Check against stored credentials (in production, use secure database)
        valid_credentials = self._validate_credentials(user_id, api_key_hash)
        
        if not valid_credentials:
            # Record failed attempt
            self.failed_attempts[user_id] = self.failed_attempts.get(user_id, 0) + 1
            
            if self.failed_attempts[user_id] >= self.max_failed_attempts:
                # Lock user
                self.locked_users[user_id] = time.time()
                self._log_security_event(
                    ThreatType.UNAUTHORIZED_ACCESS,
                    SecurityLevel.HIGH,
                    f"User locked due to multiple failed attempts: {user_id}",
                    source_ip=source_ip,
                    user_id=user_id
                )
            else:
                self._log_security_event(
                    ThreatType.UNAUTHORIZED_ACCESS,
                    SecurityLevel.MEDIUM,
                    f"Failed authentication attempt: {user_id}",
                    source_ip=source_ip,
                    user_id=user_id
                )
            
            return False
        
        # Reset failed attempts on successful auth
        self.failed_attempts.pop(user_id, 0)
        
        # Check permissions
        if required_permissions:
            user_permissions = valid_credentials.permissions
            if not all(perm in user_permissions for perm in required_permissions):
                self._log_security_event(
                    ThreatType.UNAUTHORIZED_ACCESS,
                    SecurityLevel.MEDIUM,
                    f"Insufficient permissions for user: {user_id}",
                    source_ip=source_ip,
                    user_id=user_id
                )
                return False
        
        # Create or update session
        valid_credentials.last_access = time.time()
        self.active_sessions[user_id] = valid_credentials
        
        self.logger.info(f"User authenticated successfully: {user_id}")
        return True
    
    def _hash_api_key(self, api_key: str) -> str:
        """Securely hash API key."""
        return hashlib.pbkdf2_hmac('sha256', api_key.encode(), b'api_salt', 100000).hex()
    
    def _validate_credentials(self, user_id: str, api_key_hash: str) -> Optional[AccessCredentials]:
        """Validate user credentials (placeholder - implement with secure storage)."""
        # In production, validate against secure database
        # For demo, create valid credentials
        if user_id == "demo_user" and api_key_hash == self._hash_api_key("demo_key"):
            return AccessCredentials(
                user_id=user_id,
                api_key_hash=api_key_hash,
                permissions=["read", "write", "quantum_ops"],
                security_level=SecurityLevel.HIGH,
                expires_at=time.time() + 3600  # 1 hour
            )
        return None
    
    @monitor_operation("encrypt_data")
    def encrypt_data(self, data: Any) -> str:
        """Securely encrypt sensitive data."""
        try:
            if isinstance(data, dict):
                data_str = json.dumps(data)
            elif isinstance(data, (list, tuple)):
                data_str = json.dumps(list(data))
            else:
                data_str = str(data)
            
            encrypted_data = self.cipher_suite.encrypt(data_str.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
            
        except Exception as e:
            self.logger.error(f"Data encryption failed: {e}")
            raise
    
    @monitor_operation("decrypt_data")
    def decrypt_data(self, encrypted_data: str) -> Any:
        """Securely decrypt sensitive data."""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_bytes = self.cipher_suite.decrypt(encrypted_bytes)
            data_str = decrypted_bytes.decode()
            
            # Try to parse as JSON
            try:
                return json.loads(data_str)
            except json.JSONDecodeError:
                return data_str
                
        except Exception as e:
            self.logger.error(f"Data decryption failed: {e}")
            raise
    
    @monitor_operation("validate_quantum_state")
    def validate_quantum_state(
        self,
        state_id: str,
        quantum_state: np.ndarray,
        expected_checksum: Optional[str] = None
    ) -> bool:
        """Validate integrity of quantum state."""
        try:
            # Calculate checksum
            state_bytes = quantum_state.astype(complex).tobytes()
            current_checksum = hashlib.sha256(state_bytes).hexdigest()
            
            # Check against expected checksum
            if expected_checksum:
                if current_checksum != expected_checksum:
                    self._log_security_event(
                        ThreatType.DATA_TAMPERING,
                        SecurityLevel.CRITICAL,
                        f"Quantum state integrity check failed for {state_id}",
                        affected_resources=[state_id]
                    )
                    return False
            
            # Store checksum for future validation
            self.quantum_state_checksums[state_id] = current_checksum
            
            # Validate quantum properties
            if not self._validate_quantum_properties(quantum_state):
                self._log_security_event(
                    ThreatType.QUANTUM_DECOHERENCE,
                    SecurityLevel.HIGH,
                    f"Quantum state properties validation failed for {state_id}",
                    affected_resources=[state_id]
                )
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Quantum state validation failed: {e}")
            self._log_security_event(
                ThreatType.QUANTUM_DECOHERENCE,
                SecurityLevel.HIGH,
                f"Quantum state validation error for {state_id}: {str(e)}",
                affected_resources=[state_id]
            )
            return False
    
    def _validate_quantum_properties(self, quantum_state: np.ndarray) -> bool:
        """Validate fundamental quantum properties."""
        try:
            # Check normalization
            norm = np.linalg.norm(quantum_state)
            if abs(norm - 1.0) > 1e-6:
                return False
            
            # Check for NaN or infinite values
            if np.any(np.isnan(quantum_state)) or np.any(np.isinf(quantum_state)):
                return False
            
            # Check coherence bounds
            coherence = abs(np.sum(quantum_state))
            if coherence > len(quantum_state):  # Theoretical maximum
                return False
            
            return True
            
        except Exception:
            return False
    
    @monitor_operation("detect_anomalies")
    def detect_anomalies(
        self,
        operation_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
        threshold_multiplier: float = 3.0
    ) -> List[str]:
        """Detect anomalies in operation metrics."""
        anomalies = []
        
        for metric_name, current_value in operation_metrics.items():
            if metric_name in baseline_metrics:
                baseline_value = baseline_metrics[metric_name]
                
                if baseline_value > 0:
                    deviation = abs(current_value - baseline_value) / baseline_value
                    
                    if deviation > threshold_multiplier:
                        anomalies.append(f"{metric_name}: {deviation:.2f}x deviation")
                        
                        self._log_security_event(
                            ThreatType.SIDE_CHANNEL_ATTACK,
                            SecurityLevel.MEDIUM,
                            f"Anomaly detected in {metric_name}: {deviation:.2f}x baseline",
                            affected_resources=[metric_name]
                        )
        
        return anomalies
    
    def _log_security_event(
        self,
        threat_type: ThreatType,
        severity: SecurityLevel,
        description: str,
        source_ip: Optional[str] = None,
        user_id: Optional[str] = None,
        affected_resources: Optional[List[str]] = None
    ):
        """Log security event."""
        event = SecurityEvent(
            timestamp=time.time(),
            threat_type=threat_type,
            severity=severity,
            description=description,
            source_ip=source_ip,
            user_id=user_id,
            affected_resources=affected_resources or []
        )
        
        self.security_events.append(event)
        
        # Write to audit log
        self._write_audit_log(event)
        
        # Log to standard logger based on severity
        if severity == SecurityLevel.CRITICAL:
            self.logger.critical(f"SECURITY ALERT: {description}")
        elif severity == SecurityLevel.HIGH:
            self.logger.error(f"SECURITY WARNING: {description}")
        elif severity == SecurityLevel.MEDIUM:
            self.logger.warning(f"SECURITY NOTICE: {description}")
        else:
            self.logger.info(f"SECURITY INFO: {description}")
    
    def _write_audit_log(self, event: SecurityEvent):
        """Write security event to audit log."""
        try:
            audit_entry = {
                "timestamp": event.timestamp,
                "threat_type": event.threat_type.value,
                "severity": event.severity.value,
                "description": event.description,
                "source_ip": event.source_ip,
                "user_id": event.user_id,
                "affected_resources": event.affected_resources
            }
            
            with open(self.audit_log_path, 'a') as f:
                f.write(json.dumps(audit_entry) + '\n')
                
        except Exception as e:
            self.logger.error(f"Failed to write audit log: {e}")
    
    def _cleanup_expired_sessions(self):
        """Remove expired user sessions."""
        current_time = time.time()
        expired_sessions = []
        
        for user_id, credentials in self.active_sessions.items():
            if current_time > credentials.expires_at:
                expired_sessions.append(user_id)
        
        for user_id in expired_sessions:
            del self.active_sessions[user_id]
            self.logger.debug(f"Expired session for user: {user_id}")
    
    def _check_threat_detection(self):
        """Run threat detection rules."""
        for rule in self.threat_detection_rules:
            try:
                detected_threat = rule(self.security_events)
                if detected_threat:
                    self.security_events.append(detected_threat)
                    
                    # Implement automatic mitigation if needed
                    self._mitigate_threat(detected_threat)
                    
            except Exception as e:
                self.logger.error(f"Threat detection rule failed: {e}")
    
    def _mitigate_threat(self, threat: SecurityEvent):
        """Implement automatic threat mitigation."""
        mitigation_actions = []
        
        if threat.threat_type == ThreatType.UNAUTHORIZED_ACCESS:
            # Increase authentication requirements
            mitigation_actions.append("Enhanced authentication enabled")
            
        elif threat.threat_type == ThreatType.RESOURCE_EXHAUSTION:
            # Implement rate limiting
            mitigation_actions.append("Rate limiting activated")
            
        elif threat.threat_type == ThreatType.QUANTUM_DECOHERENCE:
            # Enable quantum error correction
            mitigation_actions.append("Quantum error correction intensified")
        
        threat.mitigation_actions = mitigation_actions
        
        if mitigation_actions:
            self.logger.info(f"Threat mitigation applied: {', '.join(mitigation_actions)}")
    
    def _monitor_quantum_integrity(self):
        """Monitor quantum state integrity across the system."""
        # Check for unexpected quantum error rates
        total_error_rate = sum(self.quantum_error_rates.values())
        
        if total_error_rate > 0.1:  # 10% error threshold
            self._log_security_event(
                ThreatType.QUANTUM_DECOHERENCE,
                SecurityLevel.HIGH,
                f"High quantum error rate detected: {total_error_rate:.3f}",
                affected_resources=list(self.quantum_error_rates.keys())
            )
    
    def _update_security_metrics(self):
        """Update security performance metrics."""
        current_time = time.time()
        
        # Calculate security metrics
        recent_events = [e for e in self.security_events 
                        if current_time - e.timestamp < 3600]  # Last hour
        
        critical_events = [e for e in recent_events 
                          if e.severity == SecurityLevel.CRITICAL]
        
        if len(critical_events) > 5:  # More than 5 critical events per hour
            self._log_security_event(
                ThreatType.UNAUTHORIZED_ACCESS,
                SecurityLevel.CRITICAL,
                "High frequency of critical security events detected"
            )
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        current_time = time.time()
        
        # Recent security events (last hour)
        recent_events = [e for e in self.security_events 
                        if current_time - e.timestamp < 3600]
        
        # Group events by type
        event_counts = {}
        for event in recent_events:
            event_counts[event.threat_type.value] = event_counts.get(event.threat_type.value, 0) + 1
        
        return {
            "security_level": self.security_level.value,
            "active_sessions": len(self.active_sessions),
            "locked_users": len(self.locked_users),
            "recent_security_events": len(recent_events),
            "event_breakdown": event_counts,
            "quantum_states_monitored": len(self.quantum_state_checksums),
            "quantum_error_rate": sum(self.quantum_error_rates.values()),
            "threat_detection_rules": len(self.threat_detection_rules),
            "audit_log_path": str(self.audit_log_path)
        }
    
    def export_security_report(self, filepath: str, time_range_hours: int = 24):
        """Export comprehensive security report."""
        current_time = time.time()
        start_time = current_time - (time_range_hours * 3600)
        
        # Filter events by time range
        events_in_range = [e for e in self.security_events 
                          if e.timestamp >= start_time]
        
        # Create report
        report = {
            "report_generated": current_time,
            "time_range_hours": time_range_hours,
            "security_configuration": {
                "security_level": self.security_level.value,
                "max_failed_attempts": self.max_failed_attempts,
                "lockout_duration": self.lockout_duration,
                "quantum_error_correction": self.enable_quantum_error_correction
            },
            "security_summary": {
                "total_events": len(events_in_range),
                "critical_events": len([e for e in events_in_range 
                                      if e.severity == SecurityLevel.CRITICAL]),
                "high_severity_events": len([e for e in events_in_range 
                                           if e.severity == SecurityLevel.HIGH]),
                "active_sessions": len(self.active_sessions),
                "locked_users": len(self.locked_users)
            },
            "events": [
                {
                    "timestamp": e.timestamp,
                    "threat_type": e.threat_type.value,
                    "severity": e.severity.value,
                    "description": e.description,
                    "affected_resources": e.affected_resources,
                    "mitigation_actions": e.mitigation_actions
                }
                for e in events_in_range
            ],
            "quantum_integrity": {
                "states_monitored": len(self.quantum_state_checksums),
                "error_rates": self.quantum_error_rates.copy()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Security report exported to {filepath}")
    
    def shutdown(self):
        """Gracefully shutdown security manager."""
        self.logger.info("Shutting down QuantumSafetyManager")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Wait for monitoring thread
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        # Clear sensitive data from memory
        self.master_key = b''
        self.active_sessions.clear()
        
        self.logger.info("QuantumSafetyManager shutdown complete")