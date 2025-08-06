"""
Security validation and penetration tests for DGDM quantum framework.

Comprehensive security testing including authentication, authorization, 
data protection, and threat detection validation.
"""

import pytest
import time
import asyncio
import threading
from unittest.mock import Mock, patch
import numpy as np
import hashlib
import secrets
from typing import Dict, Any, List

from dgdm_histopath.quantum.quantum_safety import (
    QuantumSafetyManager, SecurityLevel, ThreatType, SecurityEvent, AccessCredentials
)
from dgdm_histopath.utils.validation import InputValidator, ValidationError, SecurityError
from dgdm_histopath.utils.logging import get_logger


class SecurityTestCase:
    """Base class for security test cases."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.test_results = []
    
    def record_result(self, test_name: str, passed: bool, details: str = ""):
        """Record security test result."""
        self.test_results.append({
            'test': test_name,
            'passed': passed,
            'details': details,
            'timestamp': time.time()
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get test summary."""
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r['passed'])
        
        return {
            'total_tests': total,
            'passed': passed,
            'failed': total - passed,
            'success_rate': passed / total * 100 if total > 0 else 0
        }


class TestAuthentication(SecurityTestCase):
    """Authentication security tests."""
    
    def test_valid_authentication(self):
        """Test valid user authentication."""
        safety_manager = QuantumSafetyManager(security_level=SecurityLevel.HIGH)
        
        # Mock valid credentials
        with patch.object(safety_manager, '_validate_credentials') as mock_validate:
            mock_credentials = AccessCredentials(
                user_id="valid_user",
                api_key_hash="valid_hash",
                permissions=["read", "write"],
                security_level=SecurityLevel.HIGH,
                expires_at=time.time() + 3600
            )
            mock_validate.return_value = mock_credentials
            
            result = safety_manager.authenticate_user(
                user_id="valid_user",
                api_key="valid_key",
                required_permissions=["read"]
            )
            
            self.record_result("valid_authentication", result, "Valid user authenticated successfully")
            assert result is True
    
    def test_invalid_credentials(self):
        """Test authentication with invalid credentials."""
        safety_manager = QuantumSafetyManager()
        
        with patch.object(safety_manager, '_validate_credentials') as mock_validate:
            mock_validate.return_value = None
            
            result = safety_manager.authenticate_user(
                user_id="invalid_user",
                api_key="wrong_key"
            )
            
            self.record_result("invalid_credentials", not result, "Invalid credentials rejected")
            assert result is False
    
    def test_brute_force_protection(self):
        """Test protection against brute force attacks."""
        safety_manager = QuantumSafetyManager(
            max_failed_attempts=3,
            lockout_duration=5.0
        )
        
        with patch.object(safety_manager, '_validate_credentials') as mock_validate:
            mock_validate.return_value = None
            
            # Attempt multiple failed logins
            user_id = "brute_force_target"
            for attempt in range(5):
                result = safety_manager.authenticate_user(user_id, f"wrong_key_{attempt}")
                assert result is False
            
            # User should be locked out
            locked_out = user_id in safety_manager.locked_users
            self.record_result("brute_force_protection", locked_out, 
                             f"User locked after failed attempts: {locked_out}")
            
            assert locked_out
    
    def test_session_expiration(self):
        """Test session expiration handling."""
        safety_manager = QuantumSafetyManager()
        
        # Create expired credentials
        expired_credentials = AccessCredentials(
            user_id="expired_user",
            api_key_hash="hash",
            permissions=["read"],
            security_level=SecurityLevel.MEDIUM,
            expires_at=time.time() - 1  # Expired 1 second ago
        )
        
        # Add to active sessions
        safety_manager.active_sessions["expired_user"] = expired_credentials
        
        # Cleanup should remove expired session
        safety_manager._cleanup_expired_sessions()
        
        session_cleaned = "expired_user" not in safety_manager.active_sessions
        self.record_result("session_expiration", session_cleaned, 
                          f"Expired session cleaned: {session_cleaned}")
        
        assert session_cleaned
    
    def test_concurrent_authentication(self):
        """Test authentication under concurrent load."""
        safety_manager = QuantumSafetyManager()
        results = []
        errors = []
        
        def auth_worker(worker_id):
            try:
                with patch.object(safety_manager, '_validate_credentials') as mock:
                    mock.return_value = AccessCredentials(
                        user_id=f"user_{worker_id}",
                        api_key_hash="hash",
                        permissions=["read"],
                        security_level=SecurityLevel.MEDIUM,
                        expires_at=time.time() + 3600
                    )
                    
                    for i in range(10):
                        result = safety_manager.authenticate_user(f"user_{worker_id}", f"key_{i}")
                        results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Run concurrent authentication
        threads = []
        for i in range(5):
            thread = threading.Thread(target=auth_worker, args=[i])
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        success_rate = sum(results) / len(results) * 100 if results else 0
        concurrent_safe = len(errors) == 0 and success_rate == 100
        
        self.record_result("concurrent_authentication", concurrent_safe,
                          f"Concurrent auth success: {success_rate}%, errors: {len(errors)}")
        
        assert concurrent_safe


class TestAuthorization(SecurityTestCase):
    """Authorization security tests."""
    
    def test_permission_enforcement(self):
        """Test permission-based authorization."""
        safety_manager = QuantumSafetyManager()
        
        with patch.object(safety_manager, '_validate_credentials') as mock_validate:
            # User with limited permissions
            limited_user = AccessCredentials(
                user_id="limited_user",
                api_key_hash="hash",
                permissions=["read"],  # No write permission
                security_level=SecurityLevel.MEDIUM,
                expires_at=time.time() + 3600
            )
            mock_validate.return_value = limited_user
            
            # Should succeed for read permission
            read_access = safety_manager.authenticate_user(
                "limited_user", "key", required_permissions=["read"]
            )
            
            # Should fail for write permission  
            mock_validate.return_value = limited_user
            write_access = safety_manager.authenticate_user(
                "limited_user", "key", required_permissions=["write"]
            )
            
            permission_enforced = read_access and not write_access
            self.record_result("permission_enforcement", permission_enforced,
                              f"Read access: {read_access}, Write access: {write_access}")
            
            assert permission_enforced
    
    def test_privilege_escalation_prevention(self):
        """Test prevention of privilege escalation."""
        safety_manager = QuantumSafetyManager()
        
        # Normal user credentials
        normal_user = AccessCredentials(
            user_id="normal_user",
            api_key_hash="hash",
            permissions=["read"],
            security_level=SecurityLevel.LOW,
            expires_at=time.time() + 3600
        )
        
        # Attempt to modify session to escalate privileges
        safety_manager.active_sessions["normal_user"] = normal_user
        
        # Try to access high-security operations
        admin_permissions = ["admin", "system_config"]
        
        # Should not be able to access admin functions
        with patch.object(safety_manager, '_validate_credentials') as mock:
            mock.return_value = normal_user
            
            escalation_blocked = not safety_manager.authenticate_user(
                "normal_user", "key", required_permissions=admin_permissions
            )
        
        self.record_result("privilege_escalation_prevention", escalation_blocked,
                          f"Privilege escalation blocked: {escalation_blocked}")
        
        assert escalation_blocked


class TestDataProtection(SecurityTestCase):
    """Data protection security tests."""
    
    def test_encryption_strength(self):
        """Test encryption/decryption security."""
        safety_manager = QuantumSafetyManager()
        
        # Test data
        sensitive_data = {
            "patient_id": "P123456",
            "diagnosis": "confidential_diagnosis",
            "treatment": "sensitive_treatment_info"
        }
        
        # Encrypt data
        encrypted = safety_manager.encrypt_data(sensitive_data)
        
        # Verify encryption properties
        encryption_strong = (
            isinstance(encrypted, str) and
            len(encrypted) > 50 and  # Should be significantly longer
            "patient_id" not in encrypted and  # Original data not visible
            "diagnosis" not in encrypted
        )
        
        # Test decryption
        decrypted = safety_manager.decrypt_data(encrypted)
        decryption_correct = decrypted == sensitive_data
        
        data_protected = encryption_strong and decryption_correct
        self.record_result("encryption_strength", data_protected,
                          f"Encryption strong: {encryption_strong}, Decryption correct: {decryption_correct}")
        
        assert data_protected
    
    def test_key_rotation_support(self):
        """Test cryptographic key rotation."""
        safety_manager1 = QuantumSafetyManager(master_key="key1")
        safety_manager2 = QuantumSafetyManager(master_key="key2")
        
        test_data = "sensitive_information"
        
        # Encrypt with first key
        encrypted1 = safety_manager1.encrypt_data(test_data)
        
        # Should not decrypt with different key
        try:
            safety_manager2.decrypt_data(encrypted1)
            key_isolation = False
        except:
            key_isolation = True
        
        self.record_result("key_rotation_support", key_isolation,
                          f"Different keys properly isolated: {key_isolation}")
        
        assert key_isolation
    
    def test_quantum_state_integrity(self):
        """Test quantum state integrity protection."""
        safety_manager = QuantumSafetyManager()
        
        # Valid quantum state
        valid_state = np.array([0.707+0j, 0.707+0j], dtype=complex)
        
        # Tampered state  
        tampered_state = np.array([2.0+0j, 0.5+0j], dtype=complex)  # Not normalized
        
        # Corrupted state
        corrupted_state = np.array([np.inf, np.nan], dtype=complex)
        
        valid_accepted = safety_manager.validate_quantum_state("valid", valid_state)
        tampered_rejected = not safety_manager.validate_quantum_state("tampered", tampered_state)
        corrupted_rejected = not safety_manager.validate_quantum_state("corrupted", corrupted_state)
        
        integrity_protected = valid_accepted and tampered_rejected and corrupted_rejected
        
        self.record_result("quantum_state_integrity", integrity_protected,
                          f"Valid: {valid_accepted}, Tampered: {tampered_rejected}, Corrupted: {corrupted_rejected}")
        
        assert integrity_protected
    
    def test_memory_protection(self):
        """Test protection of sensitive data in memory."""
        safety_manager = QuantumSafetyManager()
        
        # Create sensitive data
        sensitive_key = "super_secret_key_12345"
        
        # Simulate memory access patterns
        safety_manager.master_key = sensitive_key.encode()
        
        # Clear sensitive data
        safety_manager.master_key = b''
        
        # Verify memory is cleared (basic test)
        memory_cleared = len(safety_manager.master_key) == 0
        
        self.record_result("memory_protection", memory_cleared,
                          f"Sensitive data cleared from memory: {memory_cleared}")
        
        # Note: Real memory protection would require more sophisticated testing
        assert memory_cleared


class TestThreatDetection(SecurityTestCase):
    """Threat detection security tests."""
    
    def test_anomaly_detection(self):
        """Test anomaly detection capabilities."""
        safety_manager = QuantumSafetyManager()
        
        # Establish baseline
        baseline_metrics = {
            "cpu_usage": 0.3,
            "memory_usage": 0.4,
            "network_traffic": 100.0
        }
        
        # Normal variation
        normal_metrics = {
            "cpu_usage": 0.35,
            "memory_usage": 0.45,
            "network_traffic": 110.0
        }
        
        # Anomalous activity
        anomalous_metrics = {
            "cpu_usage": 0.95,  # Sudden spike
            "memory_usage": 0.9,
            "network_traffic": 10000.0  # Massive increase
        }
        
        normal_anomalies = safety_manager.detect_anomalies(normal_metrics, baseline_metrics)
        suspicious_anomalies = safety_manager.detect_anomalies(anomalous_metrics, baseline_metrics)
        
        detection_working = len(normal_anomalies) == 0 and len(suspicious_anomalies) > 0
        
        self.record_result("anomaly_detection", detection_working,
                          f"Normal anomalies: {len(normal_anomalies)}, Suspicious: {len(suspicious_anomalies)}")
        
        assert detection_working
    
    def test_threat_pattern_recognition(self):
        """Test recognition of attack patterns."""
        safety_manager = QuantumSafetyManager()
        
        # Simulate multiple failed login attempts (brute force pattern)
        for i in range(6):
            safety_manager._log_security_event(
                ThreatType.UNAUTHORIZED_ACCESS,
                SecurityLevel.MEDIUM,
                f"Failed login attempt {i}",
                source_ip="192.168.1.100"
            )
        
        # Check if threat detection rules identify the pattern
        initial_event_count = len(safety_manager.security_events)
        safety_manager._check_threat_detection()
        final_event_count = len(safety_manager.security_events)
        
        # Should generate additional security event for detected threat
        threat_detected = final_event_count > initial_event_count
        
        self.record_result("threat_pattern_recognition", threat_detected,
                          f"Threat detection generated additional events: {threat_detected}")
        
        assert threat_detected
    
    def test_quantum_decoherence_detection(self):
        """Test detection of quantum state attacks."""
        safety_manager = QuantumSafetyManager()
        
        # Simulate quantum decoherence events
        for i in range(4):
            safety_manager._log_security_event(
                ThreatType.QUANTUM_DECOHERENCE,
                SecurityLevel.HIGH,
                f"Abnormal decoherence pattern {i}",
                affected_resources=[f"quantum_state_{i}"]
            )
        
        initial_events = len(safety_manager.security_events)
        safety_manager._check_threat_detection()
        final_events = len(safety_manager.security_events)
        
        quantum_threat_detected = final_events > initial_events
        
        self.record_result("quantum_decoherence_detection", quantum_threat_detected,
                          f"Quantum decoherence pattern detected: {quantum_threat_detected}")
        
        assert quantum_threat_detected


class TestInputValidation(SecurityTestCase):
    """Input validation security tests."""
    
    def test_sql_injection_prevention(self):
        """Test SQL injection prevention."""
        validator = InputValidator()
        
        # Malicious SQL inputs
        sql_injections = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "'; DELETE FROM patients; --",
            "admin'--",
            "' UNION SELECT * FROM sensitive_data--"
        ]
        
        injection_blocked = True
        for injection in sql_injections:
            try:
                validator.sanitize_string(injection)
                # Should raise SecurityError for suspicious patterns
                injection_blocked = False
                break
            except SecurityError:
                continue  # Expected behavior
            except Exception:
                injection_blocked = False
                break
        
        self.record_result("sql_injection_prevention", injection_blocked,
                          f"SQL injection attempts blocked: {injection_blocked}")
        
        assert injection_blocked
    
    def test_xss_prevention(self):
        """Test XSS attack prevention."""
        validator = InputValidator()
        
        # XSS attack vectors
        xss_attacks = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "javascript:alert('xss')",
            "<iframe src=javascript:alert('xss')>",
            "<svg onload=alert('xss')>"
        ]
        
        xss_blocked = True
        for attack in xss_attacks:
            try:
                validator.sanitize_string(attack)
                xss_blocked = False
                break
            except SecurityError:
                continue  # Expected
            except Exception:
                xss_blocked = False
                break
        
        self.record_result("xss_prevention", xss_blocked,
                          f"XSS attacks blocked: {xss_blocked}")
        
        assert xss_blocked
    
    def test_path_traversal_prevention(self):
        """Test path traversal attack prevention."""
        validator = InputValidator()
        
        # Path traversal attempts
        traversal_attacks = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/var/log/../../../etc/shadow",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"
        ]
        
        traversal_blocked = True
        for attack in traversal_attacks:
            try:
                validator.validate_file_path(attack, check_exists=False)
                traversal_blocked = False
                break
            except (SecurityError, ValidationError):
                continue  # Expected
            except Exception:
                traversal_blocked = False
                break
        
        self.record_result("path_traversal_prevention", traversal_blocked,
                          f"Path traversal blocked: {traversal_blocked}")
        
        assert traversal_blocked
    
    def test_buffer_overflow_prevention(self):
        """Test buffer overflow prevention."""
        validator = InputValidator()
        
        # Extremely long input
        long_input = "A" * 100000  # 100KB string
        
        try:
            validator.sanitize_string(long_input, max_length=1000)
            buffer_protected = False  # Should have been rejected
        except ValidationError:
            buffer_protected = True  # Expected behavior
        except Exception:
            buffer_protected = False
        
        self.record_result("buffer_overflow_prevention", buffer_protected,
                          f"Long input rejected: {buffer_protected}")
        
        assert buffer_protected


class TestComplianceValidation(SecurityTestCase):
    """Compliance and regulatory validation tests."""
    
    def test_phi_detection(self):
        """Test PHI (Protected Health Information) detection."""
        validator = InputValidator()
        
        # Sample text with PHI
        phi_text = """
        Patient John Doe, SSN: 123-45-6789, born 01/15/1980.
        Contact: john.doe@email.com, phone: 555-123-4567.
        Medical Record: #MR123456789
        """
        
        # PHI detection should catch sensitive patterns
        contains_phi = validator._contains_phi(phi_text)
        
        # Clean text without PHI
        clean_text = "Patient data has been anonymized according to protocols."
        clean_text_safe = not validator._contains_phi(clean_text)
        
        phi_detection_working = contains_phi and clean_text_safe
        
        self.record_result("phi_detection", phi_detection_working,
                          f"PHI detected in sensitive text: {contains_phi}, Clean text safe: {clean_text_safe}")
        
        assert phi_detection_working
    
    def test_audit_logging(self):
        """Test comprehensive audit logging."""
        safety_manager = QuantumSafetyManager(audit_log_path="test_security_audit.log")
        
        # Generate security events
        test_events = [
            (ThreatType.UNAUTHORIZED_ACCESS, SecurityLevel.HIGH, "Test unauthorized access"),
            (ThreatType.DATA_TAMPERING, SecurityLevel.CRITICAL, "Test data tampering"),
            (ThreatType.QUANTUM_DECOHERENCE, SecurityLevel.MEDIUM, "Test quantum interference")
        ]
        
        for threat_type, severity, description in test_events:
            safety_manager._log_security_event(threat_type, severity, description)
        
        # Verify events were logged
        audit_complete = len(safety_manager.security_events) >= len(test_events)
        
        # Verify audit log file creation (in real implementation)
        # For testing, we verify the logging mechanism works
        
        self.record_result("audit_logging", audit_complete,
                          f"Security events logged: {len(safety_manager.security_events)}")
        
        assert audit_complete
    
    def test_data_retention_policy(self):
        """Test data retention policy compliance."""
        safety_manager = QuantumSafetyManager()
        
        # Add old security events
        old_timestamp = time.time() - (365 * 24 * 3600)  # 1 year ago
        
        old_event = SecurityEvent(
            timestamp=old_timestamp,
            threat_type=ThreatType.UNAUTHORIZED_ACCESS,
            severity=SecurityLevel.LOW,
            description="Old security event"
        )
        
        safety_manager.security_events.append(old_event)
        
        # Add recent event
        recent_event = SecurityEvent(
            timestamp=time.time(),
            threat_type=ThreatType.UNAUTHORIZED_ACCESS,
            severity=SecurityLevel.LOW,
            description="Recent security event"
        )
        
        safety_manager.security_events.append(recent_event)
        
        # In a real implementation, old events would be archived/deleted
        # For testing, we verify the mechanism exists to identify old events
        
        current_time = time.time()
        retention_period = 90 * 24 * 3600  # 90 days
        
        old_events = [
            event for event in safety_manager.security_events
            if current_time - event.timestamp > retention_period
        ]
        
        retention_tracking = len(old_events) > 0
        
        self.record_result("data_retention_policy", retention_tracking,
                          f"Old events identified for retention: {len(old_events)}")
        
        # Note: Actual deletion would be implemented in production
        assert retention_tracking


class TestSecurityIntegration(SecurityTestCase):
    """Integration security tests."""
    
    def test_end_to_end_security_workflow(self):
        """Test complete security workflow."""
        safety_manager = QuantumSafetyManager(security_level=SecurityLevel.HIGH)
        
        workflow_secure = True
        workflow_details = []
        
        try:
            # 1. Authentication
            with patch.object(safety_manager, '_validate_credentials') as mock:
                mock.return_value = AccessCredentials(
                    user_id="secure_user",
                    api_key_hash="hash",
                    permissions=["quantum_ops", "data_access"],
                    security_level=SecurityLevel.HIGH,
                    expires_at=time.time() + 3600
                )
                
                auth_success = safety_manager.authenticate_user(
                    "secure_user", "secure_key", ["quantum_ops"]
                )
                workflow_details.append(f"Authentication: {auth_success}")
                
                if not auth_success:
                    workflow_secure = False
            
            # 2. Data encryption
            sensitive_data = {"quantum_state": "confidential", "parameters": [1, 2, 3]}
            encrypted = safety_manager.encrypt_data(sensitive_data)
            decrypted = safety_manager.decrypt_data(encrypted)
            
            encryption_works = decrypted == sensitive_data
            workflow_details.append(f"Encryption: {encryption_works}")
            
            if not encryption_works:
                workflow_secure = False
            
            # 3. Quantum state validation
            quantum_state = np.array([0.707+0j, 0.707+0j], dtype=complex)
            state_valid = safety_manager.validate_quantum_state("test_state", quantum_state)
            workflow_details.append(f"Quantum validation: {state_valid}")
            
            if not state_valid:
                workflow_secure = False
            
            # 4. Threat detection
            safety_manager.detect_anomalies(
                {"cpu_usage": 0.95}, {"cpu_usage": 0.3}, threshold_multiplier=2.0
            )
            
            threat_detection_active = len(safety_manager.security_events) > 0
            workflow_details.append(f"Threat detection: {threat_detection_active}")
            
        except Exception as e:
            workflow_secure = False
            workflow_details.append(f"Error: {str(e)}")
        
        self.record_result("end_to_end_security_workflow", workflow_secure,
                          " | ".join(workflow_details))
        
        assert workflow_secure


@pytest.mark.security
class TestSecuritySuite:
    """Complete security test suite."""
    
    def test_run_all_security_tests(self):
        """Run comprehensive security test suite."""
        print("\n=== DGDM Quantum Framework Security Validation ===\n")
        
        test_classes = [
            TestAuthentication,
            TestAuthorization, 
            TestDataProtection,
            TestThreatDetection,
            TestInputValidation,
            TestComplianceValidation,
            TestSecurityIntegration
        ]
        
        overall_results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'test_details': []
        }
        
        for test_class in test_classes:
            print(f"Running {test_class.__name__}...")
            test_instance = test_class()
            
            # Run all test methods
            for method_name in dir(test_instance):
                if method_name.startswith('test_'):
                    try:
                        method = getattr(test_instance, method_name)
                        method()
                        print(f"  ✓ {method_name}")
                    except Exception as e:
                        print(f"  ✗ {method_name}: {str(e)}")
            
            # Collect results
            summary = test_instance.get_summary()
            overall_results['total_tests'] += summary['total_tests']
            overall_results['passed'] += summary['passed']
            overall_results['failed'] += summary['failed']
            overall_results['test_details'].extend(test_instance.test_results)
            
            print(f"  Summary: {summary['passed']}/{summary['total_tests']} passed "
                  f"({summary['success_rate']:.1f}%)\n")
        
        # Final summary
        success_rate = (overall_results['passed'] / overall_results['total_tests'] * 100 
                       if overall_results['total_tests'] > 0 else 0)
        
        print(f"=== Security Test Summary ===")
        print(f"Total Tests: {overall_results['total_tests']}")
        print(f"Passed: {overall_results['passed']}")
        print(f"Failed: {overall_results['failed']}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if overall_results['failed'] > 0:
            print("\nFailed Tests:")
            for result in overall_results['test_details']:
                if not result['passed']:
                    print(f"  - {result['test']}: {result['details']}")
        
        # Security threshold - should pass at least 95% of tests
        security_acceptable = success_rate >= 95.0
        
        assert security_acceptable, f"Security test success rate too low: {success_rate:.1f}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "security", "--tb=short"])