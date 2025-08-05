"""Security utilities and safety measures."""

import os
import hashlib
import secrets
import logging
import json
import time
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import hmac
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass


class RateLimiter:
    """Rate limiting for API calls and operations."""
    
    def __init__(self, max_requests: int = 100, time_window: int = 3600):
        self.max_requests = max_requests
        self.time_window = time_window  # seconds
        self.requests = {}  # {client_id: [timestamps]}
        self.logger = logging.getLogger(__name__)
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed under rate limit."""
        current_time = time.time()
        
        # Clean old requests
        if client_id in self.requests:
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if current_time - req_time < self.time_window
            ]
        else:
            self.requests[client_id] = []
        
        # Check limit
        if len(self.requests[client_id]) >= self.max_requests:
            self.logger.warning(f"Rate limit exceeded for client {client_id}")
            return False
        
        # Record request
        self.requests[client_id].append(current_time)
        return True
    
    def get_remaining_requests(self, client_id: str) -> int:
        """Get remaining requests for client."""
        current_time = time.time()
        
        if client_id in self.requests:
            recent_requests = [
                req_time for req_time in self.requests[client_id]
                if current_time - req_time < self.time_window
            ]
            return max(0, self.max_requests - len(recent_requests))
        
        return self.max_requests


class SecurityAuditor:
    """Security auditing and compliance checking."""
    
    def __init__(self, audit_log_path: Optional[Path] = None):
        self.audit_log_path = audit_log_path or Path("logs/security_audit.log")
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def log_security_event(self, event_type: str, details: Dict[str, Any], severity: str = "INFO"):
        """Log security-related events."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'severity': severity,
            'details': details,
            'process_id': os.getpid(),
            'user': os.getenv('USER', 'unknown')
        }
        
        # Log to file
        with open(self.audit_log_path, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        # Log to main logger based on severity
        if severity == "CRITICAL":
            self.logger.critical(f"Security event: {event_type} - {details}")
        elif severity == "ERROR":
            self.logger.error(f"Security event: {event_type} - {details}")
        elif severity == "WARNING":
            self.logger.warning(f"Security event: {event_type} - {details}")
        else:
            self.logger.info(f"Security event: {event_type} - {details}")
    
    def audit_file_access(self, file_path: Path, operation: str, success: bool):
        """Audit file access operations."""
        self.log_security_event(
            'file_access',
            {
                'file_path': str(file_path),
                'operation': operation,
                'success': success,
                'file_size': file_path.stat().st_size if file_path.exists() else 0
            },
            severity="WARNING" if not success else "INFO"
        )
    
    def audit_authentication(self, user_id: str, success: bool, method: str = "unknown"):
        """Audit authentication attempts."""
        self.log_security_event(
            'authentication',
            {
                'user_id': user_id,
                'success': success,
                'method': method
            },
            severity="ERROR" if not success else "INFO"
        )
    
    def audit_configuration_change(self, config_key: str, old_value: Any, new_value: Any):
        """Audit configuration changes."""
        self.log_security_event(
            'configuration_change',
            {
                'config_key': config_key,
                'old_value': str(old_value)[:100],  # Truncate for security
                'new_value': str(new_value)[:100],
                'changed_by': os.getenv('USER', 'unknown')
            },
            severity="WARNING"
        )


class DataEncryption:
    """Data encryption and decryption utilities."""
    
    def __init__(self, password: Optional[str] = None):
        if password is None:
            password = os.getenv('DGDM_ENCRYPTION_KEY', secrets.token_urlsafe(32))
        
        self.password = password.encode()
        self.logger = logging.getLogger(__name__)
    
    def _derive_key(self, salt: bytes) -> bytes:
        """Derive encryption key from password and salt."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(self.password))
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt string data."""
        try:
            # Generate salt
            salt = os.urandom(16)
            
            # Derive key
            key = self._derive_key(salt)
            
            # Encrypt
            fernet = Fernet(key)
            encrypted = fernet.encrypt(data.encode())
            
            # Combine salt and encrypted data
            result = base64.urlsafe_b64encode(salt + encrypted).decode()
            
            self.logger.debug("Data encrypted successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            raise SecurityError(f"Encryption failed: {e}")
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt string data."""
        try:
            # Decode and separate salt and encrypted data
            combined = base64.urlsafe_b64decode(encrypted_data.encode())
            salt = combined[:16]
            encrypted = combined[16:]
            
            # Derive key
            key = self._derive_key(salt)
            
            # Decrypt
            fernet = Fernet(key)
            decrypted = fernet.decrypt(encrypted)
            
            self.logger.debug("Data decrypted successfully")
            return decrypted.decode()
            
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            raise SecurityError(f"Decryption failed: {e}")
    
    def encrypt_file(self, input_path: Path, output_path: Path):
        """Encrypt a file."""
        try:
            with open(input_path, 'rb') as infile:
                data = infile.read()
            
            # Generate salt
            salt = os.urandom(16)
            key = self._derive_key(salt)
            
            # Encrypt
            fernet = Fernet(key)
            encrypted = fernet.encrypt(data)
            
            # Write salt + encrypted data
            with open(output_path, 'wb') as outfile:
                outfile.write(salt + encrypted)
            
            # Set restrictive permissions
            output_path.chmod(0o600)
            
            self.logger.info(f"File encrypted: {input_path} -> {output_path}")
            
        except Exception as e:
            self.logger.error(f"File encryption failed: {e}")
            raise SecurityError(f"File encryption failed: {e}")
    
    def decrypt_file(self, input_path: Path, output_path: Path):
        """Decrypt a file."""
        try:
            with open(input_path, 'rb') as infile:
                combined = infile.read()
            
            # Separate salt and encrypted data
            salt = combined[:16]
            encrypted = combined[16:]
            
            # Derive key and decrypt
            key = self._derive_key(salt)
            fernet = Fernet(key)
            decrypted = fernet.decrypt(encrypted)
            
            # Write decrypted data
            with open(output_path, 'wb') as outfile:
                outfile.write(decrypted)
            
            self.logger.info(f"File decrypted: {input_path} -> {output_path}")
            
        except Exception as e:
            self.logger.error(f"File decryption failed: {e}")
            raise SecurityError(f"File decryption failed: {e}")


class InputSanitizer:
    """Sanitize and validate user inputs."""
    
    DANGEROUS_PATTERNS = [
        # Command injection
        r'[;&|`]',
        r'\$\(',
        r'`.*`',
        
        # Path traversal
        r'\.\.',
        r'[/\\]\.\.', 
        
        # Script injection
        r'<script',
        r'javascript:',
        r'vbscript:',
        
        # SQL injection patterns
        r"'.*'",
        r';.*--',
        r'union.*select',
        
        # Python code injection
        r'__.*__',
        r'exec\s*\(',
        r'eval\s*\(',
        r'import\s+',
    ]
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe filesystem operations."""
        import re
        
        # Remove or replace dangerous characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Remove control characters
        sanitized = ''.join(c for c in sanitized if ord(c) >= 32)
        
        # Limit length
        sanitized = sanitized[:255]
        
        # Ensure not empty
        if not sanitized:
            sanitized = 'unnamed_file'
        
        return sanitized
    
    @staticmethod
    def sanitize_path(path_str: str) -> str:
        """Sanitize path string."""
        import re
        
        # Remove dangerous patterns
        for pattern in InputSanitizer.DANGEROUS_PATTERNS[:4]:  # Only path-related patterns
            path_str = re.sub(pattern, '', path_str, flags=re.IGNORECASE)
        
        # Normalize path separators
        path_str = path_str.replace('\\', '/')
        
        # Remove multiple consecutive slashes
        path_str = re.sub('/+', '/', path_str)
        
        return path_str
    
    @staticmethod
    def validate_input(input_str: str, allow_patterns: List[str] = None) -> bool:
        """Validate input string against dangerous patterns."""
        import re
        
        # Check for dangerous patterns
        for pattern in InputSanitizer.DANGEROUS_PATTERNS:
            if re.search(pattern, input_str, re.IGNORECASE):
                return False
        
        # Check against allowed patterns if specified  
        if allow_patterns:
            for pattern in allow_patterns:
                if re.match(pattern, input_str):
                    return True
            return False
        
        return True


class SecureStorage:
    """Secure storage for sensitive configuration and credentials."""
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.encryption = DataEncryption()
        self.logger = logging.getLogger(__name__)
    
    def store_secret(self, key: str, value: str):
        """Store encrypted secret."""
        try:
            # Load existing storage
            storage_data = self._load_storage()
            
            # Encrypt and store
            storage_data[key] = self.encryption.encrypt_data(value)
            
            # Save back
            self._save_storage(storage_data)
            
            self.logger.info(f"Secret stored: {key}")
            
        except Exception as e:
            self.logger.error(f"Failed to store secret {key}: {e}")
            raise SecurityError(f"Failed to store secret: {e}")
    
    def retrieve_secret(self, key: str) -> Optional[str]:
        """Retrieve and decrypt secret."""
        try:
            storage_data = self._load_storage()
            
            if key not in storage_data:
                return None
            
            decrypted = self.encryption.decrypt_data(storage_data[key])
            self.logger.debug(f"Secret retrieved: {key}")
            return decrypted
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve secret {key}: {e}")
            raise SecurityError(f"Failed to retrieve secret: {e}")
    
    def delete_secret(self, key: str) -> bool:
        """Delete secret from storage."""
        try:
            storage_data = self._load_storage()
            
            if key in storage_data:
                del storage_data[key]
                self._save_storage(storage_data)
                self.logger.info(f"Secret deleted: {key}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to delete secret {key}: {e}")
            raise SecurityError(f"Failed to delete secret: {e}")
    
    def _load_storage(self) -> Dict[str, str]:
        """Load storage file."""
        if not self.storage_path.exists():
            return {}
        
        try:
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    
    def _save_storage(self, data: Dict[str, str]):
        """Save storage file."""
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Set restrictive permissions
        self.storage_path.chmod(0o600)


# Global instances
rate_limiter = RateLimiter()
security_auditor = SecurityAuditor()
input_sanitizer = InputSanitizer()


def generate_api_key(length: int = 32) -> str:
    """Generate secure API key."""
    return secrets.token_urlsafe(length)


def generate_session_token(user_id: str, expiry_hours: int = 24) -> str:
    """Generate secure session token."""
    expiry = datetime.now() + timedelta(hours=expiry_hours)
    payload = {
        'user_id': user_id,
        'expires': expiry.isoformat(),
        'nonce': secrets.token_hex(16)
    }
    
    # Create HMAC signature
    secret_key = os.getenv('DGDM_SECRET_KEY', secrets.token_hex(32)).encode()
    payload_str = json.dumps(payload, sort_keys=True)
    signature = hmac.new(secret_key, payload_str.encode(), hashlib.sha256).hexdigest()
    
    # Combine payload and signature
    token_data = {
        'payload': base64.urlsafe_b64encode(payload_str.encode()).decode(),
        'signature': signature
    }
    
    return base64.urlsafe_b64encode(json.dumps(token_data).encode()).decode()


def verify_session_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify and decode session token."""
    try:
        # Decode token
        token_data = json.loads(base64.urlsafe_b64decode(token.encode()).decode())
        
        # Extract payload and signature
        payload_str = base64.urlsafe_b64decode(token_data['payload'].encode()).decode()
        provided_signature = token_data['signature']
        
        # Verify signature
        secret_key = os.getenv('DGDM_SECRET_KEY', secrets.token_hex(32)).encode()
        expected_signature = hmac.new(secret_key, payload_str.encode(), hashlib.sha256).hexdigest()
        
        if not hmac.compare_digest(provided_signature, expected_signature):
            return None
        
        # Decode payload
        payload = json.loads(payload_str)
        
        # Check expiry
        expiry = datetime.fromisoformat(payload['expires'])
        if datetime.now() > expiry:
            return None
        
        return payload
        
    except Exception:
        return None


def hash_password(password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
    """Hash password with salt."""
    if salt is None:
        salt = os.urandom(32)
    
    pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    
    return base64.b64encode(pwdhash).decode('ascii'), base64.b64encode(salt).decode('ascii')


def verify_password(password: str, hashed_password: str, salt: str) -> bool:
    """Verify password against hash."""
    try:
        salt_bytes = base64.b64decode(salt.encode('ascii'))
        expected_hash, _ = hash_password(password, salt_bytes)
        return hmac.compare_digest(expected_hash, hashed_password)
    except Exception:
        return False