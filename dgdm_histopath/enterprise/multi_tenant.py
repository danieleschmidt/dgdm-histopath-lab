#!/usr/bin/env python3
"""
Multi-Tenant Enterprise Architecture for DGDM Histopath Lab

Enterprise-grade multi-tenancy with isolation, resource management,
and advanced analytics for healthcare organizations.

Author: TERRAGON Autonomous Development System v4.0
Generated: 2025-08-08
"""

import asyncio
import logging
import json
import hashlib
import time
import uuid
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
import base64
from cryptography.fernet import Fernet

try:
    # Enterprise database
    import sqlalchemy as sa
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, relationship, Session
    from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
    SQLALCHEMY_AVAILABLE = True
    Base = declarative_base()
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    logging.warning("SQLAlchemy not available. Install with: pip install sqlalchemy")
    Base = None

try:
    # Redis for caching and session management
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available. Install with: pip install redis")

try:
    # JWT for authentication
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    logging.warning("PyJWT not available. Install with: pip install pyjwt")

try:
    # Advanced monitoring
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("Prometheus client not available. Install with: pip install prometheus_client")

from ..models.dgdm_model import DGDMModel
from ..utils.exceptions import EnterpriseError
from ..utils.validation import validate_tenant_data
from ..utils.monitoring import EnterpriseMetricsCollector
from ..utils.security import encrypt_tenant_data, decrypt_tenant_data


class TenantTier(Enum):
    """Tenant subscription tiers."""
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    RESEARCH = "research"
    CUSTOM = "custom"


class ResourceType(Enum):
    """Types of resources that can be allocated."""
    CPU_CORES = "cpu_cores"
    MEMORY_GB = "memory_gb"
    GPU_HOURS = "gpu_hours"
    STORAGE_GB = "storage_gb"
    API_REQUESTS = "api_requests"
    CONCURRENT_ANALYSES = "concurrent_analyses"
    BATCH_JOBS = "batch_jobs"
    MODEL_DEPLOYMENTS = "model_deployments"


class IsolationLevel(Enum):
    """Data and compute isolation levels."""
    SHARED = "shared"  # Shared resources, logical isolation
    DEDICATED = "dedicated"  # Dedicated resources within shared infrastructure
    PRIVATE = "private"  # Private cloud or on-premises
    HYBRID = "hybrid"  # Mix of shared and dedicated


# Database Models
if SQLALCHEMY_AVAILABLE:
    class Tenant(Base):
        """Tenant organization model."""
        __tablename__ = 'tenants'
        
        id = Column(String, primary_key=True)
        name = Column(String, nullable=False)
        domain = Column(String, unique=True, nullable=False)
        tier = Column(String, nullable=False)
        isolation_level = Column(String, nullable=False)
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        active = Column(Boolean, default=True)
        
        # Metadata
        contact_email = Column(String)
        billing_contact = Column(String)
        technical_contact = Column(String)
        metadata = Column(Text)  # JSON metadata
        
        # Relationships
        users = relationship("TenantUser", back_populates="tenant")
        resources = relationship("TenantResource", back_populates="tenant")
        analytics = relationship("TenantAnalytics", back_populates="tenant")
    
    class TenantUser(Base):
        """Tenant user model."""
        __tablename__ = 'tenant_users'
        
        id = Column(String, primary_key=True)
        tenant_id = Column(String, ForeignKey('tenants.id'), nullable=False)
        username = Column(String, nullable=False)
        email = Column(String, nullable=False)
        role = Column(String, nullable=False)
        permissions = Column(Text)  # JSON permissions
        active = Column(Boolean, default=True)
        last_login = Column(DateTime)
        created_at = Column(DateTime, default=datetime.utcnow)
        
        # Relationships
        tenant = relationship("Tenant", back_populates="users")
    
    class TenantResource(Base):
        """Tenant resource allocation model."""
        __tablename__ = 'tenant_resources'
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        tenant_id = Column(String, ForeignKey('tenants.id'), nullable=False)
        resource_type = Column(String, nullable=False)
        allocated = Column(Float, nullable=False)
        used = Column(Float, default=0.0)
        reserved = Column(Float, default=0.0)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        
        # Relationships
        tenant = relationship("Tenant", back_populates="resources")
    
    class TenantAnalytics(Base):
        """Tenant analytics and usage tracking."""
        __tablename__ = 'tenant_analytics'
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        tenant_id = Column(String, ForeignKey('tenants.id'), nullable=False)
        timestamp = Column(DateTime, default=datetime.utcnow)
        metric_name = Column(String, nullable=False)
        metric_value = Column(Float, nullable=False)
        metadata = Column(Text)  # JSON metadata
        
        # Relationships
        tenant = relationship("Tenant", back_populates="analytics")


@dataclass
class TenantConfiguration:
    """Configuration for tenant setup."""
    # Basic tenant info
    tenant_id: str
    name: str
    domain: str
    tier: TenantTier = TenantTier.PROFESSIONAL
    isolation_level: IsolationLevel = IsolationLevel.DEDICATED
    
    # Resource limits by tier
    resource_limits: Dict[ResourceType, float] = field(default_factory=dict)
    
    # Security settings
    encryption_enabled: bool = True
    audit_logging: bool = True
    data_retention_days: int = 2555  # 7 years
    
    # Integration settings
    allowed_integrations: List[str] = field(default_factory=list)
    custom_domains: List[str] = field(default_factory=list)
    
    # Feature flags
    enabled_features: Set[str] = field(default_factory=set)
    beta_features: Set[str] = field(default_factory=set)
    
    # Compliance
    compliance_requirements: List[str] = field(default_factory=lambda: [
        'HIPAA', 'GDPR', 'SOC2'
    ])
    
    # Billing
    billing_model: str = "usage_based"  # flat_rate, usage_based, custom
    billing_contact: Optional[str] = None


class TenantResourceManager:
    """Manages resource allocation and usage for tenants."""
    
    def __init__(self, database_session: Session):
        self.db_session = database_session
        self.resource_locks = {}
        self.usage_cache = {}
        self.metrics = EnterpriseMetricsCollector()
        
        # Default resource limits by tier
        self.tier_limits = {
            TenantTier.BASIC: {
                ResourceType.CPU_CORES: 2.0,
                ResourceType.MEMORY_GB: 8.0,
                ResourceType.GPU_HOURS: 10.0,
                ResourceType.STORAGE_GB: 100.0,
                ResourceType.API_REQUESTS: 1000.0,
                ResourceType.CONCURRENT_ANALYSES: 2.0,
                ResourceType.BATCH_JOBS: 5.0,
                ResourceType.MODEL_DEPLOYMENTS: 1.0
            },
            TenantTier.PROFESSIONAL: {
                ResourceType.CPU_CORES: 8.0,
                ResourceType.MEMORY_GB: 32.0,
                ResourceType.GPU_HOURS: 100.0,
                ResourceType.STORAGE_GB: 1000.0,
                ResourceType.API_REQUESTS: 10000.0,
                ResourceType.CONCURRENT_ANALYSES: 10.0,
                ResourceType.BATCH_JOBS: 25.0,
                ResourceType.MODEL_DEPLOYMENTS: 5.0
            },
            TenantTier.ENTERPRISE: {
                ResourceType.CPU_CORES: 32.0,
                ResourceType.MEMORY_GB: 128.0,
                ResourceType.GPU_HOURS: 1000.0,
                ResourceType.STORAGE_GB: 10000.0,
                ResourceType.API_REQUESTS: 100000.0,
                ResourceType.CONCURRENT_ANALYSES: 50.0,
                ResourceType.BATCH_JOBS: 100.0,
                ResourceType.MODEL_DEPLOYMENTS: 25.0
            },
            TenantTier.RESEARCH: {
                ResourceType.CPU_CORES: 64.0,
                ResourceType.MEMORY_GB: 256.0,
                ResourceType.GPU_HOURS: 5000.0,
                ResourceType.STORAGE_GB: 50000.0,
                ResourceType.API_REQUESTS: 500000.0,
                ResourceType.CONCURRENT_ANALYSES: 100.0,
                ResourceType.BATCH_JOBS: 500.0,
                ResourceType.MODEL_DEPLOYMENTS: 50.0
            }
        }
    
    async def allocate_resources(self, tenant_id: str, tier: TenantTier, 
                               custom_limits: Optional[Dict[ResourceType, float]] = None) -> bool:
        """Allocate resources for a tenant based on their tier."""
        try:
            # Get resource limits
            limits = custom_limits or self.tier_limits.get(tier, self.tier_limits[TenantTier.BASIC])
            
            # Create resource entries
            for resource_type, limit in limits.items():
                resource = TenantResource(
                    tenant_id=tenant_id,
                    resource_type=resource_type.value,
                    allocated=limit,
                    used=0.0,
                    reserved=0.0
                )
                
                # Check if resource already exists
                existing = self.db_session.query(TenantResource).filter_by(
                    tenant_id=tenant_id,
                    resource_type=resource_type.value
                ).first()
                
                if existing:
                    existing.allocated = limit
                else:
                    self.db_session.add(resource)
            
            self.db_session.commit()
            logging.info(f"Resources allocated for tenant {tenant_id}")
            return True
            
        except Exception as e:
            self.db_session.rollback()
            logging.error(f"Resource allocation failed for tenant {tenant_id}: {e}")
            return False
    
    async def check_resource_availability(self, tenant_id: str, 
                                        resource_type: ResourceType,
                                        requested_amount: float) -> bool:
        """Check if tenant has sufficient resources available."""
        try:
            resource = self.db_session.query(TenantResource).filter_by(
                tenant_id=tenant_id,
                resource_type=resource_type.value
            ).first()
            
            if not resource:
                return False
            
            available = resource.allocated - resource.used - resource.reserved
            return available >= requested_amount
            
        except Exception as e:
            logging.error(f"Resource availability check failed: {e}")
            return False
    
    async def reserve_resources(self, tenant_id: str,
                              resource_requests: Dict[ResourceType, float]) -> Optional[str]:
        """Reserve resources for a tenant operation."""
        try:
            reservation_id = str(uuid.uuid4())
            
            # Check availability for all requested resources
            for resource_type, amount in resource_requests.items():
                if not await self.check_resource_availability(tenant_id, resource_type, amount):
                    logging.warning(f"Insufficient {resource_type.value} for tenant {tenant_id}")
                    return None
            
            # Reserve resources
            for resource_type, amount in resource_requests.items():
                resource = self.db_session.query(TenantResource).filter_by(
                    tenant_id=tenant_id,
                    resource_type=resource_type.value
                ).first()
                
                if resource:
                    resource.reserved += amount
            
            self.db_session.commit()
            
            # Cache reservation details
            self.usage_cache[reservation_id] = {
                'tenant_id': tenant_id,
                'resources': resource_requests,
                'timestamp': datetime.utcnow(),
                'status': 'reserved'
            }
            
            logging.info(f"Resources reserved for tenant {tenant_id}: {reservation_id}")
            return reservation_id
            
        except Exception as e:
            self.db_session.rollback()
            logging.error(f"Resource reservation failed: {e}")
            return None
    
    async def consume_resources(self, reservation_id: str) -> bool:
        """Convert reserved resources to consumed."""
        try:
            if reservation_id not in self.usage_cache:
                return False
            
            reservation = self.usage_cache[reservation_id]
            tenant_id = reservation['tenant_id']
            resource_requests = reservation['resources']
            
            # Move from reserved to used
            for resource_type, amount in resource_requests.items():
                resource = self.db_session.query(TenantResource).filter_by(
                    tenant_id=tenant_id,
                    resource_type=resource_type.value
                ).first()
                
                if resource:
                    resource.reserved -= amount
                    resource.used += amount
            
            self.db_session.commit()
            
            # Update cache
            self.usage_cache[reservation_id]['status'] = 'consumed'
            
            # Record usage analytics
            await self._record_usage_analytics(tenant_id, resource_requests)
            
            return True
            
        except Exception as e:
            self.db_session.rollback()
            logging.error(f"Resource consumption failed: {e}")
            return False
    
    async def release_resources(self, reservation_id: str) -> bool:
        """Release reserved or used resources."""
        try:
            if reservation_id not in self.usage_cache:
                return False
            
            reservation = self.usage_cache[reservation_id]
            tenant_id = reservation['tenant_id']
            resource_requests = reservation['resources']
            status = reservation['status']
            
            # Release resources based on current status
            for resource_type, amount in resource_requests.items():
                resource = self.db_session.query(TenantResource).filter_by(
                    tenant_id=tenant_id,
                    resource_type=resource_type.value
                ).first()
                
                if resource:
                    if status == 'reserved':
                        resource.reserved -= amount
                    elif status == 'consumed':
                        resource.used -= amount
            
            self.db_session.commit()
            
            # Remove from cache
            del self.usage_cache[reservation_id]
            
            return True
            
        except Exception as e:
            self.db_session.rollback()
            logging.error(f"Resource release failed: {e}")
            return False
    
    async def get_tenant_usage(self, tenant_id: str) -> Dict[str, Any]:
        """Get current resource usage for a tenant."""
        try:
            resources = self.db_session.query(TenantResource).filter_by(
                tenant_id=tenant_id
            ).all()
            
            usage_data = {
                'tenant_id': tenant_id,
                'resources': {},
                'utilization': {}
            }
            
            for resource in resources:
                resource_info = {
                    'allocated': resource.allocated,
                    'used': resource.used,
                    'reserved': resource.reserved,
                    'available': resource.allocated - resource.used - resource.reserved
                }
                
                usage_data['resources'][resource.resource_type] = resource_info
                
                # Calculate utilization percentage
                if resource.allocated > 0:
                    utilization = ((resource.used + resource.reserved) / resource.allocated) * 100
                    usage_data['utilization'][resource.resource_type] = utilization
            
            return usage_data
            
        except Exception as e:
            logging.error(f"Usage retrieval failed: {e}")
            return {'error': str(e)}
    
    async def _record_usage_analytics(self, tenant_id: str, 
                                    resource_usage: Dict[ResourceType, float]):
        """Record usage analytics for billing and monitoring."""
        try:
            for resource_type, amount in resource_usage.items():
                analytics = TenantAnalytics(
                    tenant_id=tenant_id,
                    metric_name=f"resource_usage_{resource_type.value}",
                    metric_value=amount,
                    metadata=json.dumps({
                        'resource_type': resource_type.value,
                        'usage_timestamp': datetime.utcnow().isoformat()
                    })
                )
                
                self.db_session.add(analytics)
            
            self.db_session.commit()
            
        except Exception as e:
            logging.error(f"Usage analytics recording failed: {e}")


class TenantIsolationManager:
    """Manages data and compute isolation between tenants."""
    
    def __init__(self, isolation_level: IsolationLevel):
        self.isolation_level = isolation_level
        self.tenant_encryption_keys = {}
        self.tenant_namespaces = {}
        
    async def setup_tenant_isolation(self, tenant_id: str, config: TenantConfiguration) -> Dict[str, Any]:
        """Setup isolation for a new tenant."""
        try:
            isolation_setup = {
                'tenant_id': tenant_id,
                'isolation_level': config.isolation_level.value,
                'setup_timestamp': datetime.utcnow().isoformat()
            }
            
            # Data encryption setup
            if config.encryption_enabled:
                encryption_key = Fernet.generate_key()
                self.tenant_encryption_keys[tenant_id] = encryption_key
                isolation_setup['encryption_enabled'] = True
                isolation_setup['encryption_key_id'] = hashlib.sha256(encryption_key).hexdigest()[:16]
            
            # Namespace isolation
            namespace = f"tenant_{tenant_id}"
            self.tenant_namespaces[tenant_id] = namespace
            isolation_setup['namespace'] = namespace
            
            # Storage isolation
            storage_path = f"/data/tenants/{tenant_id}"
            Path(storage_path).mkdir(parents=True, exist_ok=True)
            isolation_setup['storage_path'] = storage_path
            
            # Network isolation (placeholder)
            if config.isolation_level in [IsolationLevel.DEDICATED, IsolationLevel.PRIVATE]:
                isolation_setup['network_isolation'] = await self._setup_network_isolation(tenant_id)
            
            # Database isolation
            isolation_setup['database_isolation'] = await self._setup_database_isolation(
                tenant_id, config.isolation_level
            )
            
            logging.info(f"Tenant isolation setup completed: {tenant_id}")
            return isolation_setup
            
        except Exception as e:
            logging.error(f"Tenant isolation setup failed: {e}")
            return {'error': str(e)}
    
    async def encrypt_tenant_data(self, tenant_id: str, data: bytes) -> bytes:
        """Encrypt data for a specific tenant."""
        try:
            if tenant_id not in self.tenant_encryption_keys:
                raise EnterpriseError(f"No encryption key found for tenant {tenant_id}")
            
            encryption_key = self.tenant_encryption_keys[tenant_id]
            fernet = Fernet(encryption_key)
            
            return fernet.encrypt(data)
            
        except Exception as e:
            logging.error(f"Data encryption failed: {e}")
            raise
    
    async def decrypt_tenant_data(self, tenant_id: str, encrypted_data: bytes) -> bytes:
        """Decrypt data for a specific tenant."""
        try:
            if tenant_id not in self.tenant_encryption_keys:
                raise EnterpriseError(f"No encryption key found for tenant {tenant_id}")
            
            encryption_key = self.tenant_encryption_keys[tenant_id]
            fernet = Fernet(encryption_key)
            
            return fernet.decrypt(encrypted_data)
            
        except Exception as e:
            logging.error(f"Data decryption failed: {e}")
            raise
    
    async def _setup_network_isolation(self, tenant_id: str) -> Dict[str, Any]:
        """Setup network isolation for tenant."""
        # Placeholder for network isolation setup
        # In a real implementation, this would configure VPCs, subnets, etc.
        return {
            'vpc_id': f"vpc-{tenant_id[:8]}",
            'subnet_id': f"subnet-{tenant_id[:8]}",
            'security_group': f"sg-{tenant_id[:8]}",
            'status': 'configured'
        }
    
    async def _setup_database_isolation(self, tenant_id: str, 
                                      isolation_level: IsolationLevel) -> Dict[str, Any]:
        """Setup database isolation based on isolation level."""
        if isolation_level == IsolationLevel.SHARED:
            # Shared database with logical isolation (row-level security)
            return {
                'type': 'shared_database',
                'schema': f"tenant_{tenant_id}",
                'rls_enabled': True
            }
        
        elif isolation_level == IsolationLevel.DEDICATED:
            # Dedicated database schema
            return {
                'type': 'dedicated_schema',
                'database_name': f"dgdm_tenant_{tenant_id}",
                'connection_pool': f"pool_{tenant_id}"
            }
        
        elif isolation_level == IsolationLevel.PRIVATE:
            # Private database instance
            return {
                'type': 'private_instance',
                'instance_id': f"db-{tenant_id}",
                'endpoint': f"db-{tenant_id}.internal"
            }
        
        else:
            return {'type': 'hybrid', 'configuration': 'custom'}


class TenantAuthenticationManager:
    """Manages authentication and authorization for multi-tenant environment."""
    
    def __init__(self, jwt_secret: str, redis_client=None):
        self.jwt_secret = jwt_secret
        self.redis_client = redis_client
        self.session_timeout = 3600  # 1 hour
        
    async def authenticate_user(self, tenant_id: str, username: str, 
                              password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user within tenant context."""
        try:
            # In a real implementation, this would verify against tenant's user database
            # For now, we'll simulate authentication
            
            # Verify user exists and password is correct (placeholder)
            user_valid = await self._verify_user_credentials(tenant_id, username, password)
            
            if not user_valid:
                return None
            
            # Get user permissions
            user_permissions = await self._get_user_permissions(tenant_id, username)
            
            # Create JWT token
            token_payload = {
                'tenant_id': tenant_id,
                'username': username,
                'permissions': user_permissions,
                'exp': datetime.utcnow() + timedelta(seconds=self.session_timeout),
                'iat': datetime.utcnow()
            }
            
            if JWT_AVAILABLE:
                token = jwt.encode(token_payload, self.jwt_secret, algorithm='HS256')
            else:
                token = base64.b64encode(json.dumps(token_payload).encode()).decode()
            
            # Store session in Redis if available
            session_id = str(uuid.uuid4())
            if self.redis_client:
                session_data = {
                    'tenant_id': tenant_id,
                    'username': username,
                    'permissions': user_permissions,
                    'token': token
                }
                
                self.redis_client.setex(
                    f"session:{session_id}", 
                    self.session_timeout, 
                    json.dumps(session_data)
                )
            
            return {
                'token': token,
                'session_id': session_id,
                'tenant_id': tenant_id,
                'username': username,
                'permissions': user_permissions,
                'expires_at': (datetime.utcnow() + timedelta(seconds=self.session_timeout)).isoformat()
            }
            
        except Exception as e:
            logging.error(f"Authentication failed: {e}")
            return None
    
    async def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate JWT token and return user context."""
        try:
            if JWT_AVAILABLE:
                payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            else:
                # Fallback base64 decoding
                payload = json.loads(base64.b64decode(token).decode())
                
                # Check expiration manually
                if 'exp' in payload:
                    exp_time = datetime.fromisoformat(payload['exp'].replace('Z', '+00:00'))
                    if datetime.utcnow() > exp_time.replace(tzinfo=None):
                        return None
            
            return {
                'tenant_id': payload['tenant_id'],
                'username': payload['username'],
                'permissions': payload.get('permissions', []),
                'valid': True
            }
            
        except Exception as e:
            logging.error(f"Token validation failed: {e}")
            return None
    
    async def check_permission(self, user_context: Dict[str, Any], 
                             required_permission: str) -> bool:
        """Check if user has required permission."""
        try:
            user_permissions = user_context.get('permissions', [])
            
            # Check direct permission
            if required_permission in user_permissions:
                return True
            
            # Check wildcard permissions
            for permission in user_permissions:
                if permission.endswith('*') and required_permission.startswith(permission[:-1]):
                    return True
            
            # Check admin role
            if 'admin' in user_permissions:
                return True
            
            return False
            
        except Exception as e:
            logging.error(f"Permission check failed: {e}")
            return False
    
    async def _verify_user_credentials(self, tenant_id: str, username: str, password: str) -> bool:
        """Verify user credentials (placeholder implementation)."""
        # In a real implementation, this would:
        # 1. Query the tenant's user database
        # 2. Verify password hash
        # 3. Check account status (active, locked, etc.)
        
        # For now, simulate successful authentication
        return True
    
    async def _get_user_permissions(self, tenant_id: str, username: str) -> List[str]:
        """Get user permissions for tenant (placeholder implementation)."""
        # In a real implementation, this would query user roles and permissions
        # For now, return default permissions
        return [
            'dgdm:analyze',
            'dgdm:view_results',
            'tenant:view_usage',
            'tenant:manage_data'
        ]


class MultiTenantManager:
    """High-level manager for multi-tenant enterprise architecture."""
    
    def __init__(self, database_url: str, jwt_secret: str, redis_url: Optional[str] = None):
        # Database setup
        if SQLALCHEMY_AVAILABLE:
            self.engine = sa.create_engine(database_url)
            Base.metadata.create_all(self.engine)
            Session = sessionmaker(bind=self.engine)
            self.db_session = Session()
        else:
            raise EnterpriseError("SQLAlchemy required for multi-tenant architecture")
        
        # Redis setup
        self.redis_client = None
        if REDIS_AVAILABLE and redis_url:
            self.redis_client = redis.from_url(redis_url)
        
        # Initialize managers
        self.resource_manager = TenantResourceManager(self.db_session)
        self.isolation_manager = TenantIsolationManager(IsolationLevel.DEDICATED)
        self.auth_manager = TenantAuthenticationManager(jwt_secret, self.redis_client)
        
        self.metrics = EnterpriseMetricsCollector()
        
        # Start metrics server if available
        if PROMETHEUS_AVAILABLE:
            start_http_server(8000)
    
    async def create_tenant(self, config: TenantConfiguration) -> Dict[str, Any]:
        """Create a new tenant with full setup."""
        try:
            # Create tenant record
            tenant = Tenant(
                id=config.tenant_id,
                name=config.name,
                domain=config.domain,
                tier=config.tier.value,
                isolation_level=config.isolation_level.value,
                contact_email=config.billing_contact,
                metadata=json.dumps({
                    'enabled_features': list(config.enabled_features),
                    'beta_features': list(config.beta_features),
                    'compliance_requirements': config.compliance_requirements
                })
            )
            
            self.db_session.add(tenant)
            self.db_session.commit()
            
            # Allocate resources
            resource_allocation = await self.resource_manager.allocate_resources(
                config.tenant_id, config.tier, config.resource_limits
            )
            
            # Setup isolation
            isolation_setup = await self.isolation_manager.setup_tenant_isolation(
                config.tenant_id, config
            )
            
            # Create admin user
            admin_user = TenantUser(
                id=str(uuid.uuid4()),
                tenant_id=config.tenant_id,
                username="admin",
                email=config.billing_contact or "admin@" + config.domain,
                role="admin",
                permissions=json.dumps([
                    'admin', 'dgdm:*', 'tenant:*'
                ])
            )
            
            self.db_session.add(admin_user)
            self.db_session.commit()
            
            # Record creation metrics
            await self.metrics.record_tenant_creation(config.tenant_id, config.tier.value)
            
            creation_result = {
                'tenant_id': config.tenant_id,
                'status': 'created',
                'timestamp': datetime.utcnow().isoformat(),
                'resource_allocation': resource_allocation,
                'isolation_setup': isolation_setup,
                'admin_user_id': admin_user.id
            }
            
            logging.info(f"Tenant created successfully: {config.tenant_id}")
            return creation_result
            
        except Exception as e:
            self.db_session.rollback()
            logging.error(f"Tenant creation failed: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    async def get_tenant_info(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive tenant information."""
        try:
            tenant = self.db_session.query(Tenant).filter_by(id=tenant_id).first()
            
            if not tenant:
                return None
            
            # Get resource usage
            resource_usage = await self.resource_manager.get_tenant_usage(tenant_id)
            
            # Get user count
            user_count = self.db_session.query(TenantUser).filter_by(
                tenant_id=tenant_id, active=True
            ).count()
            
            # Get recent analytics
            recent_analytics = self.db_session.query(TenantAnalytics).filter(
                TenantAnalytics.tenant_id == tenant_id,
                TenantAnalytics.timestamp >= datetime.utcnow() - timedelta(days=7)
            ).all()
            
            tenant_info = {
                'tenant_id': tenant.id,
                'name': tenant.name,
                'domain': tenant.domain,
                'tier': tenant.tier,
                'isolation_level': tenant.isolation_level,
                'active': tenant.active,
                'created_at': tenant.created_at.isoformat(),
                'updated_at': tenant.updated_at.isoformat(),
                'contact_email': tenant.contact_email,
                'metadata': json.loads(tenant.metadata or '{}'),
                'resource_usage': resource_usage,
                'user_count': user_count,
                'analytics_summary': {
                    'total_analyses': len([a for a in recent_analytics if 'analysis' in a.metric_name]),
                    'total_api_calls': len([a for a in recent_analytics if 'api' in a.metric_name]),
                    'last_activity': max([a.timestamp for a in recent_analytics]).isoformat() if recent_analytics else None
                }
            }
            
            return tenant_info
            
        except Exception as e:
            logging.error(f"Tenant info retrieval failed: {e}")
            return None
    
    async def list_tenants(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """List all tenants with summary information."""
        try:
            query = self.db_session.query(Tenant)
            if active_only:
                query = query.filter_by(active=True)
            
            tenants = query.all()
            
            tenant_list = []
            for tenant in tenants:
                summary = {
                    'tenant_id': tenant.id,
                    'name': tenant.name,
                    'domain': tenant.domain,
                    'tier': tenant.tier,
                    'isolation_level': tenant.isolation_level,
                    'active': tenant.active,
                    'created_at': tenant.created_at.isoformat()
                }
                tenant_list.append(summary)
            
            return tenant_list
            
        except Exception as e:
            logging.error(f"Tenant listing failed: {e}")
            return []
    
    async def update_tenant_tier(self, tenant_id: str, new_tier: TenantTier) -> bool:
        """Update tenant tier and reallocate resources."""
        try:
            tenant = self.db_session.query(Tenant).filter_by(id=tenant_id).first()
            
            if not tenant:
                return False
            
            old_tier = TenantTier(tenant.tier)
            tenant.tier = new_tier.value
            tenant.updated_at = datetime.utcnow()
            
            # Reallocate resources
            resource_reallocation = await self.resource_manager.allocate_resources(
                tenant_id, new_tier
            )
            
            if resource_reallocation:
                self.db_session.commit()
                
                # Record tier change analytics
                analytics = TenantAnalytics(
                    tenant_id=tenant_id,
                    metric_name="tier_change",
                    metric_value=1.0,
                    metadata=json.dumps({
                        'old_tier': old_tier.value,
                        'new_tier': new_tier.value,
                        'change_timestamp': datetime.utcnow().isoformat()
                    })
                )
                
                self.db_session.add(analytics)
                self.db_session.commit()
                
                logging.info(f"Tenant tier updated: {tenant_id} -> {new_tier.value}")
                return True
            else:
                self.db_session.rollback()
                return False
                
        except Exception as e:
            self.db_session.rollback()
            logging.error(f"Tenant tier update failed: {e}")
            return False
    
    async def deactivate_tenant(self, tenant_id: str) -> bool:
        """Deactivate tenant and release resources."""
        try:
            tenant = self.db_session.query(Tenant).filter_by(id=tenant_id).first()
            
            if not tenant:
                return False
            
            tenant.active = False
            tenant.updated_at = datetime.utcnow()
            
            # Deactivate all tenant users
            self.db_session.query(TenantUser).filter_by(tenant_id=tenant_id).update({
                'active': False
            })
            
            # Clear resource allocations
            self.db_session.query(TenantResource).filter_by(tenant_id=tenant_id).delete()
            
            self.db_session.commit()
            
            # Record deactivation
            analytics = TenantAnalytics(
                tenant_id=tenant_id,
                metric_name="tenant_deactivated",
                metric_value=1.0,
                metadata=json.dumps({
                    'deactivation_timestamp': datetime.utcnow().isoformat()
                })
            )
            
            self.db_session.add(analytics)
            self.db_session.commit()
            
            logging.info(f"Tenant deactivated: {tenant_id}")
            return True
            
        except Exception as e:
            self.db_session.rollback()
            logging.error(f"Tenant deactivation failed: {e}")
            return False
    
    def close(self):
        """Close database connections and cleanup."""
        if self.db_session:
            self.db_session.close()
        
        if self.redis_client:
            self.redis_client.close()
        
        logging.info("Multi-tenant manager closed")


# Export main components
__all__ = [
    'TenantTier',
    'ResourceType',
    'IsolationLevel',
    'TenantConfiguration',
    'TenantResourceManager',
    'TenantIsolationManager',
    'TenantAuthenticationManager',
    'MultiTenantManager'
]
