"""
Adaptive Security with ML-Driven Threat Detection and Response

Features:
- Real-time threat detection and analysis
- Adaptive security policies based on risk assessment
- Automated incident response and containment
- Compliance monitoring and validation
- Zero-trust architecture enforcement
"""

import asyncio
import json
import hashlib
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict, deque
import re
import ipaddress

from ..utils.logging import get_logger
from ..utils.monitoring import MetricsCollector

logger = get_logger(__name__)


class ThreatLevel(Enum):
    """Security threat levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(Enum):
    """Types of security events"""
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_VIOLATION = "authz_violation"
    SUSPICIOUS_ACCESS = "suspicious_access"
    DATA_EXFILTRATION = "data_exfiltration"
    MALICIOUS_PAYLOAD = "malicious_payload"
    BRUTE_FORCE_ATTACK = "brute_force"
    SQL_INJECTION = "sql_injection"
    XSS_ATTEMPT = "xss_attempt"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"


class ResponseAction(Enum):
    """Security response actions"""
    BLOCK_IP = "block_ip"
    DISABLE_USER = "disable_user"
    QUARANTINE_RESOURCE = "quarantine_resource"
    INCREASE_MONITORING = "increase_monitoring"
    ALERT_ADMIN = "alert_admin"
    FORCE_LOGOUT = "force_logout"
    REQUIRE_MFA = "require_mfa"
    RESTRICT_ACCESS = "restrict_access"
    LOG_INCIDENT = "log_incident"


@dataclass
class SecurityEvent:
    """Security event record"""
    event_id: str
    timestamp: datetime
    event_type: SecurityEventType
    threat_level: ThreatLevel
    source_ip: str
    user_id: Optional[str]
    resource: str
    description: str
    risk_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    response_actions: List[ResponseAction] = field(default_factory=list)


@dataclass
class SecurityPolicy:
    """Security policy definition"""
    policy_id: str
    name: str
    description: str
    conditions: List[Dict[str, Any]]
    actions: List[ResponseAction]
    enabled: bool = True
    priority: int = 100
    auto_apply: bool = True


@dataclass
class RiskProfile:
    """User/IP risk profile"""
    identifier: str
    identifier_type: str  # "user" or "ip"
    risk_score: float
    failed_attempts: int
    suspicious_activities: int
    last_activity: datetime
    behavioral_baseline: Dict[str, float] = field(default_factory=dict)
    trust_level: str = "unknown"  # unknown, trusted, suspicious, blocked


class ThreatDetector:
    """ML-based threat detection system"""
    
    def __init__(self):
        # Pattern databases
        self.malicious_patterns = self._load_malicious_patterns()
        self.anomaly_thresholds = self._initialize_anomaly_thresholds()
        
        # Behavioral baselines
        self.user_baselines: Dict[str, Dict[str, Any]] = {}
        self.ip_baselines: Dict[str, Dict[str, Any]] = {}
        
    def _load_malicious_patterns(self) -> Dict[str, List[str]]:
        """Load known malicious patterns"""
        return {
            "sql_injection": [
                r"union\s+select",
                r"or\s+1\s*=\s*1",
                r"'\s*or\s*'1'\s*=\s*'1",
                r"drop\s+table",
                r"exec\s*\(",
                r"script\s*:",
                r"javascript\s*:",
                r"<\s*script",
                r"eval\s*\(",
                r"document\s*\.\s*cookie"
            ],
            "xss": [
                r"<\s*script.*?>.*?<\s*/\s*script\s*>",
                r"javascript\s*:",
                r"on\w+\s*=",
                r"<\s*iframe",
                r"<\s*object",
                r"<\s*embed",
                r"expression\s*\(",
                r"@import",
                r"<\s*link.*?javascript"
            ],
            "path_traversal": [
                r"\.\.\/",
                r"\.\.\\",
                r"\/\.\.\/" ,
                r"\\\.\.\\",
                r"%2e%2e%2f",
                r"%2e%2e%5c",
                r"..%2f",
                r"..%5c"
            ],
            "command_injection": [
                r";\s*cat\s+",
                r";\s*ls\s+",
                r";\s*id\s*;",
                r";\s*whoami",
                r";\s*pwd",
                r"\|\s*nc\s+",
                r"&&\s*cat\s+",
                r"`\s*cat\s+",
                r"\$\(\s*cat\s+"
            ]
        }
    
    def _initialize_anomaly_thresholds(self) -> Dict[str, float]:
        """Initialize anomaly detection thresholds"""
        return {
            "failed_login_rate": 5.0,       # failures per minute
            "request_rate": 100.0,          # requests per minute
            "data_transfer_rate": 10.0,     # MB per minute
            "unusual_hours_threshold": 0.1,  # probability of access at unusual hours
            "geographic_anomaly": 0.2,      # probability of access from unusual location
            "user_agent_anomaly": 0.3       # probability of unusual user agent
        }
    
    async def analyze_event(self, event_data: Dict[str, Any]) -> SecurityEvent:
        """Analyze security event and determine threat level"""
        try:
            # Extract event information
            event_type = self._classify_event_type(event_data)
            risk_score = await self._calculate_risk_score(event_data, event_type)
            threat_level = self._determine_threat_level(risk_score)
            
            # Create security event
            security_event = SecurityEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                event_type=event_type,
                threat_level=threat_level,
                source_ip=event_data.get("source_ip", "unknown"),
                user_id=event_data.get("user_id"),
                resource=event_data.get("resource", "unknown"),
                description=self._generate_description(event_data, event_type),
                risk_score=risk_score,
                metadata=event_data
            )
            
            return security_event
            
        except Exception as e:
            logger.error(f"Event analysis failed: {e}")
            # Return safe default
            return SecurityEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                event_type=SecurityEventType.ANOMALOUS_BEHAVIOR,
                threat_level=ThreatLevel.LOW,
                source_ip=event_data.get("source_ip", "unknown"),
                user_id=event_data.get("user_id"),
                resource=event_data.get("resource", "unknown"),
                description="Event analysis failed",
                risk_score=0.3
            )
    
    def _classify_event_type(self, event_data: Dict[str, Any]) -> SecurityEventType:
        """Classify the type of security event"""
        # Check for specific indicators
        request_data = event_data.get("request_body", "") + event_data.get("query_params", "")
        
        # SQL injection detection
        if self._check_patterns(request_data, "sql_injection"):
            return SecurityEventType.SQL_INJECTION
        
        # XSS detection
        if self._check_patterns(request_data, "xss"):
            return SecurityEventType.XSS_ATTEMPT
        
        # Authentication failures
        if event_data.get("status_code") == 401 or "login" in event_data.get("endpoint", ""):
            return SecurityEventType.AUTHENTICATION_FAILURE
        
        # Authorization violations
        if event_data.get("status_code") == 403:
            return SecurityEventType.AUTHORIZATION_VIOLATION
        
        # Brute force detection
        if self._detect_brute_force(event_data):
            return SecurityEventType.BRUTE_FORCE_ATTACK
        
        # Data exfiltration detection
        if self._detect_data_exfiltration(event_data):
            return SecurityEventType.DATA_EXFILTRATION
        
        # Default to anomalous behavior
        return SecurityEventType.ANOMALOUS_BEHAVIOR
    
    def _check_patterns(self, text: str, pattern_type: str) -> bool:
        """Check text against malicious patterns"""
        if pattern_type not in self.malicious_patterns:
            return False
        
        for pattern in self.malicious_patterns[pattern_type]:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _detect_brute_force(self, event_data: Dict[str, Any]) -> bool:
        """Detect brute force attack patterns"""
        # Check for rapid authentication failures
        source_ip = event_data.get("source_ip")
        if not source_ip:
            return False
        
        # This would integrate with actual request tracking in production
        # For now, use simple heuristics
        return (
            event_data.get("status_code") == 401 and
            event_data.get("user_agent", "").count("curl") > 0
        )
    
    def _detect_data_exfiltration(self, event_data: Dict[str, Any]) -> bool:
        """Detect potential data exfiltration"""
        # Check for large data transfers
        response_size = event_data.get("response_size", 0)
        if response_size > 10 * 1024 * 1024:  # 10MB
            return True
        
        # Check for sensitive data patterns in response
        response_data = event_data.get("response_body", "")
        sensitive_patterns = [
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"\b\d{16}\b",             # Credit card
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"  # Email
        ]
        
        for pattern in sensitive_patterns:
            if re.search(pattern, response_data):
                return True
        
        return False
    
    async def _calculate_risk_score(self, event_data: Dict[str, Any], 
                                   event_type: SecurityEventType) -> float:
        """Calculate risk score for the event"""
        base_scores = {
            SecurityEventType.SQL_INJECTION: 0.9,
            SecurityEventType.XSS_ATTEMPT: 0.8,
            SecurityEventType.BRUTE_FORCE_ATTACK: 0.7,
            SecurityEventType.DATA_EXFILTRATION: 0.95,
            SecurityEventType.PRIVILEGE_ESCALATION: 0.9,
            SecurityEventType.MALICIOUS_PAYLOAD: 0.85,
            SecurityEventType.AUTHENTICATION_FAILURE: 0.3,
            SecurityEventType.AUTHORIZATION_VIOLATION: 0.5,
            SecurityEventType.SUSPICIOUS_ACCESS: 0.4,
            SecurityEventType.ANOMALOUS_BEHAVIOR: 0.2
        }
        
        risk_score = base_scores.get(event_type, 0.3)
        
        # Risk modifiers
        modifiers = []
        
        # Source IP reputation
        source_ip = event_data.get("source_ip")
        if source_ip:
            ip_risk = await self._assess_ip_risk(source_ip)
            modifiers.append(ip_risk * 0.3)
        
        # User risk profile
        user_id = event_data.get("user_id")
        if user_id:
            user_risk = await self._assess_user_risk(user_id)
            modifiers.append(user_risk * 0.2)
        
        # Time-based risk (unusual hours)
        time_risk = self._assess_time_risk(event_data.get("timestamp", datetime.now()))
        modifiers.append(time_risk * 0.1)
        
        # Geographic risk
        geo_risk = await self._assess_geographic_risk(event_data)
        modifiers.append(geo_risk * 0.2)
        
        # Apply modifiers
        for modifier in modifiers:
            risk_score = min(1.0, risk_score + modifier)
        
        return risk_score
    
    async def _assess_ip_risk(self, ip_address: str) -> float:
        """Assess risk level of IP address"""
        try:
            # Check if IP is in private range (lower risk)
            ip = ipaddress.ip_address(ip_address)
            if ip.is_private:
                return 0.1
            
            # Check against known bad IP lists (would integrate with threat intelligence)
            # For simulation, use simple heuristics
            if ip_address.startswith(("192.168.", "10.", "172.")):
                return 0.1  # Private networks
            
            # Placeholder for threat intelligence integration
            return 0.3
            
        except Exception:
            return 0.5  # Unknown/invalid IP
    
    async def _assess_user_risk(self, user_id: str) -> float:
        """Assess risk level of user"""
        # This would integrate with user behavior analysis
        # For simulation, return baseline risk
        return 0.2
    
    def _assess_time_risk(self, timestamp: datetime) -> float:
        """Assess risk based on time of access"""
        hour = timestamp.hour
        
        # Higher risk during off-hours (11PM - 6AM)
        if hour >= 23 or hour <= 6:
            return 0.3
        
        # Weekend access slightly higher risk
        if timestamp.weekday() >= 5:  # Saturday, Sunday
            return 0.1
        
        return 0.0
    
    async def _assess_geographic_risk(self, event_data: Dict[str, Any]) -> float:
        """Assess geographic risk based on location"""
        # This would integrate with geolocation and user travel patterns
        # For simulation, return baseline risk
        return 0.1
    
    def _determine_threat_level(self, risk_score: float) -> ThreatLevel:
        """Determine threat level from risk score"""
        if risk_score >= 0.8:
            return ThreatLevel.CRITICAL
        elif risk_score >= 0.6:
            return ThreatLevel.HIGH
        elif risk_score >= 0.4:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def _generate_description(self, event_data: Dict[str, Any], 
                             event_type: SecurityEventType) -> str:
        """Generate human-readable event description"""
        descriptions = {
            SecurityEventType.SQL_INJECTION: "SQL injection attempt detected",
            SecurityEventType.XSS_ATTEMPT: "Cross-site scripting attempt detected", 
            SecurityEventType.BRUTE_FORCE_ATTACK: "Brute force attack detected",
            SecurityEventType.DATA_EXFILTRATION: "Potential data exfiltration detected",
            SecurityEventType.AUTHENTICATION_FAILURE: "Authentication failure",
            SecurityEventType.AUTHORIZATION_VIOLATION: "Authorization violation",
            SecurityEventType.SUSPICIOUS_ACCESS: "Suspicious access pattern",
            SecurityEventType.ANOMALOUS_BEHAVIOR: "Anomalous behavior detected"
        }
        
        base_desc = descriptions.get(event_type, "Security event detected")
        
        # Add context
        context_parts = []
        if "source_ip" in event_data:
            context_parts.append(f"from {event_data['source_ip']}")
        if "user_id" in event_data:
            context_parts.append(f"by user {event_data['user_id']}")
        if "resource" in event_data:
            context_parts.append(f"targeting {event_data['resource']}")
        
        if context_parts:
            return f"{base_desc} {' '.join(context_parts)}"
        
        return base_desc


class AdaptiveSecurity:
    """Adaptive security system with ML-driven threat detection"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        
        # Core components
        self.threat_detector = ThreatDetector()
        self.metrics_collector = MetricsCollector()
        
        # Event storage and tracking
        self.security_events: deque = deque(maxlen=10000)
        self.active_incidents: Dict[str, SecurityEvent] = {}
        self.risk_profiles: Dict[str, RiskProfile] = {}
        
        # Security policies
        self.security_policies: Dict[str, SecurityPolicy] = {}
        self.blocked_ips: Set[str] = set()
        self.disabled_users: Set[str] = set()
        
        # Response system
        self.response_handlers: Dict[ResponseAction, callable] = {}
        self.incident_handlers: List[callable] = []
        
        # Background monitoring
        self.monitoring_active = False
        self.monitoring_task = None
        self.cleanup_task = None
        
        # Initialize system
        self._initialize_default_policies()
        self._initialize_response_handlers()
        self._start_background_monitoring()
        
        logger.info("Adaptive security system initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load security configuration"""
        default_config = {
            "monitoring_interval": 30,
            "event_retention_hours": 72,
            "risk_profile_retention_days": 30,
            "auto_response_enabled": True,
            "threat_intelligence_enabled": True,
            "behavioral_analysis_enabled": True,
            "compliance_monitoring": ["GDPR", "HIPAA", "SOC2"],
            "alert_thresholds": {
                "high_risk_events_per_hour": 10,
                "critical_events_per_hour": 3,
                "failed_logins_per_ip": 5
            }
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load security config: {e}")
        
        return default_config
    
    def _initialize_default_policies(self):
        """Initialize default security policies"""
        default_policies = [
            SecurityPolicy(
                policy_id="block_sql_injection",
                name="Block SQL Injection Attempts",
                description="Automatically block IP addresses attempting SQL injection",
                conditions=[
                    {"event_type": "sql_injection"},
                    {"risk_score": "> 0.7"}
                ],
                actions=[ResponseAction.BLOCK_IP, ResponseAction.ALERT_ADMIN],
                priority=90
            ),
            SecurityPolicy(
                policy_id="block_brute_force",
                name="Block Brute Force Attacks", 
                description="Block IP addresses after multiple failed login attempts",
                conditions=[
                    {"event_type": "brute_force"},
                    {"failed_attempts": "> 5"}
                ],
                actions=[ResponseAction.BLOCK_IP, ResponseAction.INCREASE_MONITORING],
                priority=85
            ),
            SecurityPolicy(
                policy_id="disable_compromised_user",
                name="Disable Compromised User Accounts",
                description="Disable user accounts showing signs of compromise",
                conditions=[
                    {"risk_score": "> 0.8"},
                    {"event_type": "privilege_escalation"}
                ],
                actions=[ResponseAction.DISABLE_USER, ResponseAction.ALERT_ADMIN, ResponseAction.FORCE_LOGOUT],
                priority=95
            ),
            SecurityPolicy(
                policy_id="require_mfa_suspicious",
                name="Require MFA for Suspicious Activity",
                description="Require additional authentication for suspicious activities",
                conditions=[
                    {"risk_score": "> 0.5"},
                    {"trust_level": "suspicious"}
                ],
                actions=[ResponseAction.REQUIRE_MFA, ResponseAction.INCREASE_MONITORING],
                priority=70
            )
        ]
        
        for policy in default_policies:
            self.security_policies[policy.policy_id] = policy
    
    def _initialize_response_handlers(self):
        """Initialize response action handlers"""
        self.response_handlers = {
            ResponseAction.BLOCK_IP: self._block_ip,
            ResponseAction.DISABLE_USER: self._disable_user,
            ResponseAction.QUARANTINE_RESOURCE: self._quarantine_resource,
            ResponseAction.INCREASE_MONITORING: self._increase_monitoring,
            ResponseAction.ALERT_ADMIN: self._alert_admin,
            ResponseAction.FORCE_LOGOUT: self._force_logout,
            ResponseAction.REQUIRE_MFA: self._require_mfa,
            ResponseAction.RESTRICT_ACCESS: self._restrict_access,
            ResponseAction.LOG_INCIDENT: self._log_incident
        }
    
    def _start_background_monitoring(self):
        """Start background security monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Started security background monitoring")
    
    async def _monitoring_loop(self):
        """Main security monitoring loop"""
        while self.monitoring_active:
            try:
                await self._analyze_security_trends()
                await self._update_risk_profiles()
                await self._check_compliance()
                
                await asyncio.sleep(self.config["monitoring_interval"])
                
            except Exception as e:
                logger.error(f"Security monitoring loop error: {e}")
                await asyncio.sleep(10)
    
    async def _cleanup_loop(self):
        """Background cleanup of old security data"""
        while self.monitoring_active:
            try:
                await self._cleanup_old_data()
                await asyncio.sleep(3600)  # Run hourly
                
            except Exception as e:
                logger.error(f"Security cleanup loop error: {e}")
                await asyncio.sleep(600)
    
    async def process_security_event(self, event_data: Dict[str, Any]) -> SecurityEvent:
        """Process incoming security event"""
        try:
            # Analyze event with threat detector
            security_event = await self.threat_detector.analyze_event(event_data)
            
            # Store event
            self.security_events.append(security_event)
            
            # Update risk profiles
            await self._update_risk_profile_from_event(security_event)
            
            # Check if event triggers any policies
            triggered_policies = await self._evaluate_policies(security_event)
            
            # Execute response actions if auto-response is enabled
            if self.config["auto_response_enabled"] and triggered_policies:
                await self._execute_policy_responses(security_event, triggered_policies)
            
            # Check if this creates an incident
            if security_event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                await self._create_incident(security_event)
            
            # Notify incident handlers
            for handler in self.incident_handlers:
                try:
                    await handler(security_event)
                except Exception as e:
                    logger.error(f"Incident handler failed: {e}")
            
            logger.info(
                f"Security event processed: {security_event.event_type.value} "
                f"({security_event.threat_level.value}) from {security_event.source_ip}"
            )
            
            return security_event
            
        except Exception as e:
            logger.error(f"Security event processing failed: {e}")
            raise
    
    async def _update_risk_profile_from_event(self, event: SecurityEvent):
        """Update risk profiles based on security event"""
        # Update IP risk profile
        if event.source_ip and event.source_ip != "unknown":
            ip_profile = self.risk_profiles.get(f"ip:{event.source_ip}")
            if not ip_profile:
                ip_profile = RiskProfile(
                    identifier=event.source_ip,
                    identifier_type="ip",
                    risk_score=0.1,
                    failed_attempts=0,
                    suspicious_activities=0,
                    last_activity=event.timestamp,
                    trust_level="unknown"
                )
                self.risk_profiles[f"ip:{event.source_ip}"] = ip_profile
            
            # Update profile based on event
            if event.event_type == SecurityEventType.AUTHENTICATION_FAILURE:
                ip_profile.failed_attempts += 1
            
            if event.threat_level in [ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                ip_profile.suspicious_activities += 1
            
            # Update risk score (exponential moving average)
            alpha = 0.3
            ip_profile.risk_score = alpha * event.risk_score + (1 - alpha) * ip_profile.risk_score
            ip_profile.last_activity = event.timestamp
            
            # Update trust level
            if ip_profile.risk_score > 0.8:
                ip_profile.trust_level = "blocked"
            elif ip_profile.risk_score > 0.6:
                ip_profile.trust_level = "suspicious"
            elif ip_profile.risk_score < 0.3 and ip_profile.failed_attempts == 0:
                ip_profile.trust_level = "trusted"
        
        # Update user risk profile
        if event.user_id:
            user_profile = self.risk_profiles.get(f"user:{event.user_id}")
            if not user_profile:
                user_profile = RiskProfile(
                    identifier=event.user_id,
                    identifier_type="user",
                    risk_score=0.1,
                    failed_attempts=0,
                    suspicious_activities=0,
                    last_activity=event.timestamp,
                    trust_level="unknown"
                )
                self.risk_profiles[f"user:{event.user_id}"] = user_profile
            
            # Update user profile similar to IP profile
            if event.event_type == SecurityEventType.AUTHENTICATION_FAILURE:
                user_profile.failed_attempts += 1
            
            if event.threat_level in [ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                user_profile.suspicious_activities += 1
            
            alpha = 0.3
            user_profile.risk_score = alpha * event.risk_score + (1 - alpha) * user_profile.risk_score
            user_profile.last_activity = event.timestamp
            
            # Update trust level
            if user_profile.risk_score > 0.8:
                user_profile.trust_level = "blocked"
            elif user_profile.risk_score > 0.6:
                user_profile.trust_level = "suspicious"
            elif user_profile.risk_score < 0.3 and user_profile.failed_attempts == 0:
                user_profile.trust_level = "trusted"
    
    async def _evaluate_policies(self, event: SecurityEvent) -> List[SecurityPolicy]:
        """Evaluate security policies against event"""
        triggered_policies = []
        
        for policy in self.security_policies.values():
            if not policy.enabled:
                continue
            
            try:
                if await self._policy_matches_event(policy, event):
                    triggered_policies.append(policy)
            except Exception as e:
                logger.error(f"Policy evaluation failed for {policy.policy_id}: {e}")
        
        # Sort by priority (higher number = higher priority)
        triggered_policies.sort(key=lambda p: p.priority, reverse=True)
        
        return triggered_policies
    
    async def _policy_matches_event(self, policy: SecurityPolicy, event: SecurityEvent) -> bool:
        """Check if policy conditions match the event"""
        for condition in policy.conditions:
            if not self._evaluate_condition(condition, event):
                return False
        return True
    
    def _evaluate_condition(self, condition: Dict[str, Any], event: SecurityEvent) -> bool:
        """Evaluate a single policy condition"""
        try:
            for field, criteria in condition.items():
                event_value = getattr(event, field, None)
                
                if event_value is None and field in event.metadata:
                    event_value = event.metadata[field]
                
                if event_value is None:
                    return False
                
                # Handle different criteria types
                if isinstance(criteria, str):
                    if criteria.startswith("> "):
                        threshold = float(criteria[2:])
                        if not (isinstance(event_value, (int, float)) and event_value > threshold):
                            return False
                    elif criteria.startswith("< "):
                        threshold = float(criteria[2:])
                        if not (isinstance(event_value, (int, float)) and event_value < threshold):
                            return False
                    elif criteria.startswith("= "):
                        target_value = criteria[2:]
                        if str(event_value) != target_value:
                            return False
                    else:
                        # Direct string comparison
                        if str(event_value) != criteria:
                            return False
                elif isinstance(criteria, (int, float)):
                    if event_value != criteria:
                        return False
                elif isinstance(criteria, list):
                    if event_value not in criteria:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Condition evaluation failed: {e}")
            return False
    
    async def _execute_policy_responses(self, event: SecurityEvent, policies: List[SecurityPolicy]):
        """Execute response actions for triggered policies"""
        executed_actions = set()
        
        for policy in policies:
            for action in policy.actions:
                if action in executed_actions:
                    continue  # Don't execute same action twice
                
                try:
                    if action in self.response_handlers:
                        await self.response_handlers[action](event, policy)
                        executed_actions.add(action)
                        event.response_actions.append(action)
                        
                        logger.info(f"Executed response action: {action.value} for event {event.event_id}")
                    else:
                        logger.warning(f"No handler for response action: {action.value}")
                        
                except Exception as e:
                    logger.error(f"Response action {action.value} failed: {e}")
    
    # Response action handlers
    async def _block_ip(self, event: SecurityEvent, policy: SecurityPolicy):
        """Block IP address"""
        if event.source_ip and event.source_ip != "unknown":
            self.blocked_ips.add(event.source_ip)
            logger.warning(f"Blocked IP address: {event.source_ip}")
    
    async def _disable_user(self, event: SecurityEvent, policy: SecurityPolicy):
        """Disable user account"""
        if event.user_id:
            self.disabled_users.add(event.user_id)
            logger.warning(f"Disabled user account: {event.user_id}")
    
    async def _quarantine_resource(self, event: SecurityEvent, policy: SecurityPolicy):
        """Quarantine affected resource"""
        logger.info(f"Quarantined resource: {event.resource}")
    
    async def _increase_monitoring(self, event: SecurityEvent, policy: SecurityPolicy):
        """Increase monitoring for source"""
        logger.info(f"Increased monitoring for {event.source_ip}")
    
    async def _alert_admin(self, event: SecurityEvent, policy: SecurityPolicy):
        """Send alert to administrators"""
        logger.warning(f"SECURITY ALERT: {event.description} (Risk: {event.risk_score:.2f})")
    
    async def _force_logout(self, event: SecurityEvent, policy: SecurityPolicy):
        """Force user logout"""
        if event.user_id:
            logger.info(f"Forced logout for user: {event.user_id}")
    
    async def _require_mfa(self, event: SecurityEvent, policy: SecurityPolicy):
        """Require multi-factor authentication"""
        if event.user_id:
            logger.info(f"MFA required for user: {event.user_id}")
    
    async def _restrict_access(self, event: SecurityEvent, policy: SecurityPolicy):
        """Restrict user access"""
        if event.user_id:
            logger.info(f"Restricted access for user: {event.user_id}")
    
    async def _log_incident(self, event: SecurityEvent, policy: SecurityPolicy):
        """Log security incident"""
        logger.info(f"Security incident logged: {event.event_id}")
    
    async def _create_incident(self, event: SecurityEvent):
        """Create security incident for high-risk events"""
        self.active_incidents[event.event_id] = event
        logger.warning(f"Security incident created: {event.event_id} ({event.threat_level.value})")
    
    async def _analyze_security_trends(self):
        """Analyze security trends and patterns"""
        try:
            # Analyze recent events (last hour)
            cutoff = datetime.now() - timedelta(hours=1)
            recent_events = [
                event for event in self.security_events
                if event.timestamp >= cutoff
            ]
            
            if not recent_events:
                return
            
            # Count events by type and threat level
            event_counts = defaultdict(int)
            threat_counts = defaultdict(int)
            
            for event in recent_events:
                event_counts[event.event_type] += 1
                threat_counts[event.threat_level] += 1
            
            # Check alert thresholds
            high_risk_count = threat_counts[ThreatLevel.HIGH] + threat_counts[ThreatLevel.CRITICAL]
            critical_count = threat_counts[ThreatLevel.CRITICAL]
            
            if high_risk_count > self.config["alert_thresholds"]["high_risk_events_per_hour"]:
                logger.warning(f"High risk event threshold exceeded: {high_risk_count} events in last hour")
            
            if critical_count > self.config["alert_thresholds"]["critical_events_per_hour"]:
                logger.error(f"Critical event threshold exceeded: {critical_count} events in last hour")
            
        except Exception as e:
            logger.error(f"Security trend analysis failed: {e}")
    
    async def _update_risk_profiles(self):
        """Update and decay risk profiles over time"""
        try:
            current_time = datetime.now()
            
            for profile_key, profile in list(self.risk_profiles.items()):
                # Time-based risk decay
                hours_since_activity = (current_time - profile.last_activity).total_seconds() / 3600
                
                if hours_since_activity > 24:
                    # Decay risk score over time
                    decay_factor = min(0.9, hours_since_activity / 168)  # Decay over week
                    profile.risk_score *= (1 - decay_factor)
                    
                    # Reset counters if very old activity
                    if hours_since_activity > 168:  # 1 week
                        profile.failed_attempts = max(0, profile.failed_attempts - 1)
                        profile.suspicious_activities = max(0, profile.suspicious_activities - 1)
                
                # Remove very old profiles
                if hours_since_activity > self.config["risk_profile_retention_days"] * 24:
                    del self.risk_profiles[profile_key]
            
        except Exception as e:
            logger.error(f"Risk profile update failed: {e}")
    
    async def _check_compliance(self):
        """Check compliance with security standards"""
        try:
            compliance_issues = []
            
            # Check for compliance violations
            for standard in self.config["compliance_monitoring"]:
                if standard == "GDPR":
                    # Check for data access without proper authorization
                    issues = await self._check_gdpr_compliance()
                    compliance_issues.extend(issues)
                
                elif standard == "HIPAA":
                    # Check for medical data access compliance
                    issues = await self._check_hipaa_compliance()
                    compliance_issues.extend(issues)
                
                elif standard == "SOC2":
                    # Check for SOC2 security requirements
                    issues = await self._check_soc2_compliance()
                    compliance_issues.extend(issues)
            
            if compliance_issues:
                logger.warning(f"Compliance issues detected: {compliance_issues}")
            
        except Exception as e:
            logger.error(f"Compliance check failed: {e}")
    
    async def _check_gdpr_compliance(self) -> List[str]:
        """Check GDPR compliance"""
        issues = []
        
        # Check for unauthorized data access
        recent_events = list(self.security_events)[-100:]  # Last 100 events
        
        for event in recent_events:
            if (event.event_type == SecurityEventType.AUTHORIZATION_VIOLATION and
                "personal_data" in event.metadata.get("resource_type", "")):
                issues.append(f"Unauthorized personal data access: {event.event_id}")
        
        return issues
    
    async def _check_hipaa_compliance(self) -> List[str]:
        """Check HIPAA compliance"""
        issues = []
        
        # Check for medical data breaches
        recent_events = list(self.security_events)[-100:]
        
        for event in recent_events:
            if (event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL] and
                "medical" in event.metadata.get("data_type", "")):
                issues.append(f"Potential medical data breach: {event.event_id}")
        
        return issues
    
    async def _check_soc2_compliance(self) -> List[str]:
        """Check SOC2 compliance"""
        issues = []
        
        # Check for security control failures
        if len(self.active_incidents) > 5:
            issues.append("Multiple active security incidents may indicate control failures")
        
        return issues
    
    async def _cleanup_old_data(self):
        """Clean up old security data"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.config["event_retention_hours"])
            
            # Clean old events
            original_count = len(self.security_events)
            self.security_events = deque(
                (event for event in self.security_events if event.timestamp >= cutoff_time),
                maxlen=10000
            )
            
            # Clean resolved incidents
            resolved_incidents = [
                event_id for event_id, event in self.active_incidents.items()
                if event.resolved and event.timestamp < cutoff_time
            ]
            
            for event_id in resolved_incidents:
                del self.active_incidents[event_id]
            
            logger.debug(f"Security cleanup: removed {original_count - len(self.security_events)} events")
            
        except Exception as e:
            logger.error(f"Security data cleanup failed: {e}")
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked"""
        return ip_address in self.blocked_ips
    
    def is_user_disabled(self, user_id: str) -> bool:
        """Check if user is disabled"""
        return user_id in self.disabled_users
    
    def get_risk_profile(self, identifier: str, identifier_type: str) -> Optional[RiskProfile]:
        """Get risk profile for user or IP"""
        profile_key = f"{identifier_type}:{identifier}"
        return self.risk_profiles.get(profile_key)
    
    def add_security_policy(self, policy: SecurityPolicy):
        """Add custom security policy"""
        self.security_policies[policy.policy_id] = policy
        logger.info(f"Added security policy: {policy.name}")
    
    def add_incident_handler(self, handler: callable):
        """Add incident handler function"""
        self.incident_handlers.append(handler)
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security system summary"""
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_events = [
            event for event in self.security_events
            if event.timestamp >= recent_cutoff
        ]
        
        # Event statistics
        event_stats = defaultdict(int)
        threat_stats = defaultdict(int)
        
        for event in recent_events:
            event_stats[event.event_type.value] += 1
            threat_stats[event.threat_level.value] += 1
        
        return {
            "total_events_24h": len(recent_events),
            "event_types": dict(event_stats),
            "threat_levels": dict(threat_stats),
            "active_incidents": len(self.active_incidents),
            "blocked_ips": len(self.blocked_ips),
            "disabled_users": len(self.disabled_users),
            "risk_profiles": len(self.risk_profiles),
            "active_policies": sum(1 for p in self.security_policies.values() if p.enabled)
        }
    
    async def shutdown(self):
        """Gracefully shutdown adaptive security system"""
        self.monitoring_active = False
        
        # Cancel background tasks
        for task in [self.monitoring_task, self.cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Adaptive security system shutdown complete")