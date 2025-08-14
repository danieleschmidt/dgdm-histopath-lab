"""Monitoring and health check system for quality gates."""

import os
import sys
import json
import time
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import sqlite3
from contextlib import contextmanager

from ..utils.logging import get_logger


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    check_name: str
    status: str  # "healthy", "warning", "critical"
    message: str
    details: Dict[str, Any]
    timestamp: float
    execution_time: float
    

@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    load_average: float
    timestamp: float


@dataclass
class QualityGateMetrics:
    """Quality gate execution metrics."""
    gate_name: str
    execution_count: int
    success_count: int
    failure_count: int
    average_execution_time: float
    last_execution_time: float
    last_status: str
    timestamp: float


class HealthMonitor:
    """Comprehensive health monitoring system."""
    
    def __init__(self, db_path: str = "./monitoring.db"):
        self.db_path = Path(db_path)
        self.logger = get_logger(__name__)
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_interval = 60  # seconds
        
        # Health check registry
        self.health_checks = {
            'system_resources': self._check_system_resources,
            'disk_space': self._check_disk_space,
            'memory_usage': self._check_memory_usage,
            'quality_gate_performance': self._check_quality_gate_performance,
            'database_connectivity': self._check_database_connectivity,
            'file_permissions': self._check_file_permissions
        }
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize monitoring database."""
        with sqlite3.connect(str(self.db_path)) as conn:
            # Health check results table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS health_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    check_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT,
                    details TEXT,
                    timestamp REAL NOT NULL,
                    execution_time REAL
                )
            """)
            
            # System metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cpu_usage REAL,
                    memory_usage REAL,
                    disk_usage REAL,
                    load_average REAL,
                    timestamp REAL NOT NULL
                )
            """)
            
            # Quality gate metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_gate_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gate_name TEXT NOT NULL,
                    execution_count INTEGER,
                    success_count INTEGER,
                    failure_count INTEGER,
                    average_execution_time REAL,
                    last_execution_time REAL,
                    last_status TEXT,
                    timestamp REAL NOT NULL
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_health_timestamp ON health_checks(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON system_metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_gate_metrics_name ON quality_gate_metrics(gate_name)")
    
    def start_monitoring(self, interval: int = 60):
        """Start continuous monitoring."""
        if self.monitoring_active:
            self.logger.warning("Monitoring is already active")
            return
        
        self.monitoring_interval = interval
        self.monitoring_active = True
        
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        self.logger.info(f"Health monitoring started with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Run all health checks
                self.run_health_checks()
                
                # Collect system metrics
                self._collect_system_metrics()
                
                # Sleep until next interval
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)  # Short sleep before retrying
    
    def run_health_checks(self) -> List[HealthCheckResult]:
        """Run all registered health checks."""
        results = []
        
        for check_name, check_func in self.health_checks.items():
            try:
                start_time = time.time()
                result = check_func()
                execution_time = time.time() - start_time
                
                result.execution_time = execution_time
                results.append(result)
                
                # Store in database
                self._store_health_check_result(result)
                
            except Exception as e:
                self.logger.error(f"Health check {check_name} failed: {e}")
                error_result = HealthCheckResult(
                    check_name=check_name,
                    status="critical",
                    message=f"Health check failed: {e}",
                    details={'error': str(e)},
                    timestamp=time.time(),
                    execution_time=0.0
                )
                results.append(error_result)
                self._store_health_check_result(error_result)
        
        return results
    
    def _check_system_resources(self) -> HealthCheckResult:
        """Check system resource usage."""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            status = "healthy"
            message = "System resources normal"
            
            # Determine status based on usage
            if cpu_percent > 90 or memory.percent > 90:
                status = "critical"
                message = "High resource usage detected"
            elif cpu_percent > 75 or memory.percent > 75:
                status = "warning"
                message = "Elevated resource usage"
            
            return HealthCheckResult(
                check_name="system_resources",
                status=status,
                message=message,
                details={
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_gb': memory.available / (1024**3)
                },
                timestamp=time.time(),
                execution_time=0.0
            )
            
        except ImportError:
            return HealthCheckResult(
                check_name="system_resources",
                status="warning",
                message="psutil not available - cannot monitor system resources",
                details={},
                timestamp=time.time(),
                execution_time=0.0
            )
    
    def _check_disk_space(self) -> HealthCheckResult:
        """Check available disk space."""
        try:
            import shutil
            
            total, used, free = shutil.disk_usage("/")
            free_percent = (free / total) * 100
            
            status = "healthy"
            message = f"Disk space: {free_percent:.1f}% free"
            
            if free_percent < 5:
                status = "critical"
                message = "Very low disk space"
            elif free_percent < 15:
                status = "warning"
                message = "Low disk space"
            
            return HealthCheckResult(
                check_name="disk_space",
                status=status,
                message=message,
                details={
                    'total_gb': total / (1024**3),
                    'used_gb': used / (1024**3),
                    'free_gb': free / (1024**3),
                    'free_percent': free_percent
                },
                timestamp=time.time(),
                execution_time=0.0
            )
            
        except Exception as e:
            return HealthCheckResult(
                check_name="disk_space",
                status="warning",
                message=f"Cannot check disk space: {e}",
                details={'error': str(e)},
                timestamp=time.time(),
                execution_time=0.0
            )
    
    def _check_memory_usage(self) -> HealthCheckResult:
        """Check memory usage patterns."""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            status = "healthy"
            message = "Memory usage normal"
            
            # Check for memory pressure
            if memory.percent > 95:
                status = "critical"
                message = "Critical memory usage"
            elif memory.percent > 85:
                status = "warning"
                message = "High memory usage"
            
            # Check swap usage
            if swap.percent > 50:
                if status == "healthy":
                    status = "warning"
                    message = "High swap usage detected"
                elif status == "warning":
                    message += " and high swap usage"
            
            return HealthCheckResult(
                check_name="memory_usage",
                status=status,
                message=message,
                details={
                    'memory_percent': memory.percent,
                    'memory_used_gb': memory.used / (1024**3),
                    'swap_percent': swap.percent,
                    'swap_used_gb': swap.used / (1024**3)
                },
                timestamp=time.time(),
                execution_time=0.0
            )
            
        except ImportError:
            return HealthCheckResult(
                check_name="memory_usage",
                status="warning",
                message="Cannot monitor memory usage - psutil not available",
                details={},
                timestamp=time.time(),
                execution_time=0.0
            )
    
    def _check_quality_gate_performance(self) -> HealthCheckResult:
        """Check quality gate performance trends."""
        try:
            # Get recent quality gate metrics
            metrics = self._get_recent_quality_gate_metrics(hours=24)
            
            if not metrics:
                return HealthCheckResult(
                    check_name="quality_gate_performance",
                    status="healthy",
                    message="No recent quality gate executions to analyze",
                    details={},
                    timestamp=time.time(),
                    execution_time=0.0
                )
            
            # Analyze performance trends
            total_executions = sum(m.execution_count for m in metrics)
            total_failures = sum(m.failure_count for m in metrics)
            failure_rate = (total_failures / total_executions * 100) if total_executions > 0 else 0
            
            avg_execution_time = sum(m.average_execution_time for m in metrics) / len(metrics)
            
            status = "healthy"
            message = f"Quality gates: {failure_rate:.1f}% failure rate"
            
            if failure_rate > 50:
                status = "critical"
                message = "High quality gate failure rate"
            elif failure_rate > 20:
                status = "warning"
                message = "Elevated quality gate failure rate"
            
            # Check for slow execution times
            if avg_execution_time > 300:  # 5 minutes
                if status == "healthy":
                    status = "warning"
                    message += ", slow execution times"
                else:
                    message += " and slow execution times"
            
            return HealthCheckResult(
                check_name="quality_gate_performance",
                status=status,
                message=message,
                details={
                    'total_executions': total_executions,
                    'total_failures': total_failures,
                    'failure_rate_percent': failure_rate,
                    'average_execution_time': avg_execution_time,
                    'gates_analyzed': len(metrics)
                },
                timestamp=time.time(),
                execution_time=0.0
            )
            
        except Exception as e:
            return HealthCheckResult(
                check_name="quality_gate_performance",
                status="warning",
                message=f"Cannot analyze quality gate performance: {e}",
                details={'error': str(e)},
                timestamp=time.time(),
                execution_time=0.0
            )
    
    def _check_database_connectivity(self) -> HealthCheckResult:
        """Check database connectivity and health."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM health_checks")
                row_count = cursor.fetchone()[0]
            
            return HealthCheckResult(
                check_name="database_connectivity",
                status="healthy",
                message=f"Database accessible ({row_count} health check records)",
                details={'row_count': row_count},
                timestamp=time.time(),
                execution_time=0.0
            )
            
        except Exception as e:
            return HealthCheckResult(
                check_name="database_connectivity",
                status="critical",
                message=f"Database connectivity issue: {e}",
                details={'error': str(e)},
                timestamp=time.time(),
                execution_time=0.0
            )
    
    def _check_file_permissions(self) -> HealthCheckResult:
        """Check file permissions for critical directories."""
        critical_paths = [
            Path("."),
            Path("./quality_reports"),
            Path("./quality_cache"),
            self.db_path.parent
        ]
        
        permission_issues = []
        
        for path in critical_paths:
            if path.exists():
                if not os.access(path, os.R_OK):
                    permission_issues.append(f"{path}: not readable")
                if not os.access(path, os.W_OK):
                    permission_issues.append(f"{path}: not writable")
        
        if permission_issues:
            status = "critical" if len(permission_issues) > 2 else "warning"
            message = f"File permission issues: {len(permission_issues)} found"
        else:
            status = "healthy"
            message = "File permissions OK"
        
        return HealthCheckResult(
            check_name="file_permissions",
            status=status,
            message=message,
            details={'permission_issues': permission_issues},
            timestamp=time.time(),
            execution_time=0.0
        )
    
    def _collect_system_metrics(self):
        """Collect and store system metrics."""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            # Get disk usage for current directory
            try:
                import shutil
                total, used, free = shutil.disk_usage(".")
                disk_percent = (used / total) * 100
            except:
                disk_percent = 0
            
            # Get load average (Unix-like systems only)
            try:
                load_avg = os.getloadavg()[0]
            except (AttributeError, OSError):
                load_avg = 0
            
            metrics = SystemMetrics(
                cpu_usage_percent=cpu_percent,
                memory_usage_percent=memory.percent,
                disk_usage_percent=disk_percent,
                load_average=load_avg,
                timestamp=time.time()
            )
            
            self._store_system_metrics(metrics)
            
        except ImportError:
            # psutil not available
            pass
        except Exception as e:
            self.logger.warning(f"Failed to collect system metrics: {e}")
    
    def _store_health_check_result(self, result: HealthCheckResult):
        """Store health check result in database."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    INSERT INTO health_checks 
                    (check_name, status, message, details, timestamp, execution_time)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    result.check_name,
                    result.status,
                    result.message,
                    json.dumps(result.details),
                    result.timestamp,
                    result.execution_time
                ))
        except Exception as e:
            self.logger.error(f"Failed to store health check result: {e}")
    
    def _store_system_metrics(self, metrics: SystemMetrics):
        """Store system metrics in database."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    INSERT INTO system_metrics 
                    (cpu_usage, memory_usage, disk_usage, load_average, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    metrics.cpu_usage_percent,
                    metrics.memory_usage_percent,
                    metrics.disk_usage_percent,
                    metrics.load_average,
                    metrics.timestamp
                ))
        except Exception as e:
            self.logger.error(f"Failed to store system metrics: {e}")
    
    def record_quality_gate_execution(
        self, 
        gate_name: str, 
        success: bool, 
        execution_time: float
    ):
        """Record quality gate execution for monitoring."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Get existing metrics
                cursor = conn.execute("""
                    SELECT execution_count, success_count, failure_count, average_execution_time
                    FROM quality_gate_metrics 
                    WHERE gate_name = ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, (gate_name,))
                
                row = cursor.fetchone()
                
                if row:
                    exec_count, success_count, failure_count, avg_time = row
                    new_exec_count = exec_count + 1
                    new_success_count = success_count + (1 if success else 0)
                    new_failure_count = failure_count + (0 if success else 1)
                    new_avg_time = ((avg_time * exec_count) + execution_time) / new_exec_count
                else:
                    new_exec_count = 1
                    new_success_count = 1 if success else 0
                    new_failure_count = 0 if success else 1
                    new_avg_time = execution_time
                
                # Store updated metrics
                conn.execute("""
                    INSERT INTO quality_gate_metrics 
                    (gate_name, execution_count, success_count, failure_count, 
                     average_execution_time, last_execution_time, last_status, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    gate_name,
                    new_exec_count,
                    new_success_count,
                    new_failure_count,
                    new_avg_time,
                    execution_time,
                    "success" if success else "failure",
                    time.time()
                ))
                
        except Exception as e:
            self.logger.error(f"Failed to record quality gate execution: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current overall health status."""
        results = self.run_health_checks()
        
        # Determine overall status
        statuses = [r.status for r in results]
        if "critical" in statuses:
            overall_status = "critical"
        elif "warning" in statuses:
            overall_status = "warning"
        else:
            overall_status = "healthy"
        
        return {
            'overall_status': overall_status,
            'timestamp': time.time(),
            'checks': [asdict(r) for r in results],
            'summary': {
                'total_checks': len(results),
                'healthy': len([r for r in results if r.status == "healthy"]),
                'warning': len([r for r in results if r.status == "warning"]),
                'critical': len([r for r in results if r.status == "critical"])
            }
        }
    
    def _get_recent_quality_gate_metrics(self, hours: int = 24) -> List[QualityGateMetrics]:
        """Get recent quality gate metrics."""
        cutoff_time = time.time() - (hours * 3600)
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("""
                    SELECT gate_name, execution_count, success_count, failure_count,
                           average_execution_time, last_execution_time, last_status, timestamp
                    FROM quality_gate_metrics
                    WHERE timestamp > ?
                    ORDER BY timestamp DESC
                """, (cutoff_time,))
                
                metrics = []
                for row in cursor.fetchall():
                    metrics.append(QualityGateMetrics(
                        gate_name=row[0],
                        execution_count=row[1],
                        success_count=row[2],
                        failure_count=row[3],
                        average_execution_time=row[4],
                        last_execution_time=row[5],
                        last_status=row[6],
                        timestamp=row[7]
                    ))
                
                return metrics
                
        except Exception as e:
            self.logger.error(f"Failed to get quality gate metrics: {e}")
            return []
    
    def generate_health_report(self, output_file: str = None) -> str:
        """Generate comprehensive health report."""
        health_status = self.get_health_status()
        
        report = []
        report.append("DGDM QUALITY GATES HEALTH REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Overall Status: {health_status['overall_status'].upper()}")
        report.append("")
        
        # Summary
        summary = health_status['summary']
        report.append(f"Health Check Summary:")
        report.append(f"  Total Checks: {summary['total_checks']}")
        report.append(f"  Healthy: {summary['healthy']}")
        report.append(f"  Warnings: {summary['warning']}")
        report.append(f"  Critical: {summary['critical']}")
        report.append("")
        
        # Individual check results
        report.append("Individual Health Checks:")
        report.append("-" * 30)
        
        for check in health_status['checks']:
            status_symbol = {
                'healthy': '✅',
                'warning': '⚠️',
                'critical': '❌'
            }.get(check['status'], '❓')
            
            report.append(f"{status_symbol} {check['check_name']}: {check['message']}")
            if check['details']:
                for key, value in check['details'].items():
                    if isinstance(value, float):
                        report.append(f"   {key}: {value:.2f}")
                    else:
                        report.append(f"   {key}: {value}")
            report.append("")
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def cleanup_old_data(self, days: int = 30):
        """Clean up old monitoring data."""
        cutoff_time = time.time() - (days * 24 * 3600)
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Clean up old health checks
                conn.execute("DELETE FROM health_checks WHERE timestamp < ?", (cutoff_time,))
                
                # Clean up old system metrics
                conn.execute("DELETE FROM system_metrics WHERE timestamp < ?", (cutoff_time,))
                
                # Clean up old quality gate metrics (keep aggregated data)
                conn.execute("DELETE FROM quality_gate_metrics WHERE timestamp < ?", (cutoff_time,))
                
                self.logger.info(f"Cleaned up monitoring data older than {days} days")
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {e}")


def main():
    """Main entry point for health monitoring."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DGDM Quality Gates Health Monitor")
    parser.add_argument("--start", action="store_true", help="Start monitoring daemon")
    parser.add_argument("--status", action="store_true", help="Show current health status")
    parser.add_argument("--report", help="Generate health report to file")
    parser.add_argument("--interval", type=int, default=60, help="Monitoring interval in seconds")
    parser.add_argument("--cleanup", type=int, help="Clean up data older than N days")
    
    args = parser.parse_args()
    
    monitor = HealthMonitor()
    
    if args.start:
        print(f"Starting health monitoring with {args.interval}s interval...")
        monitor.start_monitoring(args.interval)
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            print("Stopping monitoring...")
            monitor.stop_monitoring()
    
    elif args.status:
        status = monitor.get_health_status()
        print(f"Overall Status: {status['overall_status'].upper()}")
        print(f"Health Checks: {status['summary']['healthy']} healthy, "
              f"{status['summary']['warning']} warnings, "
              f"{status['summary']['critical']} critical")
    
    elif args.report:
        report = monitor.generate_health_report(args.report)
        print(f"Health report generated: {args.report}")
        print("\nReport preview:")
        print(report[:500] + "..." if len(report) > 500 else report)
    
    elif args.cleanup:
        monitor.cleanup_old_data(args.cleanup)
        print(f"Cleaned up data older than {args.cleanup} days")
    
    else:
        # Default: show current status
        status = monitor.get_health_status()
        print(monitor.generate_health_report())


if __name__ == "__main__":
    main()