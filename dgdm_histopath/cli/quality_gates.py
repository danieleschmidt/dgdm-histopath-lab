"""CLI for progressive quality gates."""

import typer
import sys
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..testing.progressive_quality_gates import (
    ProgressiveQualityRunner, 
    ProgressiveQualityConfig, 
    ProjectMaturity
)
from ..testing.robust_quality_runner import RobustQualityRunner
from ..testing.scalable_quality_gates import ScalableQualityGates
from ..utils.logging import setup_logging, get_logger

app = typer.Typer(help="Progressive Quality Gates for DGDM Histopath Lab")
console = Console()


@app.command()
def run(
    maturity: Optional[str] = typer.Option(
        None, 
        help="Project maturity level (auto-detected if not specified)",
        click_type=typer.Choice(['greenfield', 'development', 'staging', 'production'])
    ),
    output_dir: str = typer.Option(
        "./quality_reports", 
        help="Output directory for reports"
    ),
    parallel: bool = typer.Option(
        True, 
        help="Run quality gates in parallel"
    ),
    verbose: bool = typer.Option(
        False, 
        help="Enable verbose logging"
    ),
    gates: Optional[List[str]] = typer.Option(
        None, 
        help="Specific gates to run (default: all for maturity level)"
    ),
    robust: bool = typer.Option(
        False,
        help="Use robust quality runner with advanced error handling"
    ),
    recovery: bool = typer.Option(
        True,
        help="Enable automatic recovery attempts (robust mode only)"
    ),
    scalable: bool = typer.Option(
        False,
        help="Use scalable quality gates with optimization and caching"
    ),
    cache_dir: str = typer.Option(
        "./quality_cache",
        help="Cache directory for scalable mode"
    ),
    distributed: bool = typer.Option(
        False,
        help="Enable distributed processing (scalable mode only)"
    ),
    workers: Optional[int] = typer.Option(
        None,
        help="Number of worker processes (scalable mode only)"
    )
):
    """Run progressive quality gates."""
    
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level, log_file=None)
    logger = get_logger(__name__)
    
    # Create configuration
    config = ProgressiveQualityConfig()
    if maturity:
        config.maturity = ProjectMaturity(maturity)
    
    if scalable:
        # Use scalable quality gates with optimization
        quality_gates = ScalableQualityGates(
            config=config,
            output_dir=output_dir,
            cache_dir=cache_dir,
            enable_caching=True,
            enable_distributed=distributed,
            max_workers=workers
        )
        
        # Display startup information
        console.print(Panel.fit(
            f"[bold blue]DGDM Scalable Quality Gates[/bold blue]\n"
            f"Maturity Level: [bold green]{config.maturity.value.title()}[/bold green]\n"
            f"Output Directory: [italic]{output_dir}[/italic]\n"
            f"Cache Directory: [italic]{cache_dir}[/italic]\n"
            f"Distributed Processing: [{'green' if distributed else 'red'}]{'Enabled' if distributed else 'Disabled'}[/]\n"
            f"Workers: {workers or 'Auto'}",
            title="⚡ Starting Scalable Validation"
        ))
        
        # Run gates with progress indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Running scalable quality gates...", total=None)
            
            try:
                results = quality_gates.run_optimized_validation(gates)
            except Exception as e:
                console.print(f"[red]Error running scalable quality gates: {e}[/red]")
                sys.exit(1)
        
        # Display scalable results
        display_scalable_results(results, config.maturity, quality_gates.get_optimization_metrics())
        
        # Check for failures
        overall_passed = all(r.passed for r in results)
        if overall_passed:
            console.print(f"\n[bold green]✅ All scalable quality gates PASSED[/bold green]")
            sys.exit(0)
        else:
            console.print(f"\n[bold red]❌ Some scalable quality gates FAILED[/bold red]")
            sys.exit(1)
    
    elif robust:
        # Use robust quality runner
        with RobustQualityRunner(
            config=config, 
            output_dir=output_dir,
            enable_recovery=recovery
        ) as runner:
            
            # Display startup information
            console.print(Panel.fit(
                f"[bold blue]DGDM Robust Quality Gates[/bold blue]\n"
                f"Maturity Level: [bold green]{config.maturity.value.title()}[/bold green]\n"
                f"Output Directory: [italic]{output_dir}[/italic]\n"
                f"Recovery Mode: [{'green' if recovery else 'red'}]{'Enabled' if recovery else 'Disabled'}[/]",
                title="🛡️  Starting Robust Validation"
            ))
            
            # Run gates with progress indicator
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("Running robust quality gates...", total=None)
                
                try:
                    results = runner.run_validation(validators=gates, parallel=parallel)
                except Exception as e:
                    console.print(f"[red]Error running robust quality gates: {e}[/red]")
                    sys.exit(1)
            
            # Display robust results
            display_robust_results(results, config.maturity, runner.get_summary())
            
            # Check for failures
            if runner.has_failures():
                failed_validators = runner.get_failed_validators()
                console.print(f"\n[bold red]❌ Failed validators: {', '.join(failed_validators)}[/bold red]")
                sys.exit(1)
            else:
                console.print(f"\n[bold green]✅ All robust quality gates PASSED[/bold green]")
                sys.exit(0)
    else:
        # Use original progressive quality runner
        runner = ProgressiveQualityRunner(config=config, output_dir=output_dir)
        
        # Display startup information
        console.print(Panel.fit(
            f"[bold blue]DGDM Progressive Quality Gates[/bold blue]\n"
            f"Maturity Level: [bold green]{config.maturity.value.title()}[/bold green]\n"
            f"Output Directory: [italic]{output_dir}[/italic]",
            title="🚀 Starting Quality Validation"
        ))
        
        # Run gates with progress indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Running quality gates...", total=None)
            
            try:
                results = runner.run_progressive_gates(parallel=parallel)
            except Exception as e:
                console.print(f"[red]Error running quality gates: {e}[/red]")
                sys.exit(1)
        
        # Display results
        display_results(results, config.maturity)
    
    # Exit with appropriate code
    overall_passed = all(r.passed for r in results)
    if overall_passed:
        console.print(f"\n[bold green]✅ All quality gates PASSED[/bold green]")
        sys.exit(0)
    else:
        console.print(f"\n[bold red]❌ Some quality gates FAILED[/bold red]")
        sys.exit(1)


@app.command()
def status():
    """Show project maturity status and recommended gates."""
    
    # Auto-detect maturity
    config = ProgressiveQualityConfig()
    runner = ProgressiveQualityRunner(config=config)
    
    console.print(Panel.fit(
        f"[bold blue]Project Maturity Analysis[/bold blue]\n"
        f"Detected Level: [bold green]{config.maturity.value.title()}[/bold green]",
        title="📊 Current Status"
    ))
    
    # Show enabled gates for current maturity
    enabled_gates = config.enabled_gates.get(config.maturity, [])
    
    table = Table(title="Quality Gates for Current Maturity Level")
    table.add_column("Gate Name", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Threshold", style="yellow")
    
    gate_descriptions = {
        'code_compilation': ('Code compilation without errors', 'Zero syntax errors'),
        'basic_tests': ('Basic test execution', 'All tests pass'),
        'model_validation': ('Model instantiation and inference', 'Successful forward pass'),
        'code_coverage': ('Test coverage analysis', f"{config.test_coverage_thresholds[config.maturity]:.1f}%"),
        'performance_basic': ('Basic performance benchmarks', f"{config.performance_thresholds[config.maturity]['inference_time']:.1f}s inference"),
        'security_basic': ('Basic security scanning', f"Max {config.security_thresholds[config.maturity]['vulnerabilities']} vulnerabilities"),
        'integration_tests': ('Integration test execution', 'All integration tests pass'),
        'performance_advanced': ('Advanced performance testing', 'Memory and throughput optimization'),
        'security_advanced': ('Advanced security analysis', 'Comprehensive vulnerability assessment'),
        'documentation': ('Documentation quality check', 'Complete API documentation'),
        'resource_usage': ('Resource efficiency validation', 'Optimal resource utilization'),
        'compliance_checks': ('Regulatory compliance validation', 'FDA/CE compliance ready'),
        'disaster_recovery': ('Disaster recovery readiness', 'Complete backup/recovery procedures'),
        'monitoring_health': ('Monitoring and observability', 'Full observability stack')
    }
    
    for gate_name in enabled_gates:
        desc, threshold = gate_descriptions.get(gate_name, ('Unknown gate', 'N/A'))
        table.add_row(gate_name, desc, threshold)
    
    console.print(table)
    
    # Show maturity progression
    show_maturity_progression(config.maturity)


@app.command()
def upgrade():
    """Show recommendations for upgrading to next maturity level."""
    
    config = ProgressiveQualityConfig()
    runner = ProgressiveQualityRunner(config=config)
    
    recommendations = runner._get_next_maturity_recommendations()
    
    current_level = config.maturity.value.title()
    next_levels = {
        ProjectMaturity.GREENFIELD: "Development",
        ProjectMaturity.DEVELOPMENT: "Staging", 
        ProjectMaturity.STAGING: "Production",
        ProjectMaturity.PRODUCTION: "Production (Maintain)"
    }
    next_level = next_levels.get(config.maturity, "Unknown")
    
    console.print(Panel.fit(
        f"[bold blue]Maturity Upgrade Path[/bold blue]\n"
        f"Current: [bold green]{current_level}[/bold green]\n"
        f"Next Level: [bold yellow]{next_level}[/bold yellow]",
        title="📈 Upgrade Recommendations"
    ))
    
    if recommendations:
        console.print(f"\n[bold]To advance to {next_level} level:[/bold]")
        for i, rec in enumerate(recommendations, 1):
            console.print(f"{i}. {rec}")
    else:
        console.print("\n[bold green]You're at the highest maturity level![/bold green]")


@app.command()
def benchmark():
    """Run performance benchmarks for quality gates."""
    
    console.print(Panel.fit(
        "[bold blue]Quality Gate Performance Benchmark[/bold blue]\n"
        "This will measure the execution time of each quality gate.",
        title="⚡ Benchmarking"
    ))
    
    # Create runner with all maturity levels to test
    results_by_maturity = {}
    
    for maturity in ProjectMaturity:
        config = ProgressiveQualityConfig()
        config.maturity = maturity
        runner = ProgressiveQualityRunner(config=config, output_dir="./benchmark_reports")
        
        console.print(f"\n[bold]Benchmarking {maturity.value.title()} level...[/bold]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task(f"Running {maturity.value} gates...", total=None)
            results = runner.run_progressive_gates(parallel=False)  # Sequential for accurate timing
        
        results_by_maturity[maturity] = results
    
    # Display benchmark results
    display_benchmark_results(results_by_maturity)


def display_results(results, maturity_level):
    """Display quality gate results in a formatted table."""
    
    table = Table(title=f"Quality Gate Results - {maturity_level.value.title()} Level")
    table.add_column("Gate", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Score", justify="right", style="yellow")
    table.add_column("Threshold", justify="right", style="blue")
    table.add_column("Message", style="white")
    table.add_column("Time", justify="right", style="magenta")
    
    for result in results:
        status = "✅ PASS" if result.passed else "❌ FAIL"
        status_style = "bold green" if result.passed else "bold red"
        
        table.add_row(
            result.gate_name,
            f"[{status_style}]{status}[/{status_style}]",
            f"{result.score:.2f}",
            f"{result.threshold:.2f}",
            result.message[:50] + "..." if len(result.message) > 50 else result.message,
            f"{result.execution_time:.2f}s"
        )
    
    console.print(table)
    
    # Summary statistics
    passed_count = len([r for r in results if r.passed])
    total_count = len(results)
    total_time = sum(r.execution_time for r in results)
    
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"Passed: {passed_count}/{total_count}")
    console.print(f"Success Rate: {(passed_count/total_count)*100:.1f}%")
    console.print(f"Total Execution Time: {total_time:.2f}s")


def show_maturity_progression(current_maturity):
    """Show maturity level progression path."""
    
    maturity_levels = [
        ("🌱 Greenfield", "Basic functionality, minimal requirements", current_maturity == ProjectMaturity.GREENFIELD),
        ("🔨 Development", "Active development, moderate requirements", current_maturity == ProjectMaturity.DEVELOPMENT), 
        ("🧪 Staging", "Pre-production, strict requirements", current_maturity == ProjectMaturity.STAGING),
        ("🚀 Production", "Production-ready, strictest requirements", current_maturity == ProjectMaturity.PRODUCTION)
    ]
    
    console.print(f"\n[bold]Maturity Progression Path:[/bold]")
    
    for level_name, description, is_current in maturity_levels:
        if is_current:
            console.print(f"👉 [bold green]{level_name}[/bold green] - {description} [bold green](CURRENT)[/bold green]")
        else:
            console.print(f"   {level_name} - {description}")


def display_scalable_results(results, maturity_level, optimization_metrics):
    """Display scalable quality gate results with optimization metrics."""
    
    table = Table(title=f"Scalable Quality Gate Results - {maturity_level.value.title()} Level")
    table.add_column("Gate", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Score", justify="right", style="yellow")
    table.add_column("Time", justify="right", style="green")
    table.add_column("Memory", justify="right", style="blue")
    table.add_column("Message", style="white")
    
    for result in results:
        status = "✅ PASS" if result.passed else "❌ FAIL"
        status_style = "bold green" if result.passed else "bold red"
        
        table.add_row(
            result.validator_name,
            f"[{status_style}]{status}[/{status_style}]",
            f"{result.score:.1f}/{result.threshold:.1f}",
            f"{result.execution_time:.2f}s",
            f"{result.memory_peak_mb:.1f}MB",
            result.message[:50] + "..." if len(result.message) > 50 else result.message
        )
    
    console.print(table)
    
    # Optimization metrics
    console.print(f"\n[bold]Optimization Metrics:[/bold]")
    console.print(f"Cache Hits: {optimization_metrics.cache_hits}")
    console.print(f"Cache Misses: {optimization_metrics.cache_misses}")
    
    if optimization_metrics.cache_hits > 0:
        hit_rate = optimization_metrics.cache_hits / (optimization_metrics.cache_hits + optimization_metrics.cache_misses) * 100
        console.print(f"Cache Hit Rate: {hit_rate:.1f}%")
        console.print(f"Time Saved by Caching: {optimization_metrics.total_optimization_time_saved:.2f}s")
    
    if optimization_metrics.parallel_speedup > 1:
        console.print(f"Parallel Speedup: {optimization_metrics.parallel_speedup:.1f}x")
    
    if optimization_metrics.distributed_nodes > 0:
        console.print(f"Distributed Nodes: {optimization_metrics.distributed_nodes}")
    
    console.print(f"Memory Optimization: {optimization_metrics.memory_optimization * 100:.1f}%")


def display_robust_results(results, maturity_level, summary):
    """Display robust quality gate results with comprehensive information."""
    
    table = Table(title=f"Robust Quality Gate Results - {maturity_level.value.title()} Level")
    table.add_column("Gate", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Score", justify="right", style="yellow")
    table.add_column("Memory", justify="right", style="blue")
    table.add_column("Time", justify="right", style="green")
    table.add_column("Recovery", justify="center", style="magenta")
    table.add_column("Message", style="white")
    
    for result in results:
        status = "✅ PASS" if result.passed else "❌ FAIL"
        status_style = "bold green" if result.passed else "bold red"
        
        # Recovery status
        if result.recovery_attempted:
            recovery_status = "✅" if result.recovery_successful else "❌"
        else:
            recovery_status = "➖"
        
        table.add_row(
            result.validator_name,
            f"[{status_style}]{status}[/{status_style}]",
            f"{result.score:.1f}/{result.threshold:.1f}",
            f"{result.memory_peak_mb:.1f}MB",
            f"{result.execution_time:.2f}s",
            recovery_status,
            result.message[:40] + "..." if len(result.message) > 40 else result.message
        )
    
    console.print(table)
    
    # Summary statistics
    console.print(f"\n[bold]Execution Summary:[/bold]")
    console.print(f"Validation ID: {summary.get('validation_id', 'N/A')}")
    console.print(f"Total Execution Time: {summary.get('total_execution_time', 0):.2f}s")
    console.print(f"Peak Memory Usage: {summary.get('peak_memory_usage_mb', 0):.1f}MB")
    console.print(f"Recovery Attempts: {summary.get('recovery_attempts', 0)}")
    console.print(f"Successful Recoveries: {summary.get('successful_recoveries', 0)}")
    console.print(f"Total Warnings: {summary.get('total_warnings', 0)}")
    console.print(f"Total Errors: {summary.get('total_errors', 0)}")


def display_benchmark_results(results_by_maturity):
    """Display benchmark results comparing different maturity levels."""
    
    table = Table(title="Quality Gate Performance Benchmark")
    table.add_column("Maturity Level", style="cyan")
    table.add_column("Gates Run", justify="right", style="yellow")
    table.add_column("Total Time", justify="right", style="blue")
    table.add_column("Average Time", justify="right", style="green")
    table.add_column("Success Rate", justify="right", style="magenta")
    
    for maturity, results in results_by_maturity.items():
        gates_count = len(results)
        total_time = sum(r.execution_time for r in results)
        avg_time = total_time / gates_count if gates_count > 0 else 0
        success_rate = (len([r for r in results if r.passed]) / gates_count * 100) if gates_count > 0 else 0
        
        table.add_row(
            maturity.value.title(),
            str(gates_count),
            f"{total_time:.2f}s",
            f"{avg_time:.2f}s",
            f"{success_rate:.1f}%"
        )
    
    console.print(table)


if __name__ == "__main__":
    app()