#!/bin/bash
# DGDM Histopath Lab Deployment Script
# Comprehensive deployment automation for various environments

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VERSION="${VERSION:-latest}"
ENVIRONMENT="${ENVIRONMENT:-development}"
NAMESPACE="${NAMESPACE:-dgdm-histopath}"
REGISTRY="${REGISTRY:-ghcr.io/your-org}"
IMAGE_NAME="${IMAGE_NAME:-dgdm-histopath}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
DGDM Histopath Lab Deployment Script

Usage: $0 [OPTIONS] COMMAND

Commands:
    build                Build Docker images
    push                 Push Docker images to registry
    deploy-docker        Deploy using Docker Compose
    deploy-k8s           Deploy to Kubernetes
    deploy-local         Deploy locally for development
    test                 Run deployment tests
    cleanup              Clean up resources
    help                 Show this help message

Options:
    -e, --environment    Deployment environment (development|staging|production)
    -v, --version        Version tag for images
    -n, --namespace      Kubernetes namespace
    -r, --registry       Docker registry URL
    --skip-build         Skip building images
    --skip-tests         Skip running tests
    --dry-run           Show what would be deployed without actually deploying
    --force             Force deployment even if checks fail

Environment Variables:
    DGDM_SECRET_KEY     Secret key for encryption
    GRAFANA_PASSWORD    Grafana admin password
    DATABASE_PASSWORD   Database password

Examples:
    $0 build
    $0 deploy-docker -e production
    $0 deploy-k8s -v v1.0.0 -n dgdm-prod
    $0 cleanup -e staging

EOF
}

# Utility functions
check_dependencies() {
    local deps=("docker" "docker-compose")
    
    if [[ "$1" == "k8s" ]]; then
        deps+=("kubectl" "helm")
    fi
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            log_error "$dep is required but not installed"
            exit 1
        fi
    done
}

check_environment() {
    local valid_envs=("development" "staging" "production")
    
    if [[ ! " ${valid_envs[@]} " =~ " ${ENVIRONMENT} " ]]; then
        log_error "Invalid environment: $ENVIRONMENT. Valid options: ${valid_envs[*]}"
        exit 1
    fi
}

setup_environment() {
    log_info "Setting up environment: $ENVIRONMENT"
    
    # Create necessary directories
    mkdir -p "$PROJECT_ROOT"/{data,outputs,logs,cache}
    
    # Set up environment-specific configurations
    case "$ENVIRONMENT" in
        development)
            export DGDM_LOG_LEVEL=DEBUG
            export DGDM_SECRET_KEY="${DGDM_SECRET_KEY:-dev-secret-key}"
            export GRAFANA_PASSWORD="${GRAFANA_PASSWORD:-admin123}"
            ;;
        staging)
            export DGDM_LOG_LEVEL=INFO
            if [[ -z "${DGDM_SECRET_KEY:-}" ]]; then
                log_error "DGDM_SECRET_KEY must be set for staging environment"
                exit 1
            fi
            ;;
        production)
            export DGDM_LOG_LEVEL=WARNING
            if [[ -z "${DGDM_SECRET_KEY:-}" ]]; then
                log_error "DGDM_SECRET_KEY must be set for production environment"
                exit 1
            fi
            if [[ -z "${GRAFANA_PASSWORD:-}" ]]; then
                log_error "GRAFANA_PASSWORD must be set for production environment"
                exit 1
            fi
            ;;
    esac
}

# Build functions
build_images() {
    log_info "Building Docker images for $ENVIRONMENT environment..."
    
    cd "$PROJECT_ROOT"
    
    # Build main application image
    log_info "Building application image..."
    docker build \
        --target production \
        --tag "$REGISTRY/$IMAGE_NAME:$VERSION" \
        --tag "$REGISTRY/$IMAGE_NAME:latest" \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VCS_REF="$(git rev-parse HEAD)" \
        --build-arg VERSION="$VERSION" \
        .
    
    # Build development image if needed
    if [[ "$ENVIRONMENT" == "development" ]]; then
        log_info "Building development image..."
        docker build \
            --target development \
            --tag "$REGISTRY/$IMAGE_NAME:dev" \
            .
    fi
    
    log_success "Images built successfully"
}

push_images() {
    log_info "Pushing images to registry: $REGISTRY"
    
    docker push "$REGISTRY/$IMAGE_NAME:$VERSION"
    docker push "$REGISTRY/$IMAGE_NAME:latest"
    
    if [[ "$ENVIRONMENT" == "development" ]]; then
        docker push "$REGISTRY/$IMAGE_NAME:dev"
    fi
    
    log_success "Images pushed successfully"
}

# Deployment functions
deploy_docker() {
    log_info "Deploying with Docker Compose..."
    
    cd "$PROJECT_ROOT"
    
    # Create environment file
    cat > .env << EOF
DGDM_SECRET_KEY=$DGDM_SECRET_KEY
GRAFANA_PASSWORD=$GRAFANA_PASSWORD
DATABASE_PASSWORD=${DATABASE_PASSWORD:-dgdm_password}
ENVIRONMENT=$ENVIRONMENT
VERSION=$VERSION
EOF
    
    # Deploy based on environment
    case "$ENVIRONMENT" in
        development)
            docker-compose --profile development up -d
            ;;
        *)
            docker-compose up -d
            ;;
    esac
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Health checks
    check_service_health "dgdm-app" "http://localhost:8000/health"
    check_service_health "grafana" "http://localhost:3000/api/health"
    check_service_health "postgres" "postgresql://dgdm:dgdm_password@localhost:5432/dgdm_db"
    
    log_success "Docker deployment completed successfully"
    log_info "Services available at:"
    log_info "  - Application: http://localhost:8000"
    log_info "  - Monitoring: http://localhost:3000"
    log_info "  - File Browser: http://localhost:8080"
}

deploy_k8s() {
    log_info "Deploying to Kubernetes..."
    
    # Check kubectl context
    local context
    context=$(kubectl config current-context)
    log_info "Deploying to context: $context"
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        read -p "Are you sure you want to deploy to production? (yes/no): " confirm
        if [[ "$confirm" != "yes" ]]; then
            log_info "Deployment cancelled"
            exit 0
        fi
    fi
    
    cd "$PROJECT_ROOT"
    
    # Create namespace
    kubectl apply -f kubernetes/namespace.yaml
    
    # Create secrets
    create_k8s_secrets
    
    # Apply configurations
    kubectl apply -f kubernetes/configmap.yaml
    kubectl apply -f kubernetes/pvc.yaml
    kubectl apply -f kubernetes/deployment.yaml
    kubectl apply -f kubernetes/service.yaml
    kubectl apply -f kubernetes/ingress.yaml
    
    # Deploy monitoring stack
    deploy_monitoring_k8s
    
    # Wait for rollout
    log_info "Waiting for deployment rollout..."
    kubectl rollout status deployment/dgdm-app -n "$NAMESPACE" --timeout=600s
    kubectl rollout status deployment/dgdm-worker -n "$NAMESPACE" --timeout=600s
    
    # Verify deployment
    verify_k8s_deployment
    
    log_success "Kubernetes deployment completed successfully"
}

deploy_local() {
    log_info "Setting up local development environment..."
    
    cd "$PROJECT_ROOT"
    
    # Install Python dependencies
    if [[ ! -d "venv" ]]; then
        python3 -m venv venv
    fi
    
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install -e .
    
    # Install pre-commit hooks
    pre-commit install
    
    # Generate sample data
    python scripts/generate_sample_data.py
    
    # Run initial tests
    if [[ "${SKIP_TESTS:-false}" != "true" ]]; then
        pytest tests/ -v --tb=short
    fi
    
    log_success "Local development environment set up successfully"
    log_info "To activate the environment, run: source venv/bin/activate"
}

# Helper functions
create_k8s_secrets() {
    log_info "Creating Kubernetes secrets..."
    
    kubectl create secret generic dgdm-secrets \
        --from-literal=DGDM_SECRET_KEY="$DGDM_SECRET_KEY" \
        --from-literal=GRAFANA_PASSWORD="$GRAFANA_PASSWORD" \
        --from-literal=DATABASE_PASSWORD="${DATABASE_PASSWORD:-dgdm_password}" \
        --namespace="$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
}

deploy_monitoring_k8s() {
    log_info "Deploying monitoring stack..."
    
    # Add Prometheus Helm repo
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    
    # Install Prometheus stack
    helm upgrade --install prometheus-stack prometheus-community/kube-prometheus-stack \
        --namespace="$NAMESPACE" \
        --values="$PROJECT_ROOT/kubernetes/monitoring-values.yaml" \
        --wait
}

check_service_health() {
    local service_name="$1"
    local health_url="$2"
    local max_attempts=30
    local attempt=1
    
    log_info "Checking health of $service_name..."
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -s "$health_url" &> /dev/null; then
            log_success "$service_name is healthy"
            return 0
        fi
        
        log_info "Attempt $attempt/$max_attempts: $service_name not ready, waiting..."
        sleep 10
        ((attempt++))
    done
    
    log_error "$service_name failed to become healthy"
    return 1
}

verify_k8s_deployment() {
    log_info "Verifying Kubernetes deployment..."
    
    # Check pod status
    kubectl get pods -n "$NAMESPACE" -o wide
    
    # Check service endpoints
    kubectl get services -n "$NAMESPACE"
    
    # Check ingress
    kubectl get ingress -n "$NAMESPACE"
    
    # Test application endpoints
    local app_url
    app_url=$(kubectl get ingress dgdm-ingress -n "$NAMESPACE" -o jsonpath='{.spec.rules[0].host}')
    
    if [[ -n "$app_url" ]]; then
        log_info "Testing application at: $app_url"
        if curl -s "http://$app_url/health" &> /dev/null; then
            log_success "Application is responding"
        else
            log_warning "Application may not be fully ready yet"
        fi
    fi
}

run_tests() {
    log_info "Running deployment tests..."
    
    cd "$PROJECT_ROOT"
    
    # Run unit tests
    python -m pytest tests/ -v --tb=short
    
    # Run integration tests
    python -m pytest tests/ -k integration -v
    
    # Run quality gates
    python -m dgdm_histopath.testing.quality_gates
    
    log_success "All tests passed"
}

cleanup() {
    log_info "Cleaning up resources for environment: $ENVIRONMENT"
    
    case "$1" in
        docker)
            docker-compose down -v
            docker system prune -f
            ;;
        k8s)
            kubectl delete namespace "$NAMESPACE" --ignore-not-found=true
            ;;
        local)
            rm -rf venv/
            rm -rf __pycache__/
            rm -rf .pytest_cache/
            rm -rf htmlcov/
            ;;
        *)
            log_error "Unknown cleanup target: $1"
            exit 1
            ;;
    esac
    
    log_success "Cleanup completed"
}

# Main execution
main() {
    local command=""
    local skip_build=false
    local skip_tests=false
    local dry_run=false
    local force=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -r|--registry)
                REGISTRY="$2"
                shift 2
                ;;
            --skip-build)
                skip_build=true
                shift
                ;;
            --skip-tests)
                skip_tests=true
                shift
                ;;
            --dry-run)
                dry_run=true
                shift
                ;;
            --force)
                force=true
                shift
                ;;
            help|--help)
                show_help
                exit 0
                ;;
            build|push|deploy-docker|deploy-k8s|deploy-local|test|cleanup)
                command="$1"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Validate inputs
    if [[ -z "$command" ]]; then
        log_error "No command specified"
        show_help
        exit 1
    fi
    
    check_environment
    setup_environment
    
    # Execute command
    case "$command" in
        build)
            check_dependencies "docker"
            if [[ "$skip_build" != "true" ]]; then
                build_images
            else
                log_info "Skipping build as requested"
            fi
            ;;
        push)
            check_dependencies "docker"
            push_images
            ;;
        deploy-docker)
            check_dependencies "docker"
            if [[ "$skip_build" != "true" ]]; then
                build_images
            fi
            deploy_docker
            if [[ "$skip_tests" != "true" ]]; then
                run_tests
            fi
            ;;
        deploy-k8s)
            check_dependencies "k8s"
            deploy_k8s
            if [[ "$skip_tests" != "true" ]]; then
                run_tests
            fi
            ;;
        deploy-local)
            deploy_local
            ;;
        test)
            run_tests
            ;;
        cleanup)
            if [[ "$ENVIRONMENT" == "production" && "$force" != "true" ]]; then
                read -p "Are you sure you want to cleanup production? (yes/no): " confirm
                if [[ "$confirm" != "yes" ]]; then
                    log_info "Cleanup cancelled"
                    exit 0
                fi
            fi
            cleanup docker
            cleanup k8s
            cleanup local
            ;;
        *)
            log_error "Unknown command: $command"
            exit 1
            ;;
    esac
    
    log_success "Command '$command' completed successfully!"
}

# Execute main function
main "$@"