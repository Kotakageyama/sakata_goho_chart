#!/bin/bash
# Agent Workflow Shell Wrapper
# This script provides a convenient interface for LLM agents

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

show_help() {
    cat << EOF
🤖 Agent Workflow for Model Development

USAGE:
    ./scripts/agent_workflow.sh <command> [options]

COMMANDS:
    create <strategy> [date]     Create a new model branch
    run <strategy> [date]        Run complete workflow  
    status                       Show current status
    list                         List model branches
    validate <branch>            Validate branch name
    help                         Show this help

EXAMPLES:
    # Create branch for LSTM strategy
    ./scripts/agent_workflow.sh create lstm

    # Run complete workflow for transformer strategy
    ./scripts/agent_workflow.sh run transformer

    # Create branch with specific date
    ./scripts/agent_workflow.sh create ensemble 20241201

    # Check current status
    ./scripts/agent_workflow.sh status

BRANCH NAMING:
    Branches follow the pattern: model/<strategy>/<yyyymmdd>
    
    Examples:
    - model/lstm/20241201
    - model/transformer/20241202
    - model/ensemble/20241203

EOF
}

check_dependencies() {
    log "Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check git
    if ! command -v git &> /dev/null; then
        error "Git is required but not installed"
        exit 1
    fi
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        error "Not in a git repository"
        exit 1
    fi
    
    success "Dependencies OK"
}

run_python_workflow() {
    local cmd="$1"
    shift
    
    cd "$PROJECT_ROOT"
    
    if [[ "$cmd" == "run" ]]; then
        log "Starting complete workflow for strategy: $1"
        python3 scripts/agent_workflow.py run "$@"
    elif [[ "$cmd" == "status" ]]; then
        log "Getting workflow status..."
        python3 scripts/agent_workflow.py status
    elif [[ "$cmd" == "validate" ]]; then
        log "Validating branch name: $1"
        python3 scripts/agent_workflow.py validate "$1"
    fi
}

create_branch() {
    local strategy="$1"
    local date="$2"
    
    if [[ -z "$strategy" ]]; then
        error "Strategy name is required"
        exit 1
    fi
    
    log "Creating branch for strategy: $strategy"
    
    cd "$PROJECT_ROOT"
    
    if [[ -n "$date" ]]; then
        python3 scripts/branch_manager.py create "$strategy" "$date"
    else
        python3 scripts/branch_manager.py create "$strategy"
    fi
    
    success "Branch created successfully"
}

list_branches() {
    log "Listing model branches..."
    cd "$PROJECT_ROOT"
    python3 scripts/branch_manager.py list
}

main() {
    if [[ $# -eq 0 ]]; then
        show_help
        exit 0
    fi
    
    local command="$1"
    shift
    
    case "$command" in
        create)
            check_dependencies
            create_branch "$@"
            ;;
        run)
            check_dependencies
            run_python_workflow "run" "$@"
            ;;
        status)
            check_dependencies
            run_python_workflow "status" "$@"
            ;;
        list)
            check_dependencies
            list_branches
            ;;
        validate)
            check_dependencies
            run_python_workflow "validate" "$@"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            error "Unknown command: $command"
            echo
            show_help
            exit 1
            ;;
    esac
}

main "$@"