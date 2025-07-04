#!/usr/bin/env python3
"""
Agent Workflow for Automated Model Development

This script provides a complete workflow for LLM agents to:
1. Create model development branches
2. Execute training notebooks
3. Create pull requests

Usage:
    python scripts/agent_workflow.py run <strategy> [--date YYYYMMDD] [--config CONFIG_FILE]
    python scripts/agent_workflow.py status
    python scripts/agent_workflow.py validate <branch_name>
"""

import os
import sys
import json
import subprocess
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import argparse

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.branch_manager import BranchManager

class AgentWorkflow:
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.branch_manager = BranchManager()
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for the workflow"""
        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"agent_workflow_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(str(log_file)),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_config(self, config_file: Optional[str] = None) -> Dict[str, Any]:
        """Load workflow configuration"""
        default_config = {
            "notebooks": {
                "fetch_data": "01_fetch_data.ipynb",
                "train_model": "02_train_model.ipynb", 
                "backtest": "03_backtest.ipynb"
            },
            "execution_order": ["fetch_data", "train_model", "backtest"],
            "required_files": ["requirements.txt"],
            "output_dirs": ["data", "models", "results"],
            "git": {
                "auto_commit": True,
                "commit_message_template": "🤖 Automated model training: {strategy} on {date}",
                "pr_title_template": "🧠 Model Training Results: {strategy} ({date})",
                "pr_body_template": """
## 🤖 Automated Model Training Results

**Strategy:** {strategy}  
**Date:** {date}  
**Branch:** {branch}

### 📊 Training Summary
- Data fetching: ✅ Complete
- Model training: ✅ Complete  
- Backtesting: ✅ Complete

### 📁 Generated Files
{generated_files}

### 🔍 Next Steps
- [ ] Review model performance metrics
- [ ] Validate backtest results
- [ ] Consider deployment if metrics meet criteria

---
*This PR was generated automatically by the Agent Workflow system.*
                """.strip()
            }
        }
        
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                # Merge configurations (user config overrides defaults)
                default_config.update(user_config)
        
        return default_config
    
    def validate_environment(self) -> bool:
        """Validate that the environment is ready for workflow execution"""
        self.logger.info("Validating environment...")
        
        # Check required files
        required_files = ["requirements.txt", "notebooks/01_fetch_data.py", "notebooks/02_train_model.py"]
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                self.logger.error(f"Required file missing: {file_path}")
                return False
        
        # Check git repository
        try:
            subprocess.run(["git", "status"], capture_output=True, check=True)
        except subprocess.CalledProcessError:
            self.logger.error("Not in a git repository")
            return False
        
        # Check Python dependencies
        try:
            import pandas  # type: ignore
            import numpy   # type: ignore
            import sklearn # type: ignore
        except ImportError as e:
            self.logger.error(f"Missing required Python packages: {e}")
            return False
        
        self.logger.info("Environment validation passed")
        return True
    
    def execute_notebook(self, notebook_path: str, strategy: str) -> bool:
        """Execute a Jupyter notebook with error handling"""
        self.logger.info(f"Executing notebook: {notebook_path}")
        
        try:
            # Convert .py file to notebook if needed
            py_file = self.project_root / "notebooks" / f"{notebook_path.replace('.ipynb', '.py')}"
            ipynb_file = self.project_root / f"{notebook_path}"
            
            if py_file.exists() and not ipynb_file.exists():
                self.logger.info(f"Converting {py_file} to notebook format")
                subprocess.run([
                    "jupytext", "--to", "notebook", str(py_file)
                ], check=True)
            
            # Execute the notebook
            cmd = [
                "jupyter", "nbconvert", 
                "--to", "notebook",
                "--execute",
                "--inplace",
                str(ipynb_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Successfully executed {notebook_path}")
                return True
            else:
                self.logger.error(f"Failed to execute {notebook_path}: {result.stderr}")
                return False
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error executing notebook {notebook_path}: {e}")
            return False
    
    def create_commit(self, strategy: str, date: str) -> bool:
        """Create a commit with the training results"""
        try:
            # Add all changes
            subprocess.run(["git", "add", "."], check=True)
            
            # Check if there are changes to commit
            result = subprocess.run(
                ["git", "diff", "--staged", "--quiet"],
                capture_output=True
            )
            
            if result.returncode == 0:
                self.logger.info("No changes to commit")
                return True
            
            # Create commit
            commit_message = f"🤖 Automated model training: {strategy} on {date}"
            subprocess.run(["git", "commit", "-m", commit_message], check=True)
            
            self.logger.info(f"Created commit: {commit_message}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to create commit: {e}")
            return False
    
    def create_pull_request(self, strategy: str, date: str, branch: str) -> bool:
        """Create a pull request (requires GitHub CLI)"""
        try:
            # Check if gh CLI is available
            subprocess.run(["gh", "--version"], capture_output=True, check=True)
            
            # Get list of generated files
            result = subprocess.run(
                ["git", "diff", "--name-only", "main"],
                capture_output=True,
                text=True,
                check=True
            )
            generated_files = "\n".join([f"- {file}" for file in result.stdout.strip().split('\n') if file])
            
            # Create PR
            title = f"🧠 Model Training Results: {strategy} ({date})"
            body = f"""
## 🤖 Automated Model Training Results

**Strategy:** {strategy}  
**Date:** {date}  
**Branch:** {branch}

### 📊 Training Summary
- Data fetching: ✅ Complete
- Model training: ✅ Complete  
- Backtesting: ✅ Complete

### 📁 Generated Files
{generated_files}

### 🔍 Next Steps
- [ ] Review model performance metrics
- [ ] Validate backtest results
- [ ] Consider deployment if metrics meet criteria

---
*This PR was generated automatically by the Agent Workflow system.*
            """.strip()
            
            subprocess.run([
                "gh", "pr", "create",
                "--title", title,
                "--body", body,
                "--base", "main"
            ], check=True)
            
            self.logger.info(f"Created pull request: {title}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to create pull request: {e}")
            self.logger.info("You can create the PR manually using the GitHub web interface")
            return False
    
    def run_workflow(self, strategy: str, date: Optional[str] = None, config_file: Optional[str] = None) -> bool:
        """Run the complete workflow"""
        if not self.validate_environment():
            return False
        
        config = self.load_config(config_file)
        
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        
        self.logger.info(f"Starting workflow for strategy: {strategy}, date: {date}")
        
        try:
            # Step 1: Create branch
            branch_name = self.branch_manager.create_branch(strategy, date)
            self.logger.info(f"Working on branch: {branch_name}")
            
            # Step 2: Execute notebooks in order
            notebooks = config["notebooks"]
            execution_order = config["execution_order"]
            
            for step in execution_order:
                if step in notebooks:
                    notebook = notebooks[step]
                    if not self.execute_notebook(notebook, strategy):
                        self.logger.error(f"Workflow failed at step: {step}")
                        return False
            
            # Step 3: Commit changes
            if config["git"]["auto_commit"]:
                if not self.create_commit(strategy, date):
                    self.logger.error("Failed to create commit")
                    return False
            
            # Step 4: Create pull request
            if not self.create_pull_request(strategy, date, branch_name):
                self.logger.warning("Failed to create PR automatically")
            
            self.logger.info("Workflow completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Workflow failed: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current workflow status"""
        try:
            current_branch = self.branch_manager.get_current_branch()
            model_branches = self.branch_manager.list_model_branches()
            
            # Check for uncommitted changes
            result = subprocess.run(
                ["git", "diff", "--quiet"],
                capture_output=True
            )
            has_changes = result.returncode != 0
            
            return {
                "current_branch": current_branch,
                "model_branches": model_branches,
                "has_uncommitted_changes": has_changes,
                "is_model_branch": current_branch.startswith("model/"),
                "environment_valid": self.validate_environment()
            }
        except Exception as e:
            return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Agent Workflow for Model Development")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run workflow command
    run_parser = subparsers.add_parser("run", help="Run the complete workflow")
    run_parser.add_argument("strategy", help="Strategy name for the model")
    run_parser.add_argument("--date", help="Date in YYYYMMDD format (default: today)")
    run_parser.add_argument("--config", help="Path to config file")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show workflow status")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate branch naming")
    validate_parser.add_argument("branch_name", help="Branch name to validate")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    workflow = AgentWorkflow()
    
    if args.command == "run":
        success = workflow.run_workflow(args.strategy, args.date, args.config)
        sys.exit(0 if success else 1)
        
    elif args.command == "status":
        status = workflow.get_status()
        print(json.dumps(status, indent=2))
        
    elif args.command == "validate":
        manager = BranchManager()
        if re.match(manager.branch_pattern, args.branch_name):
            print(f"✅ Valid branch name: {args.branch_name}")
        else:
            print(f"❌ Invalid branch name: {args.branch_name}")
            print("Expected format: model/<strategy>/<yyyymmdd>")
            sys.exit(1)

if __name__ == "__main__":
    main()