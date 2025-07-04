#!/usr/bin/env python3
"""
Branch Manager for Model Development Workflow

This script manages branch creation and naming following the convention:
model/<strategy>/<yyyymmdd>

Usage:
    python scripts/branch_manager.py create <strategy> [date]
    python scripts/branch_manager.py list
    python scripts/branch_manager.py switch <branch_name>
"""

import subprocess
import sys
import re
from datetime import datetime
from typing import Optional, List

class BranchManager:
    def __init__(self):
        self.branch_pattern = r"^model/[\w\-_]+/\d{8}$"
        
    def get_current_date(self) -> str:
        """Get current date in YYYYMMDD format"""
        return datetime.now().strftime("%Y%m%d")
    
    def validate_strategy_name(self, strategy: str) -> bool:
        """Validate strategy name (alphanumeric, hyphens, underscores only)"""
        return bool(re.match(r"^[\w\-_]+$", strategy))
    
    def create_branch(self, strategy: str, date: Optional[str] = None) -> str:
        """Create a new branch following the naming convention"""
        if not self.validate_strategy_name(strategy):
            raise ValueError("Strategy name must contain only alphanumeric characters, hyphens, and underscores")
        
        if date is None:
            date = self.get_current_date()
        
        # Validate date format
        try:
            datetime.strptime(date, "%Y%m%d")
        except ValueError:
            raise ValueError("Date must be in YYYYMMDD format")
        
        branch_name = f"model/{strategy}/{date}"
        
        try:
            # Check if branch already exists
            result = subprocess.run(
                ["git", "rev-parse", "--verify", branch_name],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(f"Branch '{branch_name}' already exists")
                return branch_name
            
            # Create and switch to new branch
            subprocess.run(["git", "checkout", "-b", branch_name], check=True)
            print(f"Created and switched to branch: {branch_name}")
            return branch_name
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to create branch: {e}")
    
    def list_model_branches(self) -> List[str]:
        """List all branches following the model/* pattern"""
        try:
            result = subprocess.run(
                ["git", "branch", "-a"],
                capture_output=True,
                text=True,
                check=True
            )
            
            branches = []
            for line in result.stdout.split('\n'):
                line = line.strip().lstrip('* ').replace('remotes/origin/', '')
                if line.startswith('model/'):
                    branches.append(line)
            
            return sorted(list(set(branches)))
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to list branches: {e}")
    
    def switch_branch(self, branch_name: str):
        """Switch to specified branch"""
        try:
            subprocess.run(["git", "checkout", branch_name], check=True)
            print(f"Switched to branch: {branch_name}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to switch to branch '{branch_name}': {e}")
    
    def get_current_branch(self) -> str:
        """Get current branch name"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to get current branch: {e}")

def main():
    manager = BranchManager()
    
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    command = sys.argv[1]
    
    try:
        if command == "create":
            if len(sys.argv) < 3:
                print("Usage: python scripts/branch_manager.py create <strategy> [date]")
                sys.exit(1)
            
            strategy = sys.argv[2]
            date = sys.argv[3] if len(sys.argv) > 3 else None
            branch_name = manager.create_branch(strategy, date)
            print(f"Branch created: {branch_name}")
            
        elif command == "list":
            branches = manager.list_model_branches()
            current = manager.get_current_branch()
            
            print("Model development branches:")
            for branch in branches:
                marker = "* " if branch == current else "  "
                print(f"{marker}{branch}")
                
        elif command == "switch":
            if len(sys.argv) < 3:
                print("Usage: python scripts/branch_manager.py switch <branch_name>")
                sys.exit(1)
            
            branch_name = sys.argv[2]
            manager.switch_branch(branch_name)
            
        else:
            print(f"Unknown command: {command}")
            print(__doc__)
            sys.exit(1)
            
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()