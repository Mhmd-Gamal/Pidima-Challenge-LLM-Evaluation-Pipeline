#!/usr/bin/env python3
"""
Setup verification script for Pidima challenge.
Checks if all components are properly configured.
"""
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'


def print_header(text: str):
    """Print section header."""
    print(f"\n{Colors.BLUE}{'='*60}")
    print(f"{text}")
    print(f"{'='*60}{Colors.RESET}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")


def check_python_version() -> bool:
    """Check if Python version is 3.11+."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print_success(f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python {version.major}.{version.minor} (3.11+ required)")
        return False


def check_file_exists(filepath: str) -> bool:
    """Check if a file exists."""
    path = Path(filepath)
    if path.exists():
        print_success(f"{filepath}")
        return True
    else:
        print_error(f"{filepath} (missing)")
        return False


def check_directory_structure() -> bool:
    """Check if all required directories exist."""
    required_dirs = [
        "src",
        "src/api",
        "src/llm",
        "src/evaluation",
        "src/utils",
        "tests",
        "docs"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            print_success(f"{dir_path}/")
        else:
            print_error(f"{dir_path}/ (missing)")
            all_exist = False
    
    return all_exist


def check_required_files() -> bool:
    """Check if all required files exist."""
    required_files = [
        "README.md",
        "Dockerfile",
        "docker-compose.yml",
        "requirements.txt",
        ".env.example",
        "src/__init__.py",
        "src/api/__init__.py",
        "src/api/main.py",
        "src/api/models.py",
        "src/llm/__init__.py",
        "src/llm/loader.py",
        "src/llm/inference.py",
        "src/evaluation/__init__.py",
        "src/evaluation/dataset.py",
        "src/evaluation/run_evaluation.py",
        "src/evaluation/metrics.py",
        "src/utils/__init__.py",
        "src/utils/config.py",
        "src/utils/logging.py",
        "tests/__init__.py",
        "tests/test_api.py"
    ]
    
    all_exist = True
    for filepath in required_files:
        if not check_file_exists(filepath):
            all_exist = False
    
    return all_exist


def check_docker() -> bool:
    """Check if Docker is installed and running."""
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print_success(f"Docker {result.stdout.strip()}")
            return True
        else:
            print_error("Docker not found")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print_error("Docker not found or not responding")
        return False


def check_docker_compose() -> bool:
    """Check if Docker Compose is installed."""
    try:
        result = subprocess.run(
            ["docker-compose", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print_success(f"Docker Compose {result.stdout.strip()}")
            return True
        else:
            print_error("Docker Compose not found")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print_error("Docker Compose not found or not responding")
        return False


def check_python_packages() -> Tuple[bool, List[str]]:
    """Check if required Python packages can be imported."""
    required_packages = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("transformers", "Transformers"),
        ("torch", "PyTorch"),
        ("datasets", "Datasets"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("aiohttp", "aiohttp"),
        ("tqdm", "tqdm")
    ]
    
    missing = []
    all_installed = True
    
    for package_name, display_name in required_packages:
        try:
            __import__(package_name)
            print_success(f"{display_name}")
        except ImportError:
            print_error(f"{display_name} (not installed)")
            missing.append(package_name)
            all_installed = False
    
    return all_installed, missing


def check_env_file() -> bool:
    """Check if .env file exists or needs to be created."""
    env_path = Path(".env")
    example_path = Path(".env.example")
    
    if env_path.exists():
        print_success(".env file exists")
        return True
    elif example_path.exists():
        print_warning(".env file missing (copy from .env.example)")
        print("  Run: cp .env.example .env")
        return False
    else:
        print_error(".env.example file missing")
        return False


def main():
    """Run all verification checks."""
    print(f"\n{Colors.BLUE}╔════════════════════════════════════════════════════════════╗")
    print(f"║   Pidima AI Engineer Challenge - Setup Verification      ║")
    print(f"╚════════════════════════════════════════════════════════════╝{Colors.RESET}\n")
    
    checks = []
    
    # Check Python version
    print_header("1. Python Version")
    checks.append(("Python Version", check_python_version()))
    
    # Check directory structure
    print_header("2. Directory Structure")
    checks.append(("Directory Structure", check_directory_structure()))
    
    # Check required files
    print_header("3. Required Files")
    checks.append(("Required Files", check_required_files()))
    
    # Check environment file
    print_header("4. Environment Configuration")
    checks.append(("Environment File", check_env_file()))
    
    # Check Docker
    print_header("5. Docker Installation")
    docker_ok = check_docker()
    compose_ok = check_docker_compose()
    checks.append(("Docker", docker_ok and compose_ok))
    
    # Check Python packages
    print_header("6. Python Dependencies")
    packages_ok, missing = check_python_packages()
    checks.append(("Python Packages", packages_ok))
    
    # Summary
    print_header("Summary")
    
    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    
    if passed == total:
        print_success(f"All checks passed! ({passed}/{total})")
        print(f"\n{Colors.GREEN}✓ Setup is complete. You're ready to go!{Colors.RESET}")
        print("\nNext steps:")
        print("  1. Start the API: docker-compose up --build")
        print("  2. Run evaluation: python src/evaluation/run_evaluation.py")
        return 0
    else:
        print_error(f"Some checks failed ({passed}/{total} passed)")
        print(f"\n{Colors.YELLOW}Please fix the issues above before proceeding.{Colors.RESET}")
        
        if not packages_ok:
            print("\nTo install missing packages:")
            print("  pip install -r requirements.txt")
        
        if not checks[3][1]:  # Environment file
            print("\nTo create .env file:")
            print("  cp .env.example .env")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())