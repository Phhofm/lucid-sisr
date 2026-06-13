"""Test configuration for pytest."""
import sys
from pathlib import Path

# Add the workspace root to path for test imports
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
