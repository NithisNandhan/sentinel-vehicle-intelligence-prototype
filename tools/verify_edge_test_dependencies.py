"""Verify edge[test] from a fresh venv, not the preinstalled AI environment."""
import os
from pathlib import Path
import subprocess
import sys
import tomllib

root=Path('/workspace/edge')
environment=Path('/tmp/sentinel-edge-test-environment')
subprocess.run([sys.executable,'-m','venv',str(environment)],check=True)
python=str(environment/'bin/python')
requirements=tomllib.loads((root/'pyproject.toml').read_text())['project']['optional-dependencies']['test']
subprocess.run([python,'-m','pip','install','--disable-pip-version-check','--no-cache-dir',*requirements],check=True)
env=dict(os.environ,PYTHONPATH=str(root),PYTHONDONTWRITEBYTECODE='1')
subprocess.run([python,'-m','pytest','-q','-p','no:cacheprovider','tests'],cwd=root,env=env,check=True)
