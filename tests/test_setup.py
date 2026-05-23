"""
Basic setup tests to verify the project environment is correctly configured.
"""
import sys


def test_python_version():
    """Verify we are running Python 3.11."""
    assert sys.version_info.major == 3
    assert sys.version_info.minor == 11


def test_project_structure():
    """Verify all required project folders exist."""
    import os

    required_folders = [
        "src",
        "src/vision",
        "src/api",
        "src/dashboard",
        "tests",
        "data",
        "models",
        "notebooks",
        "docker",
        ".github/workflows",
    ]

    for folder in required_folders:
        assert os.path.exists(folder), f"Missing folder: {folder}"


def test_environment_is_working():
    """Verify basic Python environment works correctly."""
    result = 2 + 2
    assert result == 4
