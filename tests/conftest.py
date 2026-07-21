import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pytest_addoption(parser):
    parser.addoption("--model", action="store", default="eva18b",
                     help="model registry key for GPU integration tests")
