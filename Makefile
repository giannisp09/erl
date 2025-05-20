.PHONY: setup test run clean docker-build docker-run

# Python version
PYTHON := python3
VENV := venv

# Default target
all: setup

# Setup virtual environment and install dependencies
setup:
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install -r requirements.txt

# Run tests
test:
	. $(VENV)/bin/activate && python -m pytest tests/

# Run the stream demo
run:
	. $(VENV)/bin/activate && python demos/stream_demo.py

# Clean up generated files
clean:
	rm -rf $(VENV)
	rm -rf outputs/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Build Docker image
# docker-build:
# 	docker build -t experience-stream-demo .

# # Run Docker container
# docker-run:
# 	docker run -it --rm experience-stream-demo

# Help command
help:
	@echo "Available commands:"
	@echo "  make setup        - Create virtual environment and install dependencies"
	@echo "  make test         - Run tests"
	@echo "  make run          - Run the stream demo"
	@echo "  make clean        - Clean up generated files"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container" 