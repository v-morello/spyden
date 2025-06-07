.DEFAULT_GOAL := help
PKG = spyden
LINE_LENGTH = 79
TESTS_DIR = tests/

install: ## Install the package in development mode
	pip install -e .[dev]

uninstall: ## Uninstall the package
	pip uninstall ${PKG}

# GLORIOUS hack to autogenerate Makefile help
# This simply parses the double hashtags that follow each Makefile command
# https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
help: ## Print this help message
	@echo "Makefile help for spyden"
	@echo "========================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

test: ## Run unit tests
	MPLBACKEND=Agg pytest -vv --cov=src/ --cov-report=term-missing tests/

.PHONY: install uninstall help test
