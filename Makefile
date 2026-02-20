.PHONY: configure build install run clean all

PREFIX ?= ~/.local
all: configure build

configure: requirements.txt
	@uv venv --python 3.12
	@uv pip install -r requirements.txt > /dev/null
	@echo "Configured!"

generate-completions:
	@mkdir -p dist
	@_IMGCONV_COMPLETE=bash_source ./imgconv > dist/imgconv.bash
	@_IMGCONV_COMPLETE=zsh_source ./imgconv > dist/_imgconv.zsh
	@echo "Generated shell completions"

build: main.py
	@uv run pyinstaller --onefile main.py --log-level=FATAL
	@cp dist/main imgconv
	@$(MAKE) generate-completions
	@echo "Built successfully!"

install: build
	@install -Dm755 imgconv $(PREFIX)/bin/imgconv
	@install -Dm644 dist/imgconv.bash $(PREFIX)/share/bash-completion/completions/imgconv
	@install -Dm644 dist/_imgconv.zsh ~/.zsh/completions/_imgconv
	@echo "Installed!"

run: build
	@./imgconv

clean:
	@rm -rf imgconv dist build main.spec
	@rm -rf .venv
	@rm -rf __pycache__
	@rm -rf *.pyc
	@echo "Cleaned!"
