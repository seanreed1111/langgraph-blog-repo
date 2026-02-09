# Changelog

## [Unreleased]

### Added
- Add stage-2 two-bot interview graph with Langfuse prompt management and cyclic LangGraph orchestration (#7)
- Add `make chat SCOPE=2` and `make dev SCOPE=2` targets for running stage-2 (#7)
- Add personas module with configurable interviewer/interviewee data (#7)
- Add blog post and tutorial for stage-2 implementation
- Add matplotlib and pillow as dev dependencies
- Add graph diagram PNG for stage-2 tutorial documentation
- Add Claude skills for git analysis, refactoring, and Mermaid diagrams

### Changed
- Reorganize documentation into `docs/thoughts/` and `docs/plan/` directory structure (#8)
- Update Makefile SCOPE variable to support multi-stage workspace (#7)

### Fixed
- Fix workspace sync by using `langgraph-cli[inmem]` and correcting `uv sync` command

---

## [0.2.0] - 2026-02-07

### Added
- Add ty type checker as dev dependency with `make typecheck` target (#6)
- Add secret scanning with gitleaks pre-commit hook (#6)

### Fixed
- Fix `create_graph()` return type annotation (`StateGraph` → `CompiledStateGraph`) (#6)
- Fix Langfuse tracing not appearing in dashboard by migrating to SDK v3 singleton pattern (#5)

## [0.1.0] - 2026-02-05

### Added
- Add v0-stage-1 simple chatbot with LangGraph, Mistral AI, and Langfuse observability (#5)
- Add streaming CLI chat interface with session/user ID tracing (#5)
- Add LangGraph Platform deployment config (`langgraph.json`) (#5)
- Add Makefile with `chat`, `dev`, and `test-smoke` targets (#5)
- Add pydantic-settings config module for environment variable management (#5)
- Add custom equality and hashability to `Modifier`, `Location`, `Item`, and `Menu` models (#3)
- Add quantity-based comparison operators (`<`, `<=`, `>`, `>=`) to `Item` for same-configuration items (#3)
- Add `__add__` method to `Item` to combine quantities of same-configuration items (#3)
- Add `Location` model for restaurant location data (id, name, address, city, state, zip, country) (#2)
- Add `Menu.from_json_file()` class method to load menu from JSON file path (#2)
- Add `Menu.from_dict()` class method to load menu from dictionary (#2)
- Add `REGULAR` size option to `Size` enum (#2)

### Changed
- Configure `pyproject.toml` for src-layout with setuptools build system (#5)
- Update `Menu` model with flattened metadata fields (menu_id, menu_name, menu_version, location) (#2)
- Reorganize menu data structure to `menus/mcdonalds/breakfast-menu/` (#2)
