# Coding Standards & Architecture Principles

This reference guides the technical debt analysis by defining the "ideal state" of the codebase.

## Core Principles
1. **Decoupling:** Business logic (Core) must be strictly separated from the UI (Streamlit). Core engines should be runnable without a UI.
2. **Validation First:** All strategies and models must have accompanying backtesting or validation logic.
3. **Efficiency:** LLM calls should be optimized via preprocessing to minimize cost and latency.

## Coding Rules
- **Error Handling:** Thorough exception handling and logging for all crawling and API calls.
- **Type Hinting:** Active use of Python Type Hinting for readability and stability.
- **Documentation:** Every new utility or agent must include function-level docstrings.
- **Cost Control:** Limit `max_tokens` in LLM calls and maintain concise prompts.

## Directory Structure Norms
- `core/data/`: Data providers and database management.
- `core/engine/`: Core logic, models, and agents.
- `core/utils/`: Common utilities (notifiers, backtesters).
- `app/`: Streamlit UI pages.
