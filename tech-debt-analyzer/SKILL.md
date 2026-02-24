---
name: tech-debt-analyzer
description: Analyze and identify technical debt including code complexity, documentation gaps, architectural violations, and legacy remains. Use when the user asks to review code quality, find refactoring targets, or assess project health.
---

# Tech Debt Analyzer

This skill provides a systematic workflow for identifying and addressing technical debt within the codebase.

## Workflow

### 1. Discovery & Analysis
- **Code Complexity:** Scan for deeply nested loops, functions exceeding 50 lines, or classes with too many responsibilities.
- **Documentation Gaps:** Check for missing docstrings in public methods and classes. Verify if type hints are used consistently.
- **Architectural Debt:** Ensure logic is properly decoupled. Core engine logic should not depend on UI components.
- **Legacy Cleanup:** Identify unused imports, commented-out code, or files that don't align with the current architecture (e.g., leftover backup files).

### 2. Prioritization
Categorize findings using the following criteria:
- **High:** Critical bugs, severe performance bottlenecks, or violations of core safety mandates.
- **Medium:** Refactoring needs that hinder maintainability, missing documentation for complex logic.
- **Low:** Minor style inconsistencies, non-critical documentation gaps.

### 3. Reporting
Use the provided template `assets/tech_debt_report_template.md` to present findings to the user.

## Reference Materials
- **Coding Standards:** See [references/coding_standards.md](references/coding_standards.md) for the project's architectural principles and coding rules.

## Example Triggers
- "Check for tech debt in the `core/engine` directory."
- "Evaluate the code quality of my recent changes."
- "Find areas in the project that need refactoring."
