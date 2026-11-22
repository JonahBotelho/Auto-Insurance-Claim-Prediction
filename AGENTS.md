# Project guidance

- Read `README.md` before starting work to understand the app, setup, and prototype limitations.
- Update `README.md` when changes affect setup, behavior, usage, or documented limitations.
- Keep `README.md` minimal and concise.
- Do not perform Git actions like Commit, Push, Pull etc. Assume human actions
- Add or modify tests only when explicitly requested by the user; place unit tests in `src/tests`.
- Data insight and evaluation scripts go in `analysis/`; main application and prediction logic go in `src/`
- Use `src/feature_schema.py` for shared constants; import them from there rather than duplicating definitions.
- Make small, focused changes, avoiding rewriting too much existing code when possible.

# Code style guideline

- Add descriptive comments for complex code. Explain non-obvious reasoning, assumptions, and constraints rather than restating the code.
- Keep classes short and single focused
- Keep files short, ideally under 300 lines, with clear a clear purpose and description at the top
- Keep methods short and single focus, ideally under 30 lines but exceptions are allowed
- Use descriptive variable and file names like `FloorPlan` rather than abbreviations like `FlrPln`

# Analysis guidelines

- Always run configurations on seeds 41, 42, 43, 44, 45
