# Contributing

Contributions are welcome when they preserve the audit contract.

## Before opening a pull request

1. Add or update unit tests.
2. Run:

   ```bash
   PYTHONPATH=src python -m unittest discover -s tests -v
   ```

3. Document the metric's formula, units, intended interpretation and what it
   does not measure.
4. Do not introduce functional or cognitive names without an operational test.
5. Keep the repository limited to its public LLM scope.

Missing metadata must be recorded as `to-confirm`; never infer it.
