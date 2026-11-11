# Source publication status

Publication is in progress; the full agent application is not yet available.

Published source parts:
- src/notebook_parts/setup.py: imports, local configuration, model initialization, data discovery, data previews, PDF ingestion and RAG.
- src/notebook_parts/tools.py: delivery, customer, locker, playbook and escalation tools.
- src/notebook_parts/preprocessing.py: shared state, injection-pattern screening, deduplication, consolidation, context retrieval and noise classification.
- requirements.txt and config.example.json.

These are notebook fragments with shared globals, not independent importable modules. Use Python 3.12+ and execute setup, tools, then preprocessing in the same notebook namespace. Dependency versions follow the original export; unpinned packages are not a reproducible lockfile. The extracted source has not been rerun. Setup performs paid model smoke tests and prints dataset previews.

Remaining: preprocessing/orchestration/finalization nodes, resolution and communication agents, critic prompts and nodes, workflow graph, evaluation and reporting. A complete runnable entry point is not yet published.

The author confirms the project uses synthetic toy customer, shipment and locker data. Original data files and the playbook PDF are not included. Supply authorized demonstration inputs locally. Names and addresses can reach model, evaluation and tracing payloads; do not substitute real data without reviewing those paths.

Credentials must stay local. Copy config.example.json to config.json and fill in credentials; never commit config.json.
