# Source publication status

This repository is being populated in batches. The complete agent workflow is not yet published.

The first batch contains tool definitions extracted from the project HTML export, with explanatory comments condensed and executable logic retained. The fragment depends on setup globals from the original notebook and is not a standalone module. It has not been rerun after extraction.

The project uses synthetic toy customer, shipment, and locker data, as confirmed by the author. No real customer data is intended for this demonstration. The original data files and playbook PDF are not included in this commit.

Credentials are excluded. Copy config.example.json to config.json locally and supply your own credentials. Do not commit config.json.

Remaining source to publish: setup and retrieval, preprocessing and routing, agent schemas and prompts, graph construction, evaluation, and reporting.
