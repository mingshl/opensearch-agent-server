# OpenSearch Agent Server

A multi-agent orchestration server for OpenSearch Dashboards with context-aware routing and Model Context Protocol (MCP) integration.

## Overview

OpenSearch Agent Server enables intelligent agent-based interactions within OpenSearch Dashboards by:

- **Multi-Agent Orchestration** — Routes requests to specialized agents based on context
- **OpenSearch Integration** — Connects to OpenSearch via MCP for real-time data access
- **AG-UI Protocol** — Implements OpenSearch Dashboard's agent UI protocol with SSE streaming
- **Flexible LLM Support** — Works with AWS Bedrock, Ollama, or other LLM providers
- **Production Ready** — Includes authentication, rate limiting, error recovery, and observability

## Architecture

```
OpenSearch Dashboards (AG-UI)
            ↓
    OpenSearch Agent Server
    ├── Router (context-based)
    ├── Agent Registry
    │   ├── ART Agent (Search Relevance Testing)
    │   │   ├── Hypothesis Agent
    │   │   ├── Evaluation Agent
    │   │   ├── User Behavior Analysis Agent
    │   │   └── Online Testing Agent
    │   └── Fallback Agent
    └── OpenSearch MCP Server
            ↓
    OpenSearch Cluster
```

## Features

- **Context-Aware Routing** — Automatically selects the appropriate agent based on request context
- **Streaming Responses** — Real-time SSE streaming for interactive user experiences
- **Tool Execution** — Agents can execute tools and visualize results in the dashboard
- **Authentication & Authorization** — JWT-based auth with configurable policies
- **Rate Limiting** — Protects backend services from overload
- **Error Recovery** — Automatic retry with exponential backoff
- **Observability** — Structured logging with request tracking

## Prerequisites

- **Python 3.10+**
- **OpenSearch 2.x** (local or remote cluster)
- **LLM Provider** (choose one):
  - AWS Bedrock (requires AWS credentials)
  - Ollama (local installation)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/mingshl/opensearch-agent-server.git
   cd opensearch-agent-server
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e .
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## Configuration

Create a `.env` file with the following settings:

```bash
# OpenSearch Connection
OPENSEARCH_URL=https://localhost:9200
OPENSEARCH_USERNAME=admin
OPENSEARCH_PASSWORD=admin

# Authentication (set to false for local development)
AG_UI_AUTH_ENABLED=false

# CORS (allow OpenSearch Dashboards origin)
AG_UI_CORS_ORIGINS=http://localhost:5601

# LLM Provider — Option 1: AWS Bedrock
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
BEDROCK_INFERENCE_PROFILE_ARN=arn:aws:bedrock:...

# LLM Provider — Option 2: Ollama (local)
OLLAMA_MODEL=llama3

# Logging
AG_UI_LOG_FORMAT=human
AG_UI_LOG_LEVEL=INFO
```

## Quick Start

### Complete Setup (3-Component Stack)

To run the full demo, first build the images and start the containers:

```
docker compose build
docker compose up
```

Start the ART AG-UI server:

```
python3 run_server.py
```

Access the chat by opening http://localhost:5601, click on the AI Assistant button.

### Verify Installation

```bash
# Check server health
./scripts/test_health.sh

# List available agents
./scripts/test_agents.sh

# Test agent interaction (requires OpenSearch running)
./scripts/test_run.sh "Show me recent logs"
```

## Development

### Install Development Dependencies

```bash
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest
```

### Code Formatting

```bash
ruff format .
ruff check .
```

## Specialized Agents (ART)

The **ART Agent** (Search Relevance Testing) is a specialized orchestrator that coordinates four sub-agents to help improve search relevance:

- **Hypothesis Agent** — Analyzes search quality issues, examines results and user behavior (UBI) data, and generates actionable hypotheses. It uses pairwise experiments to quantitatively assess how search results change with proposed improvements.
- **Evaluation Agent** — Performs offline search relevance evaluation. It calculates key metrics like NDCG, MAP, and Precision using either LLM-generated judgments or click-based judgments from UBI data.
- **User Behavior Analysis Agent** — Analyzes User Behavior Insights (UBI) data to understand actual user engagement. It identifies patterns in click-through rates (CTR), zero-click rates, and engagement rankings to pinpoint where search is failing.
- **Online Testing Agent** — Runs interleaved A/B tests using simulated user behavior. This provides online-style evaluation by comparing configurations under realistic click models (position-based, cascade, etc.) and testing for statistical significance.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgments

- Built with [strands-agents](https://github.com/anthropics/strands-agents) for multi-agent orchestration
- Implements [AG-UI Protocol](https://github.com/opensearch-project/ag-ui-protocol) for OpenSearch Dashboards
- Uses [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol) for OpenSearch integration
