# Vector and LLM: Shodan RAG System

A Python package for importing Shodan JSON scan results into a Qdrant vector database and performing RAG (Retrieval-Augmented Generation) queries with configurable embedding and LLM backends. This tool enables semantic search and AI-powered analysis of network scan data by converting Shodan results into searchable vector embeddings.

## Installation

### From PyPI (when published)
```bash
pip install ai-shodan-astra-vector-data
```

### From Source
```bash
git clone https://github.com/BuildAndDestroy/ai-shodan-astra-vector-data.git
cd ai-shodan-astra-vector-data
pip install -e .
```

### With AWS Bedrock Support
```bash
pip install "ai-shodan-astra-vector-data[bedrock]"
```

## Quick Start

### 1. Set up Qdrant

Run Qdrant using Docker:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 2. Set up Ollama (recommended)

Install and run Ollama:
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve

# Pull required models (in another terminal)
ollama pull nomic-embed-text  # For embeddings
ollama pull llama3.2         # For LLM responses
```

### 3. Import Shodan Data

```bash
shodan-to-qdrant --embedding-backend ollama --collection shodan_data *.json
```

### 4. Query with RAG

```bash
llm-rag-shodan -p "What SSH servers are running in Russia?" --collection shodan_data --top-k 50
```

## Command Line Tools

After installation, two command-line tools are available:

### `shodan-to-qdrant` - Import Tool

Import Shodan JSON files into Qdrant:

```bash
# Basic import with Ollama embeddings
shodan-to-qdrant --embedding-backend ollama --collection my_scan *.json

# Import with AWS Bedrock embeddings
shodan-to-qdrant --embedding-backend bedrock --aws-region us-east-1 --collection my_scan *.json

# Import with zero vectors (testing)
shodan-to-qdrant --embedding-backend zero --vector-size 768 --collection test_scan scan.json
```

**Key Options:**
- `--embedding-backend`: Choose from `ollama`, `bedrock`, `zero`
- `--collection`: Qdrant collection name
- `--host/--port`: Qdrant connection details
- `--ollama-host/--ollama-port`: Ollama connection details
- `--aws-region`: AWS region for Bedrock

### `llm-rag-shodan` - Query Tool

Perform RAG queries on imported data:

```bash
# Basic query
llm-rag-shodan -p "What services are running on port 22?" --collection my_scan

# Multi-collection query
llm-rag-shodan -p "Compare SSH versions across datasets" --collections scan1 scan2 scan3

# High-precision query with more results
llm-rag-shodan -p "Find vulnerable Apache servers" --top-k 100 --collection web_scan

# Debug mode to see RAG pipeline
llm-rag-shodan -p "Show services in Moscow" --debug --top-k 20 --collection geo_scan
```

**Key Options:**
- `-p/--prompt`: Your question (required)
- `--collection`: Single collection to search
- `--collections`: Multiple collections to search
- `--top-k`: Number of results to retrieve (default: 5)
- `--llm-backend`: Choose from `ollama`, `bedrock`
- `--debug`: Show detailed pipeline information

## Python API

You can also use the package programmatically:

```python
from vector_and_llm import (
    OllamaEmbeddingBackend,
    OllamaLLMBackend, 
    QdrantClient
)
from vector_and_llm.tools.llm_rag_shodan import ShodanRAG
from pathlib import Path

# Set up backends
embedding_backend = OllamaEmbeddingBackend(host="localhost", port=11434)
llm_backend = OllamaLLMBackend(host="localhost", port=11434)

# Create Qdrant client
qdrant_client = QdrantClient(
    host="localhost",
    port=6333,
    collection_names=["my_collection"],
    embedding_backend=embedding_backend
)

# Import data
file_paths = [Path("scan1.json"), Path("scan2.json")]
qdrant_client.process_files(file_paths, "my_collection")

# Query data
rag_system = ShodanRAG(qdrant_client, llm_backend)
response = rag_system.query("What SSH servers are in the dataset?", top_k=10)
print(response)
```

## Features

- **RAG Query System**: Ask natural language questions about your network scan data
- **Multiple LLM Backends**: Support for Ollama (local) and AWS Bedrock (cloud) language models  
- **Multiple Embedding Backends**: Support for AWS Bedrock (Titan), Ollama (nomic-embed-text), and zero vectors
- **Multi-Collection Support**: Query across multiple datasets simultaneously for comparative analysis
- **Batch Processing**: Import multiple Shodan JSON files in a single run
- **Automatic Collection Management**: Creates Qdrant collections with appropriate vector dimensions
- **Enhanced Geographic Matching**: Intelligent country name variations for better location searches
- **Debug Mode**: Comprehensive debugging with pipeline visibility
- **Progress Tracking**: Real-time import progress with success/error counts

## Supported Backends

### Embedding Backends

| Backend | Model | Dimensions | Use Case |
|---------|-------|------------|----------|
| Ollama | nomic-embed-text | 768 | Local development, privacy |
| AWS Bedrock | amazon.titan-embed-text-v1 | 1536 | Production, cloud |
| Zero | Configurable | 384 (default) | Testing, development |

### LLM Backends

| Backend | Default Model | Use Case |
|---------|---------------|----------|
| Ollama | llama3.2 | Local development, privacy |
| AWS Bedrock | anthropic.claude-3-sonnet | Production, cloud |

## Example Workflows

### Security Analysis Workflow

```bash
# 1. Import recent scans
shodan-to-qdrant --embedding-backend ollama --collection recent_scan *.json

# 2. Find SSH servers
llm-rag-shodan -p "What SSH servers are exposed?" --collection recent_scan --top-k 50

# 3. Geographic analysis  
llm-rag-shodan -p "Which countries have the most exposed services?" --collection recent_scan --top-k 100

# 4. Vulnerability assessment
llm-rag-shodan -p "Find outdated services that might be vulnerable" --collection recent_scan --top-k 200
```

### Comparative Analysis Workflow

```bash
# 1. Import multiple time periods
shodan-to-qdrant --embedding-backend ollama --collection jan_2024 january_*.json
shodan-to-qdrant --embedding-backend ollama --collection feb_2024 february_*.json

# 2. Compare across time periods
llm-rag-shodan -p "How has the service landscape changed?" --collections jan_2024 feb_2024 --top-k 100

# 3. Track specific services
llm-rag-shodan -p "How have Apache versions changed over time?" --collections jan_2024 feb_2024 --top-k 150
```

## Configuration

### Environment Variables

You can set default values using environment variables:

```bash
export QDRANT_HOST=localhost
export QDRANT_PORT=6333
export OLLAMA_HOST=localhost  
export OLLAMA_PORT=11434
export AWS_DEFAULT_REGION=us-east-1
```

### AWS Bedrock Setup

For AWS Bedrock support:

1. Install boto3: `pip install boto3`
2. Configure AWS credentials: `aws configure`
3. Ensure you have access to the required Bedrock models

## Troubleshooting

### Common Issues

**Connection Errors**
```
✗ Error generating response: Connection refused
```
- Ensure Ollama/Qdrant services are running
- Check host/port configurations
- Verify firewall settings

**Model Not Found**
```
✗ Ollama embedding error: 404 Client Error
```
- Pull required models: `ollama pull nomic-embed-text`
- Check model names in configuration

**Low Quality Results**
- Increase `--top-k` for broader search
- Use more specific queries
- Enable debug mode to examine pipeline: `--debug`

### Debug Mode

Enable detailed logging:
```bash
llm-rag-shodan -p "your query" --debug --top-k 10
```

Debug output includes:
- Query vector details
- Search results from each collection  
- Similarity scores
- Formatted context sent to LLM

## Development

### Setting up Development Environment

```bash
git clone https://github.com/BuildAndDestroy/ai-shodan-astra-vector-data.git
cd ai-shodan-astra-vector-data

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code  
black vector_and_llm/

# Lint code
flake8 vector_and_llm/
```

### Package Structure

```
vector_and_llm/
├── lib/
│   ├── embedding/          # Embedding backends
│   │   ├── base.py
│   │   ├── ollama.py
│   │   ├── bedrock.py
│   │   └── zero.py
│   ├── llm/               # LLM backends  
│   │   ├── base.py
│   │   ├── ollama.py
│   │   └── bedrock.py
│   └── qdrant/            # Qdrant client
│       ├── client.py
│       └── utils.py
└── tools/                 # Command-line tools
    ├── shodan_to_qdrant.py
    └── llm_rag_shodan.py
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Commit: `git commit -am 'Add feature'`
5. Push: `git push origin feature-name`  
6. Create a Pull Request

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.

## Related Projects

- [ai-shodan-astra-scan](https://github.com/BuildAndDestroy/ai-shodan-astra-scan) - Scan Shodan to get data
- [Qdrant](https://qdrant.tech/) - Vector database
- [Ollama](https://ollama.ai/) - Local LLM and embedding server
- [Shodan](https://www.shodan.io/) - Internet device search engine
- [AWS Bedrock](https://aws.amazon.com/bedrock/) - Managed AI service
