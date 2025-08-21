# Shodan to Qdrant RAG System

A Python tool for importing Shodan JSON scan results into a Qdrant vector database and performing RAG (Retrieval-Augmented Generation) queries with configurable embedding and LLM backends. This tool enables semantic search and AI-powered analysis of network scan data by converting Shodan results into searchable vector embeddings.

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

## Prerequisites

### Required Services

1. **Qdrant Vector Database**
   - Running instance accessible via HTTP
   - Default: `localhost:6333`

2. **Embedding Service** (choose one):
   - **Ollama** (recommended for local development)
   - **AWS Bedrock** (for production/cloud deployments)
   - **Zero vectors** (for testing without embeddings)

3. **LLM Service** (for RAG queries):
   - **Ollama** with language models (llama3.2, etc.)
   - **AWS Bedrock** with Claude or other models

### Python Dependencies

```bash
pip install requests
```

For AWS Bedrock support:
```bash
pip install boto3
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/BuildAndDestroy/ai-shodan-astra-vector-data.git
cd ai-shodan-astra-vector-data
```

2. Install dependencies:
```bash
pip install -r requirements.txt
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
python shodan_to_qdrant.py --host 127.0.0.1 --port 6333 --ollama-host 127.0.0.1 --ollama-port 11434 --collection shodan_querytwo_electricboogaloo --embedding-backend ollama --ollama-model nomic-embed-text query_*.json
```

### 4. Query with RAG

```bash
python shodan_to_qdrant.py --host 127.0.0.1 --port 6333 --ollama-host 127.0.0.1 --ollama-port 11434 --collection shodan_querytwo_electricboogaloo --embedding-backend ollama --ollama-model nomic-embed-text -p "What SSH servers are running in Russia?" --top-k 2000
```

## Usage

### Basic Syntax

**Import Mode:**
```bash
python3 shodan_rag.py [import_options] file1.json file2.json ...
```

**Query Mode:**
```bash
python3 shodan_rag.py -p "your question" [query_options]
```

### Import Examples

#### Basic Import with Ollama Embeddings
```bash
python3 shodan_rag.py --embedding-backend ollama -f scan_results.json
```

#### Import Multiple Files
```bash
python3 shodan_rag.py --embedding-backend ollama -f file1.json -f file2.json --file file3.json
```

#### AWS Bedrock Embeddings
```bash
python3 shodan_rag.py --embedding-backend bedrock --aws-region us-east-1 -f *.json
```

#### Custom Collection Name
```bash
python3 shodan_rag.py --embedding-backend ollama --collection my_scan_data -f scan.json
```

### RAG Query Examples

#### Basic Security Analysis
```bash
# Find SSH servers
python3 shodan_rag.py -p "What SSH servers are in this dataset?"

# Geographic analysis
python3 shodan_rag.py -p "What services are running in Russian Federation?" --top-k 50

# Vulnerability hunting
python3 shodan_rag.py -p "Find outdated Apache servers that might be vulnerable" --top-k 100

# Service inventory
python3 shodan_rag.py -p "List all unique services and their versions" --top-k 2000
```

#### Multi-Collection Queries
```bash
# Compare across datasets
python3 shodan_rag.py -p "What are the differences between these scans?" --collections scan_january scan_february scan_march

# Cross-dataset geographic analysis
python3 shodan_rag.py -p "Which countries appear in all datasets?" --collections collection1 collection2 --top-k 100

# Trend analysis
python3 shodan_rag.py -p "How has the service landscape changed over time?" --collections old_scan new_scan --top-k 200
```

#### Advanced Queries with Different Backends
```bash
# Use AWS Bedrock LLM for analysis
python3 shodan_rag.py -p "Provide a comprehensive security assessment" --llm-backend bedrock --top-k 100

# Debug mode to see RAG pipeline
python3 shodan_rag.py -p "Show me all services in Moscow" --debug --top-k 50
```

## Command Line Options

### Core Options
- `files`: Shodan JSON files to import (positional arguments)
- `-f, --file`: Shodan JSON file to import (can be used multiple times)
- `-p, --prompt`: Query prompt for RAG (instead of importing files)
- `--host`: Qdrant host (default: `10.0.20.62`)
- `--port`: Qdrant port (default: `6333`)

### Collection Options
- `--collection`: Single Qdrant collection name (default: `shodan_query`)
- `--collections`: Multiple Qdrant collection names to search (overrides `--collection`)

### Embedding Backend Options
- `--embedding-backend`: Choose from `zero`, `ollama`, `bedrock` (default: `ollama`)
- `--vector-size`: Vector size for zero backend only (default: `384`)

### LLM Backend Options
- `--llm-backend`: Choose from `ollama`, `bedrock` (default: `ollama`)
- `--top-k`: Number of similar results to retrieve for RAG (default: `5`)
- `--debug`: Enable debug output showing RAG pipeline details

### Ollama Options
- `--ollama-host`: Ollama host (default: `localhost`)
- `--ollama-port`: Ollama port (default: `11434`)
- `--ollama-embed-model`: Ollama embedding model (default: `nomic-embed-text`)
- `--ollama-llm-model`: Ollama LLM model (default: `llama3.2`)

### AWS Bedrock Options
- `--aws-region`: AWS region (default: `us-east-1`)
- `--aws-embed-model`: Bedrock embedding model ID (default: `amazon.titan-embed-text-v1`)
- `--aws-llm-model`: Bedrock LLM model ID (default: `anthropic.claude-3-sonnet-20240229-v1:0`)

## Shodan JSON Format

The tool expects Shodan JSON files with the following structure:

```json
{
  "matches": [
    {
      "ip_str": "192.168.1.1",
      "port": 22,
      "product": "OpenSSH",
      "version": "7.4",
      "timestamp": "2024-01-01T12:00:00.000000",
      "location": {
        "country_name": "Russian Federation",
        "country_code": "RU",
        "city": "Moscow"
      },
      "ssh": {
        "type": "rsa"
      },
      "data": "SSH-2.0-OpenSSH_7.4..."
    }
  ]
}
```

## RAG System Architecture

### How It Works

1. **Import Phase:**
   - Shodan JSON files are processed
   - Key information is extracted and combined into embedding text
   - Text is converted to vectors using the chosen embedding backend
   - Vectors and metadata are stored in Qdrant collections

2. **Query Phase:**
   - User question is converted to a query vector
   - Similar vectors are retrieved from one or more collections
   - Retrieved data is formatted as context
   - Context + question is sent to the LLM for analysis

### Embedding Text Generation

The tool creates semantic embeddings by combining:
- IP address and port information
- Service name and version details
- Geographic location with country variations
- SSH key information (when available)
- Banner data (truncated to 500 characters)
- Enhanced country name matching for better geographic searches

Example embedding text:
```
IP: 192.168.1.1 | Port: 22 | Service: OpenSSH | Version: 7.4 | Country: Russian Federation | Location: Russia | Location: Russian Federation | Location: RU | City: Moscow | SSH Key Type: rsa | Banner: SSH-2.0-OpenSSH_7.4...
```

### Multi-Collection Support

When querying multiple collections:
- Each collection is searched in parallel
- Results are combined and ranked by similarity score
- Top-k results are selected globally across all collections
- Results include dataset attribution for context

## Vector Dimensions by Backend

| Backend | Model | Dimensions |
|---------|-------|------------|
| Ollama | nomic-embed-text | 768 |
| AWS Bedrock | amazon.titan-embed-text-v1 | 1536 |
| Zero | Configurable | 384 (default) |

## Example Workflows

### Complete Workflow Example

```bash
# 1. Start services
docker run -d -p 6333:6333 qdrant/qdrant
ollama serve &
ollama pull nomic-embed-text
ollama pull llama3.2

# 2. Import multiple scan datasets
python3 shodan_rag.py --embedding-backend ollama --collection january_scan -f january_*.json
python3 shodan_rag.py --embedding-backend ollama --collection february_scan -f february_*.json

# 3. Query single dataset
python3 shodan_rag.py -p "What are the most common services?" --collection january_scan --top-k 100

# 4. Compare across datasets
python3 shodan_rag.py -p "How did the service landscape change from January to February?" --collections january_scan february_scan --top-k 200

# 5. Geographic analysis
python3 shodan_rag.py -p "What countries have the most SSH servers?" --collections january_scan february_scan --top-k 500
```

## Troubleshooting

### Common Issues

**LLM Connection Issues**
```
✗ Error generating response: Connection refused
```
- Ensure Ollama is running: `ollama serve`
- Verify the LLM model is available: `ollama list`
- Check host/port configuration

**Embedding Generation Issues**
```
✗ Ollama embedding error: 404 Client Error
```
- Ensure embedding model is pulled: `ollama pull nomic-embed-text`
- Verify Ollama is accessible at specified host:port
- Check that embedding and LLM hosts are configured correctly

**Low Quality Results**
```
Results have low similarity scores or irrelevant content
```
- Try increasing `--top-k` for broader search results
- Use more specific queries
- Consider re-importing with enhanced embeddings
- Use debug mode to examine the RAG pipeline: `--debug`

**Multi-Collection Issues**
```
⚠️ Collection 'collection_name' returned status 404
```
- Verify collection names exist: check Qdrant dashboard
- Ensure collections were created with compatible vector dimensions
- Check collection spelling and case sensitivity

### Debug Mode

Enable comprehensive debugging:
```bash
python3 shodan_rag.py -p "your query" --debug --top-k 10
```

Debug output includes:
- Query vector details
- Search results from each collection
- Similarity scores
- Formatted context sent to LLM
- Collection attribution

### Performance Tips

- **Optimal top-k values:**
  - Small datasets: 10-50
  - Medium datasets: 50-200
  - Large datasets or broad queries: 200-2000+
- **Geographic queries:** Use higher top-k values (100+) for comprehensive location analysis
- **Comparative analysis:** Use multiple collections with moderate top-k (50-200)
- **Service inventory:** Use very high top-k (1000+) to capture all unique services

## AWS Bedrock Setup

### Prerequisites
```bash
pip install boto3
aws configure  # Set up credentials
```

### Usage with Bedrock
```bash
# Import with Bedrock embeddings
python3 shodan_rag.py --embedding-backend bedrock --aws-region us-east-1 -f scan.json

# Query with Bedrock LLM
python3 shodan_rag.py -p "Analyze security posture" --llm-backend bedrock --aws-region us-east-1
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Commit: `git commit -am 'Add feature'`
5. Push: `git push origin feature-name`
6. Create a Pull Request

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE - see the LICENSE file for details.

## Support

- **Issues**: Report bugs and request features via GitHub Issues
- **Documentation**: Check the code comments for implementation details
- **Community**: Discussions welcome in GitHub Discussions


## Related Projects

- [ai-shodan-astra-scan](https://github.com/BuildAndDestroy/ai-shodan-astra-scan) - Scan shodan to get data
- [Qdrant](https://qdrant.tech/) - Vector database
- [Ollama](https://ollama.ai/) - Local LLM and embedding server
- [Shodan](https://www.shodan.io/) - Internet device search engine
- [AWS Bedrock](https://aws.amazon.com/bedrock/) - Managed AI service


## Next Steps

* Build this out into a full package for os install
* Update the Dockerfile to mount directory for data input