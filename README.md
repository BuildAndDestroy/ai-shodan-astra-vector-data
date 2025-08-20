# ai-shodan-astra-vector-data

A Python tool for importing Shodan JSON scan results into a Qdrant vector database with configurable embedding backends. This tool enables semantic search and analysis of network scan data by converting Shodan results into searchable vector embeddings.

## Features

- **Multiple Embedding Backends**: Support for AWS Bedrock (Titan), Ollama (nomic-embed-text), and zero vectors
- **Batch Processing**: Import multiple Shodan JSON files in a single run
- **Automatic Collection Management**: Creates Qdrant collections with appropriate vector dimensions
- **Semantic Text Generation**: Intelligently combines IP, port, service, location, and banner data for meaningful embeddings
- **Error Handling**: Graceful fallbacks and detailed logging
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

* https://qdrant.tech/documentation/guides/installation/

```bash
docker run --rm -it -p 6333:6333 -v /mnt/data:/qdrant/storage qdrant/qdrant
```

### 2. Set up Ollama (recommended)

Install and run Ollama:
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve

# Pull the embedding model (in another terminal)
ollama pull nomic-embed-text
```

### 3. Import Shodan Data

```bash
python3 shodan_to_qdrant.py --embedding-backend ollama your_shodan_file.json
```

## Usage

### Basic Syntax

```bash
python3 shodan_to_qdrant.py [options] file1.json file2.json ...
```

### Examples

#### Zero Vectors (No Embeddings)
```bash
python3 shodan_to_qdrant.py scan_results.json
```

#### Ollama Embeddings (Local)
```bash
python3 shodan_to_qdrant.py \
  --embedding-backend ollama \
  --ollama-host localhost \
  --ollama-port 11434 \
  *.json
```

#### AWS Bedrock Embeddings
```bash
python3 shodan_to_qdrant.py \
  --embedding-backend bedrock \
  --aws-region us-east-1 \
  --aws-model amazon.titan-embed-text-v1 \
  scan_results.json
```

#### Custom Qdrant Instance
```bash
python3 shodan_to_qdrant.py \
  --host 10.0.20.62 \
  --port 6333 \
  --collection my_scan_collection \
  --embedding-backend ollama \
  scan_results.json
```

#### Batch Import Multiple Files
```bash
python3 shodan_to_qdrant.py \
  --embedding-backend ollama \
  query_*.json scan_*.json results_*.json
```

## Command Line Options

### Core Options
- `files`: One or more Shodan JSON files to import
- `--host`: Qdrant host (default: `10.0.20.62`)
- `--port`: Qdrant port (default: `6333`)
- `--collection`: Qdrant collection name (default: `shodan_query`)

### Embedding Backend Options
- `--embedding-backend`: Choose from `zero`, `ollama`, `bedrock` (default: `zero`)
- `--vector-size`: Vector size for zero backend only (default: `384`)

### Ollama Options
- `--ollama-host`: Ollama host (default: `localhost`)
- `--ollama-port`: Ollama port (default: `11434`)
- `--ollama-model`: Ollama model (default: `nomic-embed-text`)

### AWS Bedrock Options
- `--aws-region`: AWS region (default: `us-east-1`)
- `--aws-model`: Bedrock model ID (default: `amazon.titan-embed-text-v1`)

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
        "country_name": "United States",
        "city": "New York"
      },
      "ssh": {
        "type": "rsa"
      },
      "data": "SSH-2.0-OpenSSH_7.4..."
    }
  ]
}
```

## Embedding Text Generation

The tool creates semantic embeddings by combining:
- IP address and port
- Service name and version
- Geographic location (country, city)
- SSH key information (if available)
- Banner data (first 500 characters)

Example embedding text:
```
IP: 192.168.1.1 | Port: 22 | Service: OpenSSH | Version: 7.4 | Country: United States | City: New York | SSH Key Type: rsa | Banner: SSH-2.0-OpenSSH_7.4...
```

## Vector Dimensions

| Backend | Model | Dimensions |
|---------|-------|------------|
| Ollama | nomic-embed-text | 768 |
| AWS Bedrock | amazon.titan-embed-text-v1 | 1536 |
| Zero | Configurable | 384 (default) |

## Troubleshooting

### Common Issues

**404 Error on Ollama API**
```
✗ Ollama embedding error: 404 Client Error: Not Found
```
- Ensure Ollama is running: `ollama serve`
- Verify the model is pulled: `ollama pull nomic-embed-text`
- Check host/port configuration

**Collection Creation Failed**
```
✗ Failed to create collection 'collection_name'
```
- Verify Qdrant is running and accessible
- Check network connectivity to Qdrant host
- Ensure proper permissions

**AWS Bedrock Authentication**
```
✗ Bedrock embedding error: UnauthorizedOperation
```
- Configure AWS credentials: `aws configure`
- Ensure IAM permissions for Bedrock access
- Verify model availability in your region

### Debugging

Enable verbose output by checking the console logs. The tool provides detailed status for:
- Collection existence and creation
- Individual record processing
- Embedding generation
- Upload success/failure

## Performance Considerations

- **Rate Limiting**: Built-in delays for embedding API calls (0.1s between requests)
- **Batch Size**: Processes one record at a time for reliability
- **Memory Usage**: Loads entire JSON files into memory
- **Network**: Consider proximity to Qdrant and embedding services

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Commit: `git commit -am 'Add feature'`
5. Push: `git push origin feature-name`
6. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

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