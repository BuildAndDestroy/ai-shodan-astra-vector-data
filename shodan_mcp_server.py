#!/usr/bin/env python3
"""
MCP Server for Shodan Astra Linux Data
Provides structured access to Qdrant-stored Shodan scan results
"""

import json
import asyncio
from typing import Optional, List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

# Initialize Qdrant client
QDRANT_HOST = "192.168.122.16"
QDRANT_PORT = 6333
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Available collections
COLLECTIONS = [
    "shodan_query_august_2025",
    "shodan_query_december_2025", 
    "shodan_query_january_2026",
    "shodan_query_november_2025",
    "shodan_query_september_2025"
]

# Initialize MCP server
app = Server("shodan-astra-scanner")

def build_filter(
    country: Optional[str] = None,
    city: Optional[str] = None,
    product: Optional[str] = None,
    version_contains: Optional[str] = None,
    port: Optional[int] = None,
    ip_prefix: Optional[str] = None
) -> Optional[Filter]:
    """Build Qdrant filter from parameters"""
    conditions = []
    
    if country:
        conditions.append(
            FieldCondition(key="location.country_name", match=MatchValue(value=country))
        )
    
    if city:
        conditions.append(
            FieldCondition(key="location.city", match=MatchValue(value=city))
        )
    
    if product:
        conditions.append(
            FieldCondition(key="product", match=MatchValue(value=product))
        )
    
    if port:
        conditions.append(
            FieldCondition(key="port", match=MatchValue(value=port))
        )
    
    # Note: version_contains and ip_prefix require text matching
    # These would need custom implementation or payload filtering
    
    if not conditions:
        return None
    
    return Filter(must=conditions)


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available MCP tools"""
    return [
        Tool(
            name="search_astra_systems",
            description="Search for Astra Linux systems with filters",
            inputSchema={
                "type": "object",
                "properties": {
                    "country": {
                        "type": "string",
                        "description": "Filter by country name (e.g., 'United States', 'Russia')"
                    },
                    "city": {
                        "type": "string",
                        "description": "Filter by city name"
                    },
                    "product": {
                        "type": "string",
                        "description": "Filter by product (e.g., 'OpenSSH')"
                    },
                    "port": {
                        "type": "integer",
                        "description": "Filter by port number"
                    },
                    "collections": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": f"Collections to search (available: {', '.join(COLLECTIONS)})"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return (default: 100)",
                        "default": 100
                    }
                }
            }
        ),
        Tool(
            name="get_country_statistics",
            description="Get statistics about Astra Linux systems by country",
            inputSchema={
                "type": "object",
                "properties": {
                    "collections": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Collections to analyze"
                    }
                }
            }
        ),
        Tool(
            name="get_version_distribution",
            description="Get distribution of OpenSSH/Astra versions",
            inputSchema={
                "type": "object",
                "properties": {
                    "country": {
                        "type": "string",
                        "description": "Optional country filter"
                    },
                    "collections": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Collections to analyze"
                    }
                }
            }
        ),
        Tool(
            name="get_ip_details",
            description="Get detailed information about a specific IP",
            inputSchema={
                "type": "object",
                "properties": {
                    "ip_address": {
                        "type": "string",
                        "description": "IP address to lookup"
                    },
                    "collections": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Collections to search"
                    }
                },
                "required": ["ip_address"]
            }
        ),
        Tool(
            name="list_collections",
            description="List all available Shodan scan collections",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> List[TextContent]:
    """Handle tool calls"""
    
    if name == "list_collections":
        return [TextContent(
            type="text",
            text=json.dumps({
                "collections": COLLECTIONS,
                "count": len(COLLECTIONS)
            }, indent=2)
        )]
    
    elif name == "search_astra_systems":
        collections = arguments.get("collections", COLLECTIONS)
        limit = arguments.get("limit", 100)
        
        search_filter = build_filter(
            country=arguments.get("country"),
            city=arguments.get("city"),
            product=arguments.get("product"),
            port=arguments.get("port")
        )
        
        all_results = []
        for collection in collections:
            try:
                results = client.scroll(
                    collection_name=collection,
                    scroll_filter=search_filter,
                    limit=limit,
                    with_payload=True
                )
                
                for point in results[0]:
                    all_results.append({
                        "collection": collection,
                        "ip": point.payload["ip_str"],
                        "port": point.payload["port"],
                        "product": point.payload.get("product"),
                        "version": point.payload.get("version"),
                        "country": point.payload["location"]["country_name"],
                        "city": point.payload["location"]["city"],
                        "timestamp": point.payload["timestamp"]
                    })
            except Exception as e:
                all_results.append({
                    "error": f"Failed to query {collection}: {str(e)}"
                })
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "total_results": len(all_results),
                "results": all_results
            }, indent=2)
        )]
    
    elif name == "get_country_statistics":
        collections = arguments.get("collections", COLLECTIONS)
        country_stats = {}
        
        for collection in collections:
            try:
                # Scroll through all points (this is simplified - production would use aggregation)
                results = client.scroll(
                    collection_name=collection,
                    limit=10000,
                    with_payload=True
                )
                
                for point in results[0]:
                    country = point.payload["location"]["country_name"]
                    if country not in country_stats:
                        country_stats[country] = 0
                    country_stats[country] += 1
            except Exception as e:
                country_stats[f"error_{collection}"] = str(e)
        
        # Sort by count
        sorted_stats = dict(sorted(country_stats.items(), key=lambda x: x[1], reverse=True))
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "country_distribution": sorted_stats,
                "total_countries": len(sorted_stats),
                "total_systems": sum(sorted_stats.values())
            }, indent=2)
        )]
    
    elif name == "get_ip_details":
        ip_address = arguments["ip_address"]
        collections = arguments.get("collections", COLLECTIONS)
        
        results = []
        for collection in collections:
            try:
                # Search for exact IP match
                ip_filter = Filter(
                    must=[
                        FieldCondition(key="ip_str", match=MatchValue(value=ip_address))
                    ]
                )
                
                points = client.scroll(
                    collection_name=collection,
                    scroll_filter=ip_filter,
                    limit=10,
                    with_payload=True
                )
                
                for point in points[0]:
                    results.append({
                        "collection": collection,
                        "full_payload": point.payload
                    })
            except Exception as e:
                results.append({
                    "error": f"Failed to query {collection}: {str(e)}"
                })
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "ip": ip_address,
                "found_in_collections": len(results),
                "details": results
            }, indent=2)
        )]
    
    elif name == "get_version_distribution":
        collections = arguments.get("collections", COLLECTIONS)
        country_filter = arguments.get("country")
        
        version_stats = {}
        
        search_filter = None
        if country_filter:
            search_filter = Filter(
                must=[
                    FieldCondition(key="location.country_name", match=MatchValue(value=country_filter))
                ]
            )
        
        for collection in collections:
            try:
                results = client.scroll(
                    collection_name=collection,
                    scroll_filter=search_filter,
                    limit=10000,
                    with_payload=True
                )
                
                for point in results[0]:
                    version = point.payload.get("version", "unknown")
                    if version not in version_stats:
                        version_stats[version] = 0
                    version_stats[version] += 1
            except Exception as e:
                version_stats[f"error_{collection}"] = str(e)
        
        sorted_versions = dict(sorted(version_stats.items(), key=lambda x: x[1], reverse=True))
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "version_distribution": sorted_versions,
                "total_unique_versions": len(sorted_versions),
                "filter_applied": {"country": country_filter} if country_filter else None
            }, indent=2)
        )]
    
    else:
        return [TextContent(
            type="text",
            text=json.dumps({"error": f"Unknown tool: {name}"})
        )]


async def main():
    """Run the MCP server"""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())