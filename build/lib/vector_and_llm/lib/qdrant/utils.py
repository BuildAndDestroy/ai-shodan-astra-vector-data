#!/usr/bin/env python3
"""
Qdrant utility functions.
"""

from typing import Dict, Any


def create_embedding_text(match: Dict[str, Any]) -> str:
    """Create text string from Shodan match for embedding.
    
    Location information is given MUCH higher weight for better semantic search matching.
    The embedding text is structured with location first and repeated for better vector alignment.
    """
    # Build location context with heavy weighting
    location_parts = []
    if match.get("location"):
        location = match["location"]
        if location.get("country_name"):
            country = location["country_name"]
            location_parts.append(country)
            
            # Add common country name variations for better matching
            country_variations = {
                "Russian Federation": ["Russia", "Russian Federation", "RU"],
                "United States": ["USA", "US", "United States", "America"],
                "United Kingdom": ["UK", "Britain", "United Kingdom", "GB"],
                "China": ["China", "People's Republic of China", "PRC", "CN"],
                "Germany": ["Germany", "Deutschland", "DE"],
                "Canada": ["Canada", "CA"],
                "France": ["France", "FR"],
                "India": ["India", "IN"],
                "Japan": ["Japan", "JP"],
                "Australia": ["Australia", "AU"],
                "Brazil": ["Brazil", "BR"],
                "Mexico": ["Mexico", "MX"],
            }
            
            for full_name, variations in country_variations.items():
                if country == full_name:
                    location_parts.extend(variations)
                    break
        
        if location.get("country_code"):
            location_parts.append(location["country_code"])
            
        if location.get("city"):
            location_parts.append(location["city"])
    
    # PRIORITY: Location information first (repeated 3x for semantic weight)
    parts = []
    if location_parts:
        location_str = " ".join(location_parts)
        # Repeat location info multiple times to heavily weight it in embeddings
        parts.append(f"Location: {location_str}")
        parts.append(f"Geographic Region: {location_str}")
        parts.append(f"Server Location: {location_str}")
    
    # Service information (weighted less than location)
    if match.get("product"):
        parts.append(f"Service: {match['product']}")
        
    if match.get("version"):
        parts.append(f"Version: {match['version']}")
    
    # Network details
    if match.get("ip_str"):
        parts.append(f"IP: {match['ip_str']}")
    
    if match.get("port"):
        parts.append(f"Port: {match['port']}")
    
    # SSH-specific info if available
    if match.get("ssh") and match["ssh"].get("type"):
        parts.append(f"SSH Key Type: {match['ssh']['type']}")
        
    # Optional banner data (truncated to avoid token limits)
    if match.get("data"):
        banner = match["data"][:200]  # Reduced from 500 to avoid diluting location signal
        parts.append(f"Banner Data: {banner}")
    
    return " | ".join(parts)


def format_search_results(results: list) -> str:
    """Format search results into readable context for LLM."""
    if not results:
        return "No relevant data found in the database."
    
    context_parts = []
    for i, result in enumerate(results, 1):
        payload = result.get("payload", {})
        score = result.get("score", 0)
        collection = result.get("collection", "unknown")
        
        # Format individual result
        parts = []
        if payload.get("ip_str"):
            parts.append(f"IP: {payload['ip_str']}")
        if payload.get("port"):
            parts.append(f"Port: {payload['port']}")
        if payload.get("product"):
            parts.append(f"Service: {payload['product']}")
        if payload.get("version"):
            parts.append(f"Version: {payload['version']}")
        
        location = payload.get("location", {})
        if location:
            loc_parts = []
            if location.get("city"):
                loc_parts.append(location["city"])
            if location.get("country_name"):
                loc_parts.append(location["country_name"])
            if loc_parts:
                parts.append(f"Location: {', '.join(loc_parts)}")
        
        if payload.get("timestamp"):
            parts.append(f"Timestamp: {payload['timestamp']}")
        
        # Add collection information
        parts.append(f"Dataset: {collection}")
        
        result_text = f"Result {i} (similarity: {score:.3f}): {' | '.join(parts)}"
        context_parts.append(result_text)
    
    return "\n".join(context_parts)