#!/usr/bin/env python3
"""
Qdrant utility functions.
"""

from typing import Dict, Any


def create_embedding_text(match: Dict[str, Any]) -> str:
    """Create text string from Shodan match for embedding."""
    # Combine key fields for semantic embedding
    parts = []
    
    if match.get("ip_str"):
        parts.append(f"IP: {match['ip_str']}")
    
    if match.get("port"):
        parts.append(f"Port: {match['port']}")
        
    if match.get("product"):
        parts.append(f"Service: {match['product']}")
        
    if match.get("version"):
        parts.append(f"Version: {match['version']}")
        
    # Enhanced location handling with multiple variations
    if match.get("location"):
        location = match["location"]
        if location.get("country_name"):
            country = location["country_name"]
            parts.append(f"Country: {country}")
            
            # Add common country name variations for better matching
            country_variations = {
                "Russian Federation": ["Russia", "Russian Federation", "RU"],
                "United States": ["USA", "US", "United States", "America"],
                "United Kingdom": ["UK", "Britain", "United Kingdom", "GB"],
                "China": ["China", "People's Republic of China", "PRC", "CN"],
                "Germany": ["Germany", "Deutschland", "DE"],
                # Add more as needed
            }
            
            for full_name, variations in country_variations.items():
                if country == full_name:
                    parts.extend([f"Location: {var}" for var in variations])
                    break
            
        if location.get("country_code"):
            parts.append(f"Country Code: {location['country_code']}")
            
        if location.get("city"):
            parts.append(f"City: {location['city']}")
            
        # Add geographic context
        if location.get("country_name") and location.get("city"):
            parts.append(f"Geographic Location: {location['city']}, {location['country_name']}")
    
    # Add SSH-specific info if available
    if match.get("ssh") and match["ssh"].get("type"):
        parts.append(f"SSH Key Type: {match['ssh']['type']}")
        
    # Optionally include banner data (truncated to avoid token limits)
    if match.get("data"):
        banner = match["data"][:500]  # First 500 chars
        parts.append(f"Banner: {banner}")
    
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