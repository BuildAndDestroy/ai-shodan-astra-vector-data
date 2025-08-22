#!/usr/bin/env python3
"""
Setup configuration for vector-and-llm package.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="vector-and-llm",
    version="1.0.0",
    author="BuildAndDestroy",
    author_email="devreap1@gmail.com",
    description="A Python tool for importing Shodan JSON scan results into Qdrant vector database and performing RAG queries",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BuildAndDestroy/ai-shodan-astra-vector-data",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Topic :: Internet",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "bedrock": ["boto3>=1.40.0"],
        "dev": ["pytest>=7.0.0", "black>=22.0.0", "flake8>=4.0.0"],
    },
    entry_points={
        "console_scripts": [
            "shodan-to-qdrant=vector_and_llm.tools.shodan_to_qdrant:main",
            "llm-rag-shodan=vector_and_llm.tools.llm_rag_shodan:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)