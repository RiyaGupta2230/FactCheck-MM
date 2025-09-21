#!/usr/bin/env python3
"""
Knowledge Base Integration Utilities

Comprehensive integration with external knowledge bases including Wikidata,
DBpedia, and Wikipedia for enhanced fact verification with caching and
offline fallback capabilities.

Example Usage:
    >>> from fact_verification.utils import KnowledgeBaseConnector
    >>> 
    >>> # Initialize with caching enabled
    >>> kb_connector = KnowledgeBaseConnector(enable_cache=True)
    >>> 
    >>> # Query different knowledge bases
    >>> wikidata_info = kb_connector.query_wikidata("Albert Einstein")
    >>> dbpedia_info = kb_connector.query_dbpedia("Albert_Einstein")
    >>> wikipedia_summary = kb_connector.fetch_wikipedia_summary("Albert Einstein")
    >>> 
    >>> # Batch processing
    >>> entities = ["Barack Obama", "Climate change", "COVID-19"]
    >>> results = kb_connector.query_entities_batch(entities)
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import hashlib
import time
import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from urllib.parse import quote, urljoin
import re

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.utils.logging_utils import get_logger

# Optional dependencies for HTTP requests and parsing
try:
    import requests
    from requests.adapters import HTTPAdapter
    from requests.packages.urllib3.util.retry import Retry
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from SPARQLWrapper import SPARQLWrapper, JSON, XML
    SPARQL_AVAILABLE = True
except ImportError:
    SPARQL_AVAILABLE = False

try:
    import wikipedia
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


@dataclass
class KnowledgeBaseResult:
    """Container for knowledge base query results."""
    
    entity: str
    source: str
    data: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: str = ""
    cache_hit: bool = False
    query_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class EntityInfo:
    """Comprehensive entity information from multiple knowledge bases."""
    
    entity_name: str
    wikidata_id: str = ""
    dbpedia_uri: str = ""
    wikipedia_url: str = ""
    description: str = ""
    aliases: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    related_entities: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    coordinates: Optional[Tuple[float, float]] = None
    birth_date: Optional[str] = None
    death_date: Optional[str] = None
    occupation: List[str] = field(default_factory=list)
    nationality: List[str] = field(default_factory=list)


class KnowledgeBaseCache:
    """Local caching system for knowledge base queries."""
    
    def __init__(self, cache_dir: str = "data/cache", max_age_days: int = 30):
        """
        Initialize cache system.
        
        Args:
            cache_dir: Directory for cache storage
            max_age_days: Maximum age for cached entries in days
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_days = max_age_days
        
        # Initialize SQLite cache database
        self.db_path = self.cache_dir / "knowledge_base_cache.db"
        self._init_cache_db()
    
    def _init_cache_db(self):
        """Initialize cache database."""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_key TEXT UNIQUE NOT NULL,
                    entity TEXT NOT NULL,
                    source TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
    
    def _generate_cache_key(self, entity: str, source: str, query_params: Optional[Dict] = None) -> str:
        """Generate cache key for query."""
        
        key_data = f"{entity}:{source}"
        if query_params:
            key_data += f":{json.dumps(query_params, sort_keys=True)}"
        
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(
        self, 
        entity: str, 
        source: str, 
        query_params: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached result."""
        
        cache_key = self._generate_cache_key(entity, source, query_params)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                '''SELECT data, created_at FROM cache_entries 
                   WHERE cache_key = ? AND 
                   datetime(created_at) > datetime('now', '-{} days')'''.format(self.max_age_days),
                (cache_key,)
            )
            
            row = cursor.fetchone()
            if row:
                # Update access time
                conn.execute(
                    'UPDATE cache_entries SET accessed_at = CURRENT_TIMESTAMP WHERE cache_key = ?',
                    (cache_key,)
                )
                conn.commit()
                
                return json.loads(row[0])
        
        return None
    
    def set(
        self, 
        entity: str, 
        source: str, 
        data: Dict[str, Any], 
        query_params: Optional[Dict] = None
    ):
        """Store result in cache."""
        
        cache_key = self._generate_cache_key(entity, source, query_params)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                '''INSERT OR REPLACE INTO cache_entries 
                   (cache_key, entity, source, data, created_at, accessed_at) 
                   VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)''',
                (cache_key, entity, source, json.dumps(data))
            )
            conn.commit()
    
    def clear_expired(self):
        """Remove expired cache entries."""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                '''DELETE FROM cache_entries 
                   WHERE datetime(created_at) <= datetime('now', '-{} days')'''.format(self.max_age_days)
            )
            removed_count = cursor.rowcount
            conn.commit()
            
        return removed_count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        
        with sqlite3.connect(self.db_path) as conn:
            # Total entries
            total_entries = conn.execute('SELECT COUNT(*) FROM cache_entries').fetchone()[0]
            
            # Entries by source
            source_counts = conn.execute(
                'SELECT source, COUNT(*) FROM cache_entries GROUP BY source'
            ).fetchall()
            
            # Recent entries (last 7 days)
            recent_entries = conn.execute(
                '''SELECT COUNT(*) FROM cache_entries 
                   WHERE datetime(created_at) > datetime('now', '-7 days')'''
            ).fetchone()[0]
        
        return {
            'total_entries': total_entries,
            'source_distribution': dict(source_counts),
            'recent_entries': recent_entries,
            'cache_dir': str(self.cache_dir),
            'db_size_mb': self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0
        }


class KnowledgeBaseConnector:
    """
    Comprehensive knowledge base connector with support for multiple sources.
    
    Provides unified access to Wikidata, DBpedia, and Wikipedia with intelligent
    caching, offline fallback, and batch processing capabilities for enhanced
    fact verification workflows.
    """
    
    def __init__(
        self,
        enable_cache: bool = True,
        cache_dir: str = "data/cache",
        cache_max_age_days: int = 30,
        request_timeout: int = 30,
        max_retries: int = 3,
        enable_offline_fallback: bool = True,
        offline_data_dir: str = "data/offline_kb",
        user_agent: str = "FactCheck-MM/1.0",
        logger: Optional[Any] = None
    ):
        """
        Initialize knowledge base connector.
        
        Args:
            enable_cache: Enable local caching
            cache_dir: Cache directory path
            cache_max_age_days: Maximum cache age in days
            request_timeout: HTTP request timeout in seconds
            max_retries: Maximum number of request retries
            enable_offline_fallback: Enable offline data fallback
            offline_data_dir: Directory for offline knowledge base data
            user_agent: User agent string for HTTP requests
            logger: Optional logger instance
        """
        self.logger = logger or get_logger("KnowledgeBaseConnector")
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.enable_offline_fallback = enable_offline_fallback
        self.offline_data_dir = Path(offline_data_dir)
        self.offline_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize cache
        self.cache = None
        if enable_cache:
            self.cache = KnowledgeBaseCache(cache_dir, cache_max_age_days)
        
        # Initialize HTTP session
        self.session = None
        if REQUESTS_AVAILABLE:
            self.session = self._create_http_session(user_agent)
        
        # Initialize knowledge base endpoints
        self.endpoints = {
            'wikidata': 'https://query.wikidata.org/sparql',
            'dbpedia': 'https://dbpedia.org/sparql',
            'wikipedia_api': 'https://en.wikipedia.org/api/rest_v1/',
            'wikipedia_search': 'https://en.wikipedia.org/w/api.php'
        }
        
        # Initialize SPARQL wrappers
        self.sparql_wrappers = {}
        if SPARQL_AVAILABLE:
            self._init_sparql_wrappers()
        
        # Load offline data if available
        self.offline_data = {}
        if enable_offline_fallback:
            self._load_offline_data()
        
        self.logger.info("Initialized KnowledgeBaseConnector")
        self.logger.info(f"Available components: requests={REQUESTS_AVAILABLE}, "
                        f"sparql={SPARQL_AVAILABLE}, wikipedia={WIKIPEDIA_AVAILABLE}")
    
    def _create_http_session(self, user_agent: str):
        """Create configured HTTP session with retries."""
        
        session = requests.Session()
        session.headers.update({'User-Agent': user_agent})
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _init_sparql_wrappers(self):
        """Initialize SPARQL wrapper instances."""
        
        try:
            # Wikidata SPARQL endpoint
            self.sparql_wrappers['wikidata'] = SPARQLWrapper(self.endpoints['wikidata'])
            self.sparql_wrappers['wikidata'].setReturnFormat(JSON)
            
            # DBpedia SPARQL endpoint
            self.sparql_wrappers['dbpedia'] = SPARQLWrapper(self.endpoints['dbpedia'])
            self.sparql_wrappers['dbpedia'].setReturnFormat(JSON)
            
            self.logger.info("SPARQL wrappers initialized")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize SPARQL wrappers: {e}")
    
    def _load_offline_data(self):
        """Load offline knowledge base data."""
        
        # Load entity mappings
        mappings_file = self.offline_data_dir / "entity_mappings.json"
        if mappings_file.exists():
            try:
                with open(mappings_file, 'r') as f:
                    self.offline_data['entity_mappings'] = json.load(f)
                self.logger.info(f"Loaded {len(self.offline_data['entity_mappings'])} entity mappings")
            except Exception as e:
                self.logger.warning(f"Failed to load entity mappings: {e}")
        
        # Load entity summaries
        summaries_file = self.offline_data_dir / "entity_summaries.json"
        if summaries_file.exists():
            try:
                with open(summaries_file, 'r') as f:
                    self.offline_data['entity_summaries'] = json.load(f)
                self.logger.info(f"Loaded {len(self.offline_data['entity_summaries'])} entity summaries")
            except Exception as e:
                self.logger.warning(f"Failed to load entity summaries: {e}")
    
    def query_wikidata(
        self,
        entity: str,
        properties: Optional[List[str]] = None,
        limit: int = 10
    ) -> KnowledgeBaseResult:
        """
        Query Wikidata for entity information.
        
        Args:
            entity: Entity name or Wikidata ID
            properties: Specific properties to retrieve
            limit: Maximum number of results
            
        Returns:
            KnowledgeBaseResult with query results
        """
        start_time = time.time()
        source = "wikidata"
        
        # Check cache first
        cache_params = {'properties': properties, 'limit': limit}
        if self.cache:
            cached_result = self.cache.get(entity, source, cache_params)
            if cached_result:
                return KnowledgeBaseResult(
                    entity=entity,
                    source=source,
                    data=cached_result,
                    cache_hit=True,
                    query_time=time.time() - start_time
                )
        
        # Try online query
        if SPARQL_AVAILABLE and 'wikidata' in self.sparql_wrappers:
            try:
                result = self._query_wikidata_sparql(entity, properties, limit)
                
                # Cache successful result
                if self.cache and result['success']:
                    self.cache.set(entity, source, result, cache_params)
                
                return KnowledgeBaseResult(
                    entity=entity,
                    source=source,
                    data=result,
                    success=result['success'],
                    error_message=result.get('error', ''),
                    query_time=time.time() - start_time
                )
                
            except Exception as e:
                self.logger.warning(f"Wikidata SPARQL query failed: {e}")
        
        # Fallback to offline data
        if self.enable_offline_fallback:
            offline_result = self._query_offline_wikidata(entity)
            return KnowledgeBaseResult(
                entity=entity,
                source=f"{source}_offline",
                data=offline_result,
                success=offline_result['success'],
                error_message=offline_result.get('error', ''),
                query_time=time.time() - start_time
            )
        
        # Return empty result
        return KnowledgeBaseResult(
            entity=entity,
            source=source,
            data={'success': False, 'error': 'No data available'},
            success=False,
            error_message="No data available",
            query_time=time.time() - start_time
        )
    
    def _query_wikidata_sparql(
        self,
        entity: str,
        properties: Optional[List[str]] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Execute SPARQL query against Wikidata."""
        
        # Build SPARQL query
        if entity.startswith('Q') and entity[1:].isdigit():
            # Entity is already a Wikidata ID
            entity_filter = f"wd:{entity}"
        else:
            # Search by label
            entity_filter = f'?item rdfs:label "{entity}"@en'
        
        # Build property selection
        if properties:
            prop_queries = []
            for prop in properties:
                prop_queries.append(f'OPTIONAL {{ ?item wdt:{prop} ?{prop.lower()} }}')
            property_clause = '\n    '.join(prop_queries)
        else:
            # Default properties
            property_clause = '''
    OPTIONAL { ?item wdt:P31 ?instance_of }
    OPTIONAL { ?item wdt:P279 ?subclass_of }
    OPTIONAL { ?item wdt:P106 ?occupation }
    OPTIONAL { ?item wdt:P27 ?country }
    OPTIONAL { ?item wdt:P569 ?birth_date }
    OPTIONAL { ?item wdt:P570 ?death_date }
    OPTIONAL { ?item wdt:P625 ?coordinates }
    OPTIONAL { ?item schema:description ?description FILTER(LANG(?description) = "en") }
    '''
        
        query = f'''
        SELECT DISTINCT ?item ?itemLabel ?itemDescription {' '.join(f'?{p.lower()}' for p in (properties or [])) if properties else '?instance_of ?subclass_of ?occupation ?country ?birth_date ?death_date ?coordinates ?description'} WHERE {{
            {entity_filter} .
            {property_clause}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
        }}
        LIMIT {limit}
        '''
        
        try:
            sparql = self.sparql_wrappers['wikidata']
            sparql.setQuery(query)
            
            results = sparql.query().convert()
            
            if 'results' in results and 'bindings' in results['results']:
                bindings = results['results']['bindings']
                
                processed_results = []
                for binding in bindings:
                    result_item = {}
                    
                    for var, value in binding.items():
                        if 'value' in value:
                            result_item[var] = value['value']
                    
                    processed_results.append(result_item)
                
                return {
                    'success': True,
                    'results': processed_results,
                    'count': len(processed_results),
                    'query': query
                }
            else:
                return {
                    'success': False,
                    'error': 'No results found',
                    'results': [],
                    'count': 0
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'results': [],
                'count': 0
            }
    
    def _query_offline_wikidata(self, entity: str) -> Dict[str, Any]:
        """Query offline Wikidata cache."""
        
        if 'entity_mappings' not in self.offline_data:
            return {'success': False, 'error': 'No offline data available'}
        
        # Simple lookup in cached mappings
        entity_lower = entity.lower()
        for cached_entity, data in self.offline_data['entity_mappings'].items():
            if cached_entity.lower() == entity_lower:
                return {
                    'success': True,
                    'results': [data],
                    'count': 1,
                    'source': 'offline_cache'
                }
        
        return {'success': False, 'error': 'Entity not found in offline cache'}
    
    def query_dbpedia(
        self,
        entity: str,
        properties: Optional[List[str]] = None,
        limit: int = 10
    ) -> KnowledgeBaseResult:
        """
        Query DBpedia for entity information.
        
        Args:
            entity: Entity name or DBpedia URI
            properties: Specific properties to retrieve
            limit: Maximum number of results
            
        Returns:
            KnowledgeBaseResult with query results
        """
        start_time = time.time()
        source = "dbpedia"
        
        # Check cache first
        cache_params = {'properties': properties, 'limit': limit}
        if self.cache:
            cached_result = self.cache.get(entity, source, cache_params)
            if cached_result:
                return KnowledgeBaseResult(
                    entity=entity,
                    source=source,
                    data=cached_result,
                    cache_hit=True,
                    query_time=time.time() - start_time
                )
        
        # Try online query
        if SPARQL_AVAILABLE and 'dbpedia' in self.sparql_wrappers:
            try:
                result = self._query_dbpedia_sparql(entity, properties, limit)
                
                # Cache successful result
                if self.cache and result['success']:
                    self.cache.set(entity, source, result, cache_params)
                
                return KnowledgeBaseResult(
                    entity=entity,
                    source=source,
                    data=result,
                    success=result['success'],
                    error_message=result.get('error', ''),
                    query_time=time.time() - start_time
                )
                
            except Exception as e:
                self.logger.warning(f"DBpedia SPARQL query failed: {e}")
        
        # Fallback to offline data
        if self.enable_offline_fallback:
            offline_result = self._query_offline_dbpedia(entity)
            return KnowledgeBaseResult(
                entity=entity,
                source=f"{source}_offline",
                data=offline_result,
                success=offline_result['success'],
                error_message=offline_result.get('error', ''),
                query_time=time.time() - start_time
            )
        
        # Return empty result
        return KnowledgeBaseResult(
            entity=entity,
            source=source,
            data={'success': False, 'error': 'No data available'},
            success=False,
            error_message="No data available",
            query_time=time.time() - start_time
        )
    
    def _query_dbpedia_sparql(
        self,
        entity: str,
        properties: Optional[List[str]] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Execute SPARQL query against DBpedia."""
        
        # Convert entity name to DBpedia resource URI
        if not entity.startswith('http://dbpedia.org/resource/'):
            entity_uri = f"dbr:{entity.replace(' ', '_')}"
        else:
            entity_uri = f"<{entity}>"
        
        # Build property selection
        if properties:
            prop_queries = []
            for prop in properties:
                prop_queries.append(f'OPTIONAL {{ {entity_uri} dbo:{prop} ?{prop.lower()} }}')
            property_clause = '\n    '.join(prop_queries)
        else:
            # Default properties
            property_clause = f'''
    OPTIONAL {{ {entity_uri} dbo:abstract ?abstract FILTER(LANG(?abstract) = "en") }}
    OPTIONAL {{ {entity_uri} dbo:birthDate ?birthDate }}
    OPTIONAL {{ {entity_uri} dbo:deathDate ?deathDate }}
    OPTIONAL {{ {entity_uri} dbo:birthPlace ?birthPlace }}
    OPTIONAL {{ {entity_uri} dbo:nationality ?nationality }}
    OPTIONAL {{ {entity_uri} dbo:occupation ?occupation }}
    OPTIONAL {{ {entity_uri} rdfs:label ?label FILTER(LANG(?label) = "en") }}
    OPTIONAL {{ {entity_uri} rdfs:comment ?comment FILTER(LANG(?comment) = "en") }}
    '''
        
        query = f'''
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX dbr: <http://dbpedia.org/resource/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT DISTINCT * WHERE {{
            {property_clause}
        }}
        LIMIT {limit}
        '''
        
        try:
            sparql = self.sparql_wrappers['dbpedia']
            sparql.setQuery(query)
            
            results = sparql.query().convert()
            
            if 'results' in results and 'bindings' in results['results']:
                bindings = results['results']['bindings']
                
                processed_results = []
                for binding in bindings:
                    result_item = {}
                    
                    for var, value in binding.items():
                        if 'value' in value:
                            result_item[var] = value['value']
                    
                    processed_results.append(result_item)
                
                return {
                    'success': True,
                    'results': processed_results,
                    'count': len(processed_results),
                    'query': query
                }
            else:
                return {
                    'success': False,
                    'error': 'No results found',
                    'results': [],
                    'count': 0
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'results': [],
                'count': 0
            }
    
    def _query_offline_dbpedia(self, entity: str) -> Dict[str, Any]:
        """Query offline DBpedia cache."""
        
        # Similar to offline Wikidata query
        return {'success': False, 'error': 'DBpedia offline data not implemented'}
    
    def fetch_wikipedia_summary(
        self,
        entity: str,
        sentences: int = 3,
        auto_suggest: bool = True
    ) -> KnowledgeBaseResult:
        """
        Fetch Wikipedia summary for entity.
        
        Args:
            entity: Entity name
            sentences: Number of sentences in summary
            auto_suggest: Enable auto-suggestion for entity names
            
        Returns:
            KnowledgeBaseResult with Wikipedia summary
        """
        start_time = time.time()
        source = "wikipedia"
        
        # Check cache first
        cache_params = {'sentences': sentences, 'auto_suggest': auto_suggest}
        if self.cache:
            cached_result = self.cache.get(entity, source, cache_params)
            if cached_result:
                return KnowledgeBaseResult(
                    entity=entity,
                    source=source,
                    data=cached_result,
                    cache_hit=True,
                    query_time=time.time() - start_time
                )
        
        # Try Wikipedia API
        if WIKIPEDIA_AVAILABLE:
            try:
                result = self._fetch_wikipedia_api(entity, sentences, auto_suggest)
                
                # Cache successful result
                if self.cache and result['success']:
                    self.cache.set(entity, source, result, cache_params)
                
                return KnowledgeBaseResult(
                    entity=entity,
                    source=source,
                    data=result,
                    success=result['success'],
                    error_message=result.get('error', ''),
                    query_time=time.time() - start_time
                )
                
            except Exception as e:
                self.logger.warning(f"Wikipedia API query failed: {e}")
        
        # Fallback to REST API if available
        if self.session:
            try:
                result = self._fetch_wikipedia_rest(entity)
                
                if self.cache and result['success']:
                    self.cache.set(entity, source, result, cache_params)
                
                return KnowledgeBaseResult(
                    entity=entity,
                    source=f"{source}_rest",
                    data=result,
                    success=result['success'],
                    error_message=result.get('error', ''),
                    query_time=time.time() - start_time
                )
                
            except Exception as e:
                self.logger.warning(f"Wikipedia REST API query failed: {e}")
        
        # Fallback to offline data
        if self.enable_offline_fallback:
            offline_result = self._query_offline_wikipedia(entity)
            return KnowledgeBaseResult(
                entity=entity,
                source=f"{source}_offline",
                data=offline_result,
                success=offline_result['success'],
                error_message=offline_result.get('error', ''),
                query_time=time.time() - start_time
            )
        
        # Return empty result
        return KnowledgeBaseResult(
            entity=entity,
            source=source,
            data={'success': False, 'error': 'No data available'},
            success=False,
            error_message="No data available",
            query_time=time.time() - start_time
        )
    
    def _fetch_wikipedia_api(
        self,
        entity: str,
        sentences: int = 3,
        auto_suggest: bool = True
    ) -> Dict[str, Any]:
        """Fetch Wikipedia content using wikipedia-api library."""
        
        try:
            # Set language
            wikipedia.set_lang("en")
            
            # Search for the page
            if auto_suggest:
                try:
                    # Try to find the page, with auto-suggestion
                    page = wikipedia.page(entity, auto_suggest=True)
                except wikipedia.DisambiguationError as e:
                    # Take the first disambiguation option
                    page = wikipedia.page(e.options[0])
                except wikipedia.PageError:
                    # Try searching
                    search_results = wikipedia.search(entity, results=1)
                    if search_results:
                        page = wikipedia.page(search_results[0])
                    else:
                        return {
                            'success': False,
                            'error': f'No Wikipedia page found for "{entity}"'
                        }
            else:
                page = wikipedia.page(entity)
            
            # Get summary
            summary = wikipedia.summary(page.title, sentences=sentences)
            
            return {
                'success': True,
                'title': page.title,
                'summary': summary,
                'url': page.url,
                'categories': page.categories if hasattr(page, 'categories') else [],
                'links': page.links[:20] if hasattr(page, 'links') else [],  # Limit to first 20 links
                'images': page.images[:5] if hasattr(page, 'images') else [],  # Limit to first 5 images
                'content_length': len(page.content) if hasattr(page, 'content') else 0
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _fetch_wikipedia_rest(self, entity: str) -> Dict[str, Any]:
        """Fetch Wikipedia content using REST API."""
        
        try:
            # Encode entity name for URL
            encoded_entity = quote(entity.replace(' ', '_'))
            
            # Try to get page summary
            summary_url = f"{self.endpoints['wikipedia_api']}page/summary/{encoded_entity}"
            
            response = self.session.get(summary_url, timeout=self.request_timeout)
            
            if response.status_code == 200:
                data = response.json()
                
                return {
                    'success': True,
                    'title': data.get('title', ''),
                    'summary': data.get('extract', ''),
                    'url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                    'description': data.get('description', ''),
                    'thumbnail': data.get('thumbnail', {}),
                    'coordinates': data.get('coordinates', {}),
                    'page_id': data.get('pageid', '')
                }
            else:
                return {
                    'success': False,
                    'error': f'HTTP {response.status_code}: {response.text}'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _query_offline_wikipedia(self, entity: str) -> Dict[str, Any]:
        """Query offline Wikipedia cache."""
        
        if 'entity_summaries' not in self.offline_data:
            return {'success': False, 'error': 'No offline Wikipedia data available'}
        
        entity_lower = entity.lower()
        for cached_entity, summary in self.offline_data['entity_summaries'].items():
            if cached_entity.lower() == entity_lower:
                return {
                    'success': True,
                    'title': cached_entity,
                    'summary': summary,
                    'source': 'offline_cache'
                }
        
        return {'success': False, 'error': 'Entity not found in offline cache'}
    
    def query_entities_batch(
        self,
        entities: List[str],
        sources: Optional[List[str]] = None,
        max_workers: int = 5
    ) -> Dict[str, List[KnowledgeBaseResult]]:
        """
        Query multiple entities across knowledge bases.
        
        Args:
            entities: List of entity names
            sources: Knowledge base sources to query
            max_workers: Maximum concurrent workers
            
        Returns:
            Dictionary mapping entities to their results across sources
        """
        if not sources:
            sources = ['wikidata', 'dbpedia', 'wikipedia']
        
        results = {}
        
        # Process entities sequentially to avoid overwhelming APIs
        for entity in entities:
            entity_results = []
            
            for source in sources:
                if source == 'wikidata':
                    result = self.query_wikidata(entity)
                elif source == 'dbpedia':
                    result = self.query_dbpedia(entity)
                elif source == 'wikipedia':
                    result = self.fetch_wikipedia_summary(entity)
                else:
                    continue
                
                entity_results.append(result)
                
                # Small delay to be respectful to APIs
                time.sleep(0.1)
            
            results[entity] = entity_results
        
        return results
    
    def create_unified_entity_info(
        self,
        entity: str,
        query_all_sources: bool = True
    ) -> EntityInfo:
        """
        Create unified entity information from multiple sources.
        
        Args:
            entity: Entity name
            query_all_sources: Whether to query all available sources
            
        Returns:
            EntityInfo object with consolidated information
        """
        entity_info = EntityInfo(entity_name=entity)
        
        if query_all_sources:
            # Query all sources
            wikidata_result = self.query_wikidata(entity)
            dbpedia_result = self.query_dbpedia(entity)
            wikipedia_result = self.fetch_wikipedia_summary(entity)
            
            # Merge results
            if wikidata_result.success and wikidata_result.data.get('results'):
                wd_data = wikidata_result.data['results'][0]
                entity_info.wikidata_id = wd_data.get('item', '').split('/')[-1] if 'item' in wd_data else ''
                entity_info.description = wd_data.get('itemDescription', '')
                
                # Extract dates
                if 'birth_date' in wd_data:
                    entity_info.birth_date = wd_data['birth_date']
                if 'death_date' in wd_data:
                    entity_info.death_date = wd_data['death_date']
                
                # Extract coordinates
                if 'coordinates' in wd_data:
                    coords_str = wd_data['coordinates']
                    # Parse coordinates (simplified)
                    coord_match = re.search(r'Point\(([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\)', coords_str)
                    if coord_match:
                        entity_info.coordinates = (float(coord_match.group(2)), float(coord_match.group(1)))
            
            if dbpedia_result.success and dbpedia_result.data.get('results'):
                db_data = dbpedia_result.data['results'][0]
                entity_info.dbpedia_uri = f"http://dbpedia.org/resource/{entity.replace(' ', '_')}"
                
                if 'abstract' in db_data and not entity_info.description:
                    entity_info.description = db_data['abstract'][:500] + "..." if len(db_data['abstract']) > 500 else db_data['abstract']
            
            if wikipedia_result.success:
                wp_data = wikipedia_result.data
                entity_info.wikipedia_url = wp_data.get('url', '')
                
                if not entity_info.description and 'summary' in wp_data:
                    entity_info.description = wp_data['summary']
                
                if 'categories' in wp_data:
                    entity_info.categories = wp_data['categories'][:10]  # Limit categories
        
        return entity_info
    
    def export_cache_data(
        self,
        output_file: str,
        format: str = 'json'
    ):
        """
        Export cached data for offline use.
        
        Args:
            output_file: Output file path
            format: Export format ('json' or 'csv')
        """
        if not self.cache:
            self.logger.warning("No cache available for export")
            return
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get all cache entries
        with sqlite3.connect(self.cache.db_path) as conn:
            cursor = conn.execute(
                'SELECT entity, source, data, created_at FROM cache_entries ORDER BY created_at DESC'
            )
            
            entries = cursor.fetchall()
        
        if format == 'json':
            export_data = []
            for entity, source, data_str, created_at in entries:
                try:
                    data = json.loads(data_str)
                    export_data.append({
                        'entity': entity,
                        'source': source,
                        'data': data,
                        'created_at': created_at
                    })
                except json.JSONDecodeError:
                    continue
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
        
        elif format == 'csv' and PANDAS_AVAILABLE:
            rows = []
            for entity, source, data_str, created_at in entries:
                try:
                    data = json.loads(data_str)
                    rows.append({
                        'entity': entity,
                        'source': source,
                        'success': data.get('success', False),
                        'result_count': len(data.get('results', [])),
                        'created_at': created_at
                    })
                except json.JSONDecodeError:
                    continue
            
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
        
        self.logger.info(f"Exported {len(entries)} cache entries to {output_path}")
    
    def get_connector_info(self) -> Dict[str, Any]:
        """Get information about connector configuration and capabilities."""
        
        cache_stats = {}
        if self.cache:
            cache_stats = self.cache.get_cache_stats()
        
        return {
            'endpoints': self.endpoints,
            'components_available': {
                'requests': REQUESTS_AVAILABLE,
                'sparql': SPARQL_AVAILABLE,
                'wikipedia': WIKIPEDIA_AVAILABLE,
                'bs4': BS4_AVAILABLE,
                'pandas': PANDAS_AVAILABLE
            },
            'cache_enabled': self.cache is not None,
            'cache_stats': cache_stats,
            'offline_fallback_enabled': self.enable_offline_fallback,
            'offline_data_loaded': len(self.offline_data),
            'request_timeout': self.request_timeout,
            'max_retries': self.max_retries
        }


def main():
    """Example usage of KnowledgeBaseConnector."""
    
    # Initialize connector
    kb_connector = KnowledgeBaseConnector(
        enable_cache=True,
        cache_max_age_days=7,  # Short cache for demo
        request_timeout=10
    )
    
    print("=== KnowledgeBaseConnector Example ===")
    print(f"Connector info: {kb_connector.get_connector_info()}")
    
    # Test entities
    test_entities = [
        "Albert Einstein",
        "Barack Obama", 
        "Climate change",
        "COVID-19"
    ]
    
    print(f"\nTesting knowledge base queries...")
    
    # Test individual queries
    for entity in test_entities[:2]:  # Limit for demo
        print(f"\n--- Querying: {entity} ---")
        
        # Wikipedia query
        wikipedia_result = kb_connector.fetch_wikipedia_summary(entity, sentences=2)
        print(f"Wikipedia: {'Success' if wikipedia_result.success else 'Failed'}")
        if wikipedia_result.success:
            summary = wikipedia_result.data.get('summary', '')
            print(f"  Summary: {summary[:100]}...")
            print(f"  Cache hit: {wikipedia_result.cache_hit}")
            print(f"  Query time: {wikipedia_result.query_time:.3f}s")
        
        # Wikidata query (if available)
        if SPARQL_AVAILABLE:
            wikidata_result = kb_connector.query_wikidata(entity, limit=1)
            print(f"Wikidata: {'Success' if wikidata_result.success else 'Failed'}")
            if wikidata_result.success and wikidata_result.data.get('results'):
                result = wikidata_result.data['results'][0]
                print(f"  Label: {result.get('itemLabel', 'N/A')}")
                print(f"  Description: {result.get('itemDescription', 'N/A')[:100]}...")
                print(f"  Cache hit: {wikidata_result.cache_hit}")
        
        # DBpedia query (if available)
        if SPARQL_AVAILABLE:
            dbpedia_result = kb_connector.query_dbpedia(entity, limit=1)
            print(f"DBpedia: {'Success' if dbpedia_result.success else 'Failed'}")
            if dbpedia_result.success and dbpedia_result.data.get('results'):
                result = dbpedia_result.data['results'][0]
                print(f"  Abstract: {result.get('abstract', 'N/A')[:100]}...")
                print(f"  Cache hit: {dbpedia_result.cache_hit}")
    
    # Test batch processing
    print(f"\n=== Batch Processing Test ===")
    batch_results = kb_connector.query_entities_batch(
        test_entities[:2],  # Limit for demo
        sources=['wikipedia'],  # Just Wikipedia for demo
        max_workers=2
    )
    
    for entity, results in batch_results.items():
        print(f"\n{entity}:")
        for result in results:
            print(f"  {result.source}: {'Success' if result.success else 'Failed'} "
                  f"(cache: {result.cache_hit}, time: {result.query_time:.3f}s)")
    
    # Test unified entity info
    print(f"\n=== Unified Entity Info Test ===")
    unified_info = kb_connector.create_unified_entity_info(
        "Albert Einstein",
        query_all_sources=False  # Only use cache/offline for demo
    )
    
    print(f"Entity: {unified_info.entity_name}")
    print(f"Description: {unified_info.description[:100]}..." if unified_info.description else "No description")
    print(f"Wikipedia URL: {unified_info.wikipedia_url}")
    print(f"Categories: {unified_info.categories[:3]}")
    
    # Cache statistics
    if kb_connector.cache:
        print(f"\n=== Cache Statistics ===")
        cache_stats = kb_connector.cache.get_cache_stats()
        for key, value in cache_stats.items():
            print(f"  {key}: {value}")
        
        # Clear expired entries
        expired_count = kb_connector.cache.clear_expired()
        print(f"  Expired entries removed: {expired_count}")
    
    # Export cache (if data available)
    if kb_connector.cache:
        try:
            cache_stats = kb_connector.cache.get_cache_stats()
            if cache_stats['total_entries'] > 0:
                print(f"\n=== Cache Export Test ===")
                export_file = "data/cache/knowledge_base_export.json"
                kb_connector.export_cache_data(export_file, format='json')
                print(f"Cache exported to: {export_file}")
        except Exception as e:
            print(f"Cache export failed: {e}")


if __name__ == "__main__":
    main()
