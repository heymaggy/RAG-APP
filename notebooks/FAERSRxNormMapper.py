# File: FAERSRxNormMapper.py
# Description: A class to map drug names to RxNorm concepts using the NIH RxNav API.
#              Includes caching to a local SQLite database to minimize API calls.
# Author: Gemini
# Date: August 03, 2025

import requests
import sqlite3
import time
import pandas as pd
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

class FAERSRxNormMapper:
    """
    Maps drug names to their standardized RxNorm counterparts using the NIH RxNav API.
    Caches results in a local SQLite database to improve performance and reduce API load.
    """
    
    def __init__(self, cache_db: str = "faers_rxnorm_cache.db"):
        """
        Initializes the mapper, sets up the API session, and connects to the cache.

        Args:
            cache_db (str): The file path for the SQLite cache database.
        """
        self.base_url = "https://rxnav.nlm.nih.gov/REST"
        self.cache_db = cache_db
        self.cache_conn = None
        self._connect_to_cache()
        
        # Set up a robust requests session with retries for network resilience
        self.session = self._create_robust_session()
        
        print(f"FAERSRxNormMapper initialized. Cache located at: '{self.cache_db}'")

    def _connect_to_cache(self):
        """Establishes a connection to the SQLite database and creates the cache table if it doesn't exist."""
        try:
            self.cache_conn = sqlite3.connect(self.cache_db, check_same_thread=False)
            cursor = self.cache_conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rxnorm_cache (
                    drug_name TEXT PRIMARY KEY,
                    standard_name TEXT,
                    rxcui TEXT,
                    score REAL,
                    mapping_method TEXT
                )
            ''')
            self.cache_conn.commit()
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            # Exit or handle gracefully if cache is critical
            raise

    def _create_robust_session(self):
        """
        Creates a requests.Session object with automatic retries for HTTP errors.
        This makes the API calls more resilient to transient network issues.
        """
        session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504] # Retry on server errors
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)
        return session

    def _query_cache(self, drug_name: str):
        """
        Checks the local cache for a previously mapped drug name.

        Args:
            drug_name (str): The drug name to look up.

        Returns:
            A dictionary with the cached result, or None if not found.
        """
        try:
            cursor = self.cache_conn.cursor()
            cursor.execute(
                "SELECT standard_name, rxcui, score, mapping_method FROM rxnorm_cache WHERE drug_name = ?",
                (drug_name,)
            )
            result = cursor.fetchone()
            if result:
                return {
                    'standard_name': result[0],
                    'rxcui': result[1],
                    'score': result[2],
                    'mapping_method': result[3]
                }
        except sqlite3.Error as e:
            print(f"Cache query error for '{drug_name}': {e}")
        return None

    def _update_cache(self, drug_name: str, result: dict):
        """
        Stores a new mapping result in the local cache.

        Args:
            drug_name (str): The original drug name (the key).
            result (dict): The mapping result to store.
        """
        try:
            cursor = self.cache_conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO rxnorm_cache (drug_name, standard_name, rxcui, score, mapping_method) VALUES (?, ?, ?, ?, ?)",
                (
                    drug_name,
                    result.get('standard_name'),
                    result.get('rxcui'),
                    result.get('score'),
                    result.get('mapping_method')
                )
            )
            self.cache_conn.commit()
        except sqlite3.Error as e:
            print(f"Cache update error for '{drug_name}': {e}")

    def _query_rxnorm_api(self, drug_name: str):
        """
        Queries the RxNorm API to get the standardized name and RxCUI.
        
        Args:
            drug_name (str): The drug name to standardize.

        Returns:
            A dictionary with the mapping result, or None if no match is found.
        """
        url = f"{self.base_url}/approximateTerm.json"
        params = {'term': drug_name, 'maxEntries': 1}
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status() # Raises an exception for bad status codes
            data = response.json()
            
            candidates = data.get('approximateGroup', {}).get('candidate', [])
            if candidates:
                best_match = candidates[0]
                return {
                    'standard_name': best_match.get('rxcuiName'),
                    'rxcui': best_match.get('rxcui'),
                    'score': best_match.get('score'),
                    'mapping_method': 'approximateTerm'
                }
        except requests.exceptions.RequestException as e:
            print(f"API request failed for '{drug_name}': {e}")
        
        return None

    def get_standardized_drug_name(self, drug_name: str):
        """
        The main public method. Gets a standardized drug name, using the cache first.
        If not in the cache, it queries the RxNorm API and caches the result.

        Args:
            drug_name (str): The drug name to standardize.

        Returns:
            A dictionary containing the standardized info, or None if it cannot be mapped.
        """
        if not drug_name or pd.isna(drug_name):
            return None
        
        # Check cache first
        cached_result = self._query_cache(drug_name)
        if cached_result:
            cached_result['mapping_method'] = 'cache'
            return cached_result
        
        # If not in cache, query API
        api_result = self._query_rxnorm_api(drug_name)
        
        # If API returns a result, update cache
        if api_result:
            self._update_cache(drug_name, api_result)
        else:
            # Cache failures too, to avoid re-querying for names known to fail
            self._update_cache(drug_name, {'mapping_method': 'api_fail'})

        # A small delay to respect API rate limits (40 requests/second max)
        time.sleep(0.03)
        
        return api_result

    def close_connection(self):
        """Closes the connection to the cache database."""
        if self.cache_conn:
            self.cache_conn.close()
            print("Cache connection closed.")
