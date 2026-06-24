import json
import utils
import mlflow
import logging
from uuid import uuid4
from datetime import datetime
from databricks.vector_search.client import VectorSearchClient

class Cache:
    def __init__(self, vsc, config):
        mlflow.set_tracking_uri("databricks")
        self.vsc = vsc
        self.config = config
        self._index = None

    def _get_index(self):
        """Return a cached index handle, creating it on first access."""
        if self._index is None:
            self._index = self.vsc.get_index(
                index_name=self.config.VS_INDEX_FULLNAME_CACHE,
                endpoint_name=self.config.VECTOR_SEARCH_ENDPOINT_NAME_CACHE,
            )
        return self._index

    def create_cache(self):
        # Create or wait for the endpoint
        utils.create_or_wait_for_endpoint(self.vsc, self.config.VECTOR_SEARCH_ENDPOINT_NAME_CACHE)
        logging.info(f"Vector search endpoint '{self.config.VECTOR_SEARCH_ENDPOINT_NAME_CACHE}' is ready")

        # Create or update the main index
        utils.create_or_update_direct_index(
            self.vsc,
            self.config.VECTOR_SEARCH_ENDPOINT_NAME_CACHE,
            self.config.VS_INDEX_FULLNAME_CACHE,
            self.config.VECTOR_SEARCH_INDEX_SCHEMA_CACHE,
            self.config.VECTOR_SEARCH_INDEX_CONFIG_CACHE,
        )
        # Invalidate cached handle — the index may have been recreated
        self._index = None
        logging.info(f"Main index '{self.config.VS_INDEX_FULLNAME_CACHE}' created/updated and is ready")
        logging.info("Environment setup completed successfully")

    @staticmethod
    def load_data(file_path):
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                data.append(json.loads(line))
        return data

    def get_embedding(self, text):
        from mlflow.deployments import get_deploy_client
        client = get_deploy_client("databricks")
        response = client.predict(
        endpoint=self.config.EMBEDDING_MODEL_SERVING_ENDPOINT_NAME,
        inputs={"input": [text]})
        return response.data[0]['embedding']

    def warm_cache(self, batch_size=100):
        vs_index_cache = self._get_index()
        # Load dataset
        data = Cache.load_data(self.config.CACHE_WARMING_FILE_PATH)
        logging.info(f"Loaded {len(data)} documents from {self.config.CACHE_WARMING_FILE_PATH}")
        documents = []
        now = datetime.now().isoformat()
        for idx, item in enumerate(data):
            if 'question' in item and 'answer' in item:
                embedding = self.get_embedding(item['question'])
                doc = {
                    "id": str(idx),
                    "creator": "system",
                    "question": item["question"],
                    "answer": item["answer"],
                    "access_level": 0,
                    "created_at": now,
                    "last_accessed": now,
                    "text_vector": embedding
                }
                documents.append(doc)

            # Upsert when batch size is reached
            if len(documents) >= batch_size:
                try:
                    vs_index_cache.upsert(documents)
                    print(f"Successfully upserted batch of {len(documents)} documents.")
                except Exception as e:
                    print(f"Error upserting batch: {str(e)}")
                documents = []  # Clear the batch

        # Upsert any remaining documents
        if documents:
            try:
                vs_index_cache.upsert(documents)
                print(f"Successfully upserted final batch of {len(documents)} documents.")
            except Exception as e:
                print(f"Error upserting final batch: {str(e)}")

        logging.info(f"Finished loading documents into the index.")
        logging.info("Cache warming completed successfully")

    def get_from_cache(self, question, creator=None, access_level=None):
        """Look up a semantically similar question in the cache.

        Args:
            question: The user query string.
            creator: If set, only return cache entries created by this creator.
            access_level: If set, only return entries with access_level <= this value.

        Returns:
            dict with 'question', 'answer' (empty string on miss), and 'cache_hit' bool.
        """
        vs_index_cache = self._get_index()
        qa = {"question": question, "answer": "", "cache_hit": False}

        results = vs_index_cache.similarity_search(
            query_vector=self.get_embedding(question),
            columns=["id", "question", "answer", "creator", "access_level"],
            num_results=3  # fetch a few candidates for filtering
        )

        if not results or results['result']['row_count'] == 0:
            return qa

        for row in results['result']['data_array']:
            record_id = row[0]
            cached_question = row[1]
            cached_answer = row[2]
            cached_creator = row[3]
            cached_access_level = row[4]
            score = row[5]  # score is appended as the last column

            try:
                if float(score) < self.config.SIMILARITY_THRESHOLD:
                    continue  # below threshold — not similar enough

                # Apply access control filters
                if access_level is not None and cached_access_level is not None:
                    if int(cached_access_level) > int(access_level):
                        continue  # user lacks access
                if creator is not None and cached_creator != creator:
                    continue  # creator mismatch

                # Cache hit — update last_accessed timestamp
                qa["answer"] = cached_answer
                qa["cache_hit"] = True
                logging.info(f"Cache hit: score={score}")
                self._touch_entry(record_id)
                return qa

            except (ValueError, TypeError):
                logging.warning(f"Invalid score or access_level value in cache entry {record_id}")
                continue

        logging.info("Cache hit: False (no candidates passed threshold/filters)")
        return qa

    def _touch_entry(self, record_id):
        """Update last_accessed timestamp on a cache entry (best-effort)."""
        try:
            vs_index_cache = self._get_index()
            vs_index_cache.upsert([{
                "id": record_id,
                "last_accessed": datetime.now().isoformat()
            }])
        except Exception as e:
            # Non-critical — don't fail the request if timestamp update fails
            logging.debug(f"Failed to update last_accessed for {record_id}: {e}")

    def store_in_cache(self, question, answer, creator="user", access_level=0):
        """Store a response in the cache if it meets minimum quality criteria."""
        min_len = getattr(self.config, 'MIN_RESPONSE_LENGTH', 0)
        if min_len and len(answer.strip()) < min_len:
            logging.info(f"Response too short ({len(answer.strip())} chars), skipping cache store")
            return

        vs_index_cache = self._get_index()
        now = datetime.now().isoformat()
        document = {
            "id": str(uuid4()),
            "creator": creator,
            "question": question,
            "answer": answer,
            "access_level": access_level,
            "created_at": now,
            "last_accessed": now,
            "text_vector": self.get_embedding(question),
        }
        vs_index_cache.upsert([document])

    def evict(self, strategy='FIFO', max_documents=1000, batch_size=100):
        total_docs = self.get_indexed_row_count()

        if total_docs <= max_documents:
            logging.info(f"Cache size ({total_docs}) is within limit ({max_documents}). No eviction needed.")
            return

        docs_to_remove = total_docs - max_documents
        logging.info(f"Evicting {docs_to_remove} documents from cache using {strategy} strategy...")

        if strategy == 'FIFO':
            self._evict_by_timestamp(docs_to_remove, batch_size, sort_field="created_at")
        elif strategy == 'LRU':
            self._evict_by_timestamp(docs_to_remove, batch_size, sort_field="last_accessed")
        else:
            raise ValueError(f"Unknown eviction strategy: {strategy}")

        logging.info("Cache eviction completed.")

    def _evict_by_timestamp(self, docs_to_remove, batch_size, sort_field):
        """Evict cache entries by timestamp ordering.

        Uses zero-vector similarity search to retrieve candidates, then sorts
        by the timestamp field and deletes the oldest. Note: the zero-vector
        query returns entries nearest to the origin in embedding space, which
        is arbitrary — not all entries are guaranteed to be retrieved.
        For caches larger than the fetch size, eviction order is approximate.
        """
        index = self._get_index()

        while docs_to_remove > 0:
            # Fetch more candidates than needed so we can sort and pick the oldest
            fetch_size = min(docs_to_remove * 2, max(batch_size, docs_to_remove))
            results = index.similarity_search(
                query_vector=[0] * self.config.EMBEDDING_DIMENSION,
                columns=["id", sort_field],
                num_results=fetch_size,
            )

            if not results or results['result']['row_count'] == 0:
                break

            rows = results['result']['data_array']
            # Sort by timestamp ascending (oldest/least-recently-used first)
            # row[1] is the timestamp field, row[2] is the score (appended by VS)
            rows.sort(key=lambda r: r[1] if r[1] else "")

            # Take only as many as we need to remove
            to_delete = rows[:min(docs_to_remove, len(rows))]
            ids_to_remove = [row[0] for row in to_delete]
            index.delete(ids_to_remove)

            docs_to_remove -= len(ids_to_remove)
            logging.info(f"Removed {len(ids_to_remove)} documents from cache ({sort_field} ordering).")

    def get_indexed_row_count(self):
        index = self._get_index()
        description = index.describe()
        return description.get('status', {}).get('indexed_row_count', 0)

    def clear_cache(self):
        logging.info(f"Cleaning cache on endpoint {self.config.VECTOR_SEARCH_ENDPOINT_NAME_CACHE}...")
        if utils.index_exists(self.vsc, self.config.VECTOR_SEARCH_ENDPOINT_NAME_CACHE, self.config.VS_INDEX_FULLNAME_CACHE):
            try:
                self.vsc.delete_index(self.config.VECTOR_SEARCH_ENDPOINT_NAME_CACHE, self.config.VS_INDEX_FULLNAME_CACHE)
                self._index = None  # Invalidate cached handle
                logging.info(f"Cache index {self.config.VS_INDEX_FULLNAME_CACHE} deleted successfully")
            except Exception as e:
                logging.error(f"Error deleting cache index {self.config.VS_INDEX_FULLNAME_CACHE}: {str(e)}")
        else:
            logging.info(f"Cache index {self.config.VS_INDEX_FULLNAME_CACHE} does not exist")
