"""
Log clustering using embeddings and MinHash signatures
"""
import re
import hashlib
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict, Counter
import logging
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class LogCluster:
    """Represents a cluster of similar log messages"""
    cluster_id: str
    exemplar: str  # Representative log message
    members: List[str]  # All log messages in cluster
    member_ids: List[str]  # IDs of log messages
    count: int
    first_seen: datetime
    last_seen: datetime
    embedding: np.ndarray
    minhash_signature: List[int]
    novelty_score: float
    pattern: str  # Extracted pattern


@dataclass
class ClusteringResult:
    """Result of log clustering operation"""
    clusters: List[LogCluster]
    unclustered_logs: List[str]
    total_logs: int
    clustered_logs: int
    novelty_detected: int
    processing_time: float


class LogPreprocessor:
    """Preprocess log messages for clustering"""

    def __init__(self):
        # Common patterns to remove/replace
        self.ip_pattern = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
        self.url_pattern = re.compile(r'https?://[^\s]+')
        self.uuid_pattern = re.compile(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b')
        self.timestamp_pattern = re.compile(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?')
        self.number_pattern = re.compile(r'\b\d+\b')
        self.hex_pattern = re.compile(r'\b0x[0-9a-f]+\b')

    def preprocess(self, log_message: str) -> str:
        """Preprocess a single log message"""
        # Convert to lowercase
        processed = log_message.lower()

        # Remove/replace common patterns
        processed = self.ip_pattern.sub('<IP>', processed)
        processed = self.url_pattern.sub('<URL>', processed)
        processed = self.uuid_pattern.sub('<UUID>', processed)
        processed = self.timestamp_pattern.sub('<TIMESTAMP>', processed)
        processed = self.hex_pattern.sub('<HEX>', processed)

        # Replace numbers (but keep some context)
        processed = self.number_pattern.sub('<NUM>', processed)

        # Remove extra whitespace
        processed = ' '.join(processed.split())

        return processed

    def extract_pattern(self, log_message: str) -> str:
        """Extract structural pattern from log message"""
        # Replace variable parts with placeholders
        pattern = log_message
        pattern = self.ip_pattern.sub('<IP>', pattern)
        pattern = self.url_pattern.sub('<URL>', pattern)
        pattern = self.uuid_pattern.sub('<UUID>', pattern)
        pattern = self.timestamp_pattern.sub('<TIMESTAMP>', pattern)
        pattern = self.number_pattern.sub('<NUM>', pattern)
        pattern = self.hex_pattern.sub('<HEX>', pattern)

        return pattern


class MinHashSignature:
    """Generate MinHash signatures for log messages"""

    def __init__(self, num_hashes: int = 128, shingle_size: int = 3):
        """
        Initialize MinHash generator

        Args:
            num_hashes: Number of hash functions to use
            shingle_size: Size of shingles for similarity
        """
        self.num_hashes = num_hashes
        self.shingle_size = shingle_size

        # Generate random hash functions
        np.random.seed(42)  # For reproducibility
        self.hash_functions = []
        for _ in range(num_hashes):
            a = np.random.randint(1, 2**32 - 1)
            b = np.random.randint(0, 2**32 - 1)
            self.hash_functions.append((a, b))

    def generate_signature(self, text: str) -> List[int]:
        """Generate MinHash signature for text"""
        # Create shingles
        shingles = self._create_shingles(text)

        if not shingles:
            return [0] * self.num_hashes

        # Generate signature
        signature = []
        for a, b in self.hash_functions:
            min_hash = float('inf')
            for shingle in shingles:
                # Hash the shingle
                shingle_hash = hashlib.md5(shingle.encode()).hexdigest()
                shingle_int = int(shingle_hash[:16], 16)  # Use first 16 hex chars

                # Apply hash function
                hash_value = (a * shingle_int + b) % (2**32)
                min_hash = min(min_hash, hash_value)

            signature.append(int(min_hash))

        return signature

    def _create_shingles(self, text: str) -> List[str]:
        """Create shingles from text"""
        words = text.split()
        if len(words) < self.shingle_size:
            return [' '.join(words)] if words else []

        shingles = []
        for i in range(len(words) - self.shingle_size + 1):
            shingle = ' '.join(words[i:i + self.shingle_size])
            shingles.append(shingle)

        return shingles

    def calculate_similarity(self, sig1: List[int], sig2: List[int]) -> float:
        """Calculate Jaccard similarity between two signatures"""
        if len(sig1) != len(sig2):
            return 0.0

        matches = sum(1 for h1, h2 in zip(sig1, sig2) if h1 == h2)
        return matches / len(sig1)


class EmbeddingClusterer:
    """Cluster logs using sentence embeddings"""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', similarity_threshold: float = 0.8):
        """
        Initialize embedding-based clusterer

        Args:
            model_name: Sentence transformer model to use
            similarity_threshold: Similarity threshold for clustering
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.model = None

    def _load_model(self):
        """Lazy load the sentence transformer model"""
        if self.model is None:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

    def encode_logs(self, log_messages: List[str]) -> np.ndarray:
        """Encode log messages to embeddings"""
        self._load_model()

        # Preprocess logs
        preprocessor = LogPreprocessor()
        processed_logs = [preprocessor.preprocess(log) for log in log_messages]

        # Generate embeddings
        embeddings = self.model.encode(processed_logs, show_progress_bar=False)

        return np.array(embeddings)

    def cluster_logs(self, log_messages: List[str], embeddings: Optional[np.ndarray] = None) -> Dict[int, List[int]]:
        """
        Cluster log messages based on embeddings

        Args:
            log_messages: List of log messages
            embeddings: Pre-computed embeddings (optional)

        Returns:
            Dictionary mapping cluster IDs to lists of message indices
        """
        if embeddings is None:
            embeddings = self.encode_logs(log_messages)

        # Use DBSCAN for clustering
        clustering = DBSCAN(
            eps=1 - self.similarity_threshold,  # Convert similarity to distance
            min_samples=2,
            metric='cosine'
        ).fit(embeddings)

        # Group by cluster
        clusters = defaultdict(list)
        for idx, cluster_id in enumerate(clustering.labels_):
            if cluster_id != -1:  # -1 indicates noise/outlier
                clusters[cluster_id].append(idx)

        return dict(clusters)


class LogClusteringEngine:
    """Main log clustering engine combining multiple techniques"""

    def __init__(self,
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 minhash_hashes: int = 128,
                 similarity_threshold: float = 0.8):
        """
        Initialize log clustering engine

        Args:
            embedding_model: Model for embedding generation
            minhash_hashes: Number of MinHash functions
            similarity_threshold: Similarity threshold for clustering
        """
        self.preprocessor = LogPreprocessor()
        self.minhash = MinHashSignature(minhash_hashes)
        self.embedder = EmbeddingClusterer(embedding_model, similarity_threshold)

        # Cluster storage
        self.existing_clusters: Dict[str, LogCluster] = {}
        self.cluster_counter = 0

    def cluster_logs(self, log_messages: List[str], log_ids: List[str] = None,
                    timestamps: List[datetime] = None) -> ClusteringResult:
        """
        Cluster log messages using hybrid approach

        Args:
            log_messages: List of log messages to cluster
            log_ids: Optional IDs for the log messages
            timestamps: Optional timestamps for the messages

        Returns:
            ClusteringResult with clustered and unclustered logs
        """
        import time
        start_time = time.time()

        if log_ids is None:
            log_ids = [f"log_{i}" for i in range(len(log_messages))]

        if timestamps is None:
            timestamps = [datetime.now()] * len(log_messages)

        # Step 1: Preprocess all logs
        processed_logs = [self.preprocessor.preprocess(msg) for msg in log_messages]

        # Step 2: Generate MinHash signatures for fast similarity
        signatures = [self.minhash.generate_signature(log) for log in processed_logs]

        # Step 3: Find exact duplicates first
        duplicate_groups = self._find_duplicates(processed_logs, log_ids, timestamps)

        # Step 4: Cluster remaining logs using embeddings
        unclustered_indices = []
        unclustered_logs = []
        unclustered_ids = []
        unclustered_timestamps = []

        for i, (log, log_id, timestamp) in enumerate(zip(processed_logs, log_ids, timestamps)):
            if not any(i in group_indices for group_indices in duplicate_groups.values()):
                unclustered_indices.append(i)
                unclustered_logs.append(log)
                unclustered_ids.append(log_id)
                unclustered_timestamps.append(timestamp)

        # Generate embeddings for unclustered logs
        if unclustered_logs:
            embeddings = self.embedder.encode_logs(unclustered_logs)
            embedding_clusters = self.embedder.cluster_logs(unclustered_logs, embeddings)
        else:
            embedding_clusters = {}

        # Step 5: Create final clusters
        final_clusters = []

        # Add duplicate clusters
        for duplicate_indices in duplicate_groups.values():
            cluster_logs = [log_messages[i] for i in duplicate_indices]
            cluster_ids = [log_ids[i] for i in duplicate_indices]
            cluster_timestamps = [timestamps[i] for i in duplicate_indices]

            cluster = self._create_cluster(
                cluster_logs, cluster_ids, cluster_timestamps,
                signatures[duplicate_indices[0]]
            )
            final_clusters.append(cluster)

        # Add embedding-based clusters
        for cluster_indices in embedding_clusters.values():
            original_indices = [unclustered_indices[i] for i in cluster_indices]
            cluster_logs = [log_messages[i] for i in original_indices]
            cluster_ids = [log_ids[i] for i in original_indices]
            cluster_timestamps = [timestamps[i] for i in original_indices]

            # Use signature from first log in cluster
            cluster_signature = signatures[original_indices[0]]

            cluster = self._create_cluster(
                cluster_logs, cluster_ids, cluster_timestamps, cluster_signature
            )
            final_clusters.append(cluster)

        # Identify unclustered logs
        clustered_indices = set()
        for cluster in final_clusters:
            for log_id in cluster.member_ids:
                if log_id in log_ids:
                    clustered_indices.add(log_ids.index(log_id))

        unclustered_messages = [
            log_messages[i] for i in range(len(log_messages))
            if i not in clustered_indices
        ]

        processing_time = time.time() - start_time

        return ClusteringResult(
            clusters=final_clusters,
            unclustered_logs=unclustered_messages,
            total_logs=len(log_messages),
            clustered_logs=len(log_messages) - len(unclustered_messages),
            novelty_detected=sum(1 for c in final_clusters if c.novelty_score > 0.8),
            processing_time=processing_time
        )

    def _find_duplicates(self, processed_logs: List[str], log_ids: List[str],
                        timestamps: List[datetime]) -> Dict[str, List[int]]:
        """Find exact duplicate log messages"""
        log_to_indices = defaultdict(list)

        for i, log in enumerate(processed_logs):
            log_to_indices[log].append(i)

        # Only keep groups with multiple occurrences
        duplicate_groups = {
            log: indices for log, indices in log_to_indices.items()
            if len(indices) > 1
        }

        return duplicate_groups

    def _create_cluster(self, log_messages: List[str], log_ids: List[str],
                       timestamps: List[datetime], signature: List[int]) -> LogCluster:
        """Create a LogCluster from a group of similar logs"""
        # Use first message as exemplar
        exemplar = log_messages[0]

        # Extract pattern
        pattern = self.preprocessor.extract_pattern(exemplar)

        # Generate cluster ID
        cluster_id = f"cluster_{self.cluster_counter}"
        self.cluster_counter += 1

        # Calculate novelty score (simplified - compare to existing clusters)
        novelty_score = self._calculate_novelty_score(signature)

        # Create embedding for the exemplar
        embeddings = self.embedder.encode_logs([exemplar])
        embedding = embeddings[0] if len(embeddings) > 0 else np.array([])

        cluster = LogCluster(
            cluster_id=cluster_id,
            exemplar=exemplar,
            members=log_messages,
            member_ids=log_ids,
            count=len(log_messages),
            first_seen=min(timestamps),
            last_seen=max(timestamps),
            embedding=embedding,
            minhash_signature=signature,
            novelty_score=novelty_score,
            pattern=pattern
        )

        # Store cluster for future novelty detection
        self.existing_clusters[cluster_id] = cluster

        return cluster

    def _calculate_novelty_score(self, signature: List[int]) -> float:
        """Calculate novelty score compared to existing clusters"""
        if not self.existing_clusters:
            return 1.0  # Completely novel

        max_similarity = 0.0
        for existing_cluster in self.existing_clusters.values():
            similarity = self.minhash.calculate_similarity(signature, existing_cluster.minhash_signature)
            max_similarity = max(max_similarity, similarity)

        # Novelty is inverse of similarity
        return 1.0 - max_similarity

    def find_similar_clusters(self, log_message: str, threshold: float = 0.8) -> List[Tuple[LogCluster, float]]:
        """Find clusters similar to a log message"""
        processed = self.preprocessor.preprocess(log_message)
        signature = self.minhash.generate_signature(processed)

        similar_clusters = []
        for cluster in self.existing_clusters.values():
            similarity = self.minhash.calculate_similarity(signature, cluster.minhash_signature)
            if similarity >= threshold:
                similar_clusters.append((cluster, similarity))

        # Sort by similarity (highest first)
        similar_clusters.sort(key=lambda x: x[1], reverse=True)

        return similar_clusters

    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get statistics about current clusters"""
        if not self.existing_clusters:
            return {"total_clusters": 0}

        total_logs = sum(cluster.count for cluster in self.existing_clusters.values())
        avg_cluster_size = total_logs / len(self.existing_clusters)

        novelty_scores = [cluster.novelty_score for cluster in self.existing_clusters.values()]

        return {
            "total_clusters": len(self.existing_clusters),
            "total_logs": total_logs,
            "avg_cluster_size": avg_cluster_size,
            "max_cluster_size": max(cluster.count for cluster in self.existing_clusters.values()),
            "novelty_score_avg": np.mean(novelty_scores),
            "novelty_score_max": max(novelty_scores),
            "high_novelty_clusters": sum(1 for score in novelty_scores if score > 0.8)
        }
