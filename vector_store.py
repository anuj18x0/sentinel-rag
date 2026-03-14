import os
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

class VectorStore:
    def __init__(self):
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")
        
        # Initialize client with Cloud credentials
        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key
        )
        
        # Initialize embedding model
        self.model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        self.collection_name = "monitoring_insights"
        
        # Ensure collection exists
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=384,  # Size for all-MiniLM-L6-v2
                        distance=Distance.COSINE
                    )
                )
        except Exception as e:
            print(f"Error checking/creating collection: {e}")

    def upsert_insight(self, insight_id: str, text: str, metadata: dict):
        vector = self.model.encode(text).tolist()
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=insight_id,
                    vector=vector,
                    payload={"text": text, **metadata}
                )
            ]
        )

    def search(self, query: str, limit: int = 5):
        vector = self.model.encode(query).tolist()
        # Qdrant v1.17+ uses query_points instead of search
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit
        )
        return [{"payload": point.payload, "score": point.score} for point in results.points]

vector_store = VectorStore()
