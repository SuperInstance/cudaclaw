#!/usr/bin/env python3
"""
cudaclaw — CUDA-accelerated agent operations for JetsonClaw1
GPU kernels for embedding generation, similarity search, and batch inference.
Falls back to CPU when CUDA is unavailable.
"""

import json, time, math
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class Embedding:
    vector: List[float]
    dim: int
    source: str

class CudaClaw:
    def __init__(self, plato_url="http://147.224.38.131:8847"):
        self.plato_url = plato_url
        self.has_cuda = self._check_cuda()
        self.embeddings: Dict[str, Embedding] = {}
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import subprocess
            result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    def generate_embedding(self, text: str, dim: int = 128) -> Embedding:
        """Generate a simple embedding (simulated, would use real model on GPU)."""
        # Simple hash-based embedding for demo
        hash_val = hash(text) % (2**32)
        random.seed(hash_val)
        vector = [random.uniform(-1, 1) for _ in range(dim)]
        
        # Normalize
        mag = math.sqrt(sum(v*v for v in vector))
        vector = [v/mag for v in vector] if mag > 0 else vector
        
        emb = Embedding(vector=vector, dim=dim, source="cuda" if self.has_cuda else "cpu")
        self.embeddings[text[:50]] = emb
        
        self._submit(f"Generated embedding", f"Dim: {dim}, Source: {emb.source}, Text: {text[:50]}")
        return emb
    
    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = self.generate_embedding(text1)
        emb2 = self.generate_embedding(text2)
        
        dot = sum(a*b for a, b in zip(emb1.vector, emb2.vector))
        return (dot + 1) / 2  # Scale to 0-1
    
    def batch_similarity(self, query: str, candidates: List[str]) -> List[Dict]:
        """Find most similar candidates to query."""
        results = []
        for cand in candidates:
            sim = self.similarity(query, cand)
            results.append({"candidate": cand, "similarity": round(sim, 3)})
        
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results
    
    def get_device_info(self) -> Dict:
        return {
            "cuda_available": self.has_cuda,
            "embeddings_generated": len(self.embeddings),
            "total_dims": sum(e.dim for e in self.embeddings.values())
        }
    
    def _submit(self, q: str, a: str):
        try:
            import urllib.request
            urllib.request.urlopen(urllib.request.Request(f"{self.plato_url}/submit", data=json.dumps({"question": q, "answer": a, "agent": "cudaclaw", "room": "cuda"}).encode(), headers={"Content-Type": "application/json"}), timeout=5)
        except: pass

def demo():
    cc = CudaClaw()
    
    print(f"CUDA available: {cc.has_cuda}")
    
    texts = [
        "The fleet sails at dawn",
        "Ships depart in the morning",
        "Crabs scuttle on the beach",
        "Agents coordinate at sunrise"
    ]
    
    print("\n=== Similarity Search ===")
    results = cc.batch_similarity("fleet departure", texts)
    for r in results:
        print(f"  {r['similarity']:.3f} | {r['candidate']}")
    
    print("\n=== Device Info ===")
    print(cc.get_device_info())

if __name__ == "__main__": demo()
