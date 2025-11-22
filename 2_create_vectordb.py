#!/usr/bin/env python3
"""
STEP 2: 벡터 DB 생성
추출된 텍스트를 청킹하고 임베딩하여 Qdrant에 저장
"""

import json
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

class VectorDBBuilder:
    def __init__(self):
        self.embedder = SentenceTransformer(os.getenv('EMBEDDING_MODEL', 'BAAI/bge-m3'))
        self.qdrant = QdrantClient(path=os.getenv('QDRANT_PATH', './data/qdrant_db'))
        self.collection_name = "company_docs"
        
    def init_collection(self):
        """컬렉션 초기화"""
        # 기존 컬렉션 삭제
        try:
            self.qdrant.delete_collection(self.collection_name)
        except:
            pass
        
        # 새 컬렉션 생성
        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )
        print(f"✅ 컬렉션 생성 완료: {self.collection_name}")
    
    def chunk_text(self, text, chunk_size=200, overlap=50):
        """텍스트를 청크로 분할 (최적화: 200 words, 50 overlap = 25%)"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.strip()) > 50:  # 너무 짧은 청크 제외 (100→50으로 완화)
                chunks.append(chunk)

        return chunks
    
    def create_embeddings(self, extracted_json_path):
        """JSON에서 텍스트를 읽고 임베딩 생성"""
        print(f"\n📚 텍스트 로드: {extracted_json_path}")
        
        with open(extracted_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"총 {data['total_pages']}페이지")
        
        # 청킹
        all_chunks = []
        for page in data['pages']:
            chunks = self.chunk_text(page['text'])
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append({
                    "content": chunk,
                    "page": page['page_number'],
                    "chunk_idx": chunk_idx
                })
        
        print(f"생성된 청크 수: {len(all_chunks)}")
        
        # 임베딩 및 저장
        points = []
        print("\n🔄 임베딩 생성 중...")
        
        for idx, chunk_data in enumerate(tqdm(all_chunks)):
            # 임베딩
            vector = self.embedder.encode(chunk_data['content']).tolist()
            
            # 포인트 생성
            point = PointStruct(
                id=idx,
                vector=vector,
                payload={
                    "content": chunk_data['content'],
                    "page": chunk_data['page'],
                    "chunk_idx": chunk_data['chunk_idx'],
                    "source": "uploaded_report.pdf"
                }
            )
            points.append(point)
        
        # Qdrant에 저장
        print(f"\n💾 Qdrant에 저장 중...")
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"✅ 저장 완료: {len(points)}개 청크")
        
        return len(points)

def main():
    # 경로 설정
    extracted_json = Path('data/extracted_text.json')
    
    if not extracted_json.exists():
        print(f"❌ {extracted_json}을 찾을 수 없습니다.")
        print("먼저 1_parse_pdf.py를 실행하세요.")
        return
    
    # 벡터 DB 빌더 초기화
    builder = VectorDBBuilder()
    
    # 컬렉션 생성
    builder.init_collection()
    
    # 임베딩 생성 및 저장
    chunk_count = builder.create_embeddings(extracted_json)
    
    print(f"\n✅ 벡터 DB 생성 완료!")
    print(f"   - 총 청크 수: {chunk_count}")
    print(f"   - 컬렉션: {builder.collection_name}")

if __name__ == "__main__":
    main()
