# ingest.py
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from chromadb import PersistentClient
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

load_dotenv(override=True)

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
DB_NAME = str(Path(__file__).parent / "vector_db")
COLLECTION_NAME = "docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
KNOWLEDGE_BASE_PATH = Path(__file__).parent / "knowledge-base"
AVERAGE_CHUNK_SIZE = 100  # chars, used only for splitter sizing


class Result(BaseModel):
    page_content: str
    metadata: dict


def fetch_documents():
    documents = []
    for folder in KNOWLEDGE_BASE_PATH.iterdir():
        if not folder.is_dir():
            continue
        doc_type = folder.name
        for file in folder.rglob("*.md"):
            with open(file, "r", encoding="utf-8") as f:
                documents.append(
                    {
                        "type": doc_type,
                        "source": file.as_posix(),
                        "text": f.read(),
                    }
                )
    print(f"✅ loaded {len(documents)} documents")
    return documents


def create_chunks(documents):
    """Rule-based chunking - deterministic, no LLM calls."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=AVERAGE_CHUNK_SIZE * 3,  # ~300 chars
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: list[Result] = []
    for doc in tqdm(documents, desc="Chunking"):
        splits = splitter.split_text(doc["text"])
        for i, text in enumerate(splits):
            all_chunks.append(
                Result(
                    page_content=text.strip(),
                    metadata={
                        "source": doc["source"],
                        "type": doc["type"],
                        "chunk_id": i,
                    },
                )
            )
    print(f"✅ Created {len(all_chunks)} chunks from {len(documents)} docs")
    return all_chunks


def create_embeddings(chunks: list[Result]):
    if not chunks:
        print("❌ No chunks to embed!")
        return

    print(f"Processing {len(chunks)} chunks")

    # basic validation
    valid_chunks = [
        c for c in chunks if isinstance(c.page_content, str) and c.page_content.strip()
    ]
    if len(valid_chunks) != len(chunks):
        print(f"⚠️ Filtered invalid chunks: {len(chunks) - len(valid_chunks)} removed")
        chunks = valid_chunks

    chroma = PersistentClient(path=DB_NAME)

    # delete existing collection if present
    existing = [c.name for c in chroma.list_collections()]
    if COLLECTION_NAME in existing:
        print(f"🧹 Deleting existing collection '{COLLECTION_NAME}'")
        chroma.delete_collection(COLLECTION_NAME)

    hf_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [f"doc_{i}" for i in range(len(chunks))]

    print("🔧 Generating embeddings...")
    embeddings = hf_embeddings.embed_documents(texts)
    print(f"✅ Generated {len(embeddings)} embeddings")

    collection = chroma.create_collection(name=COLLECTION_NAME)
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    print(f"✅ Vectorstore created with {collection.count()} documents")


if __name__ == "__main__":
    print(f"DB path: {DB_NAME}")
    documents = fetch_documents()
    chunks = create_chunks(documents)
    print(f"Sample chunk: {chunks[0].page_content[:200]!r}")
    create_embeddings(chunks)
    print("🎉 Ingestion complete")
