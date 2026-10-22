from rag_podcast.retrieval.PgvectorRecaller import PgvectorRecaller
from rag_podcast.retrieval.search import search
from rag_podcast.db import async_session, get_session
import asyncio
from sqlalchemy import select 
from rag_podcast.embedding.embedder import embed_batch
from pathlib import Path
import json

query = "How to operate a SASS business in tech sector"
async def main():
    async with async_session() as session:
        res = await search(query=query, session=session, 
             top_k=10, podcast_id=3, recaller=PgvectorRecaller())

    with open(Path(__file__).resolve().parent.parent.parent / "chunks.json", 'w') as f:
        for i in range(len(res)):
            chunk = res[i]
            print(f"------ Chunk {chunk.chunk_id} ------")
            print(f"Title of Episode: {chunk.episode_title}")
            print(f"Duration: {chunk.start} - {chunk.end}")
            print(f"Score: {chunk.effective_score}")
            print(f"Text: {chunk.text[:50]}")

            json.dump(chunk.__dict__, f, indent=3)

            

asyncio.run(main())
