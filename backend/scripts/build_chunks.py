# from rag_podcast.indexing.chunker import build_chunks
import json
import numpy as np
from pathlib import Path
import ast

with open(Path(__file__).resolve().parent.parent.parent/"aligned_test.json", "r",
          encoding='utf-8') as f:
    data = json.load(f)[:2]

print(type(data))
# print(data[0])

print(f'INPUT:{len(data)} segments, total span {data[0].get('start')} -> {data[-1].get('end')}')
words = [w for ws in data for w in ws.get('words', []) if 'start' in w and 'end' in w]
# words = [w for w in data[2].get('words',[]) if 'start' in w and 'end' in w]
print(words)
print(len(words))
# chunks = build_chunks(data, min_duration=10.0, max_duration=20.0)


# print('OUTPUT: %d chunks' % len(chunks))
# for i, c in enumerate(chunks, 1):
#     print()
#     print(f'--- Chunk {i} ---')
#     print('text: ', repr(c.text))
#     print('start:', c.start, ' end:', c.end, ' duration:', c.end - c.start)
#     print('words:', [w['word'] for w in c.words])

