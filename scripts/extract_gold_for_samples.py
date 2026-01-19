import pandas as pd

# P2S 입력 (문단)
p2s_input = pd.read_csv('datasets/paragraph/test_10.csv')
p2s_keys = set(zip(p2s_input['book_name'], p2s_input['문단식별자']))
print(f'P2S 입력 (book_name, 문단식별자) ({len(p2s_keys)}개)')

# S2P 입력 (문장)
s2p_input = pd.read_csv('datasets/sentence/test_10.csv')
s2p_keys = set(zip(s2p_input['book_name'], s2p_input['문장식별자']))
print(f'S2P 입력 (book_name, 문장식별자) ({len(s2p_keys)}개)')

# P2S 정답: sentence에서 해당 (book_name, 문단식별자) 추출
sentence_gold = pd.read_csv('datasets/sentence/test.csv')
p2s_gold = sentence_gold[sentence_gold.apply(lambda r: (r['book_name'], r['문단식별자']) in p2s_keys, axis=1)]
p2s_gold.to_csv('datasets/sentence/test_10_gold.csv', index=False)
print(f'P2S 정답 추출: {len(p2s_gold)}행')

# S2P 정답: phrase에서 해당 (book_name, 문장식별자) 추출
phrase_gold = pd.read_csv('datasets/phrase/test.csv')
s2p_gold = phrase_gold[phrase_gold.apply(lambda r: (r['book_name'], r['문장식별자']) in s2p_keys, axis=1)]
s2p_gold.to_csv('datasets/phrase/test_10_gold.csv', index=False)
print(f'S2P 정답 추출: {len(s2p_gold)}행')
