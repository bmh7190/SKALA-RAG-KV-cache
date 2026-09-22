"""RAG 구현 위치.

TODO: 제공 문서 풀에서 고른 원문의 총 페이지 수가 200 이내인지 검증한다.
TODO: 사용자 지정 BAAI/bge-m3의 검색 품질을 시험한다. sentence-transformers를
사용하는 HuggingFaceEmbeddings + 로컬 FAISS의 dense 벡터 경로를 먼저 구현한다.
모델 카드에 따르면 BGE-M3 검색 질의에는 별도 instruction 접두어가 필요하지 않다.
TODO: 모델 token 기준 분할, PDF 로딩, 출처·페이지 추적, 관련성 확인을 구현한다.
이 경로는 BGE-M3의 sparse/multi-vector 검색을 자동으로 사용하지 않는다.
모델 다운로드, 임베딩 실행, 벡터 인덱스 생성은 이 초기 구조에 포함하지 않는다.
"""
