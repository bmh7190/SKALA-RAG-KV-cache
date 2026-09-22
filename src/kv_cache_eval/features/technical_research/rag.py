"""RAG 구현 위치.

TODO: 제공 문서 풀에서 고른 원문의 총 페이지 수가 200 이내인지 검증한다.
TODO: intfloat/multilingual-e5-small 검색 품질을 시험한 뒤, E5의 query:/passage:
접두어와 모델 token 기준 분할을 구현한다.
TODO: PDF 로딩, 검색, 출처/페이지 추적 및 관련성 확인을 구현한다.
모델 다운로드, 임베딩 실행, 벡터 인덱스 생성은 이 초기 구조에 포함하지 않는다.
"""
