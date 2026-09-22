"""PYTHONPATH에 이 디렉터리를 넣은 테스트 실행에서 모든 소켓 연결을 막는다."""

import socket


def _blocked(*args, **kwargs):
    raise RuntimeError("오프라인 테스트에서 네트워크 연결은 금지됩니다")


socket.socket.connect = _blocked
socket.socket.connect_ex = _blocked
socket.create_connection = _blocked
