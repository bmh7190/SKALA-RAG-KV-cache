"""Offline tests must never call external services, even if .env exists."""
import socket

def blocked(*args, **kwargs):
    raise RuntimeError('Network disabled by offline test guard')

socket.socket.connect = blocked
socket.socket.connect_ex = blocked
socket.create_connection = blocked
