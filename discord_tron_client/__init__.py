__all__ = ["websocket_client"]


def websocket_client(*args, **kwargs):
    from .ws_client import websocket_client as _websocket_client
    return _websocket_client(*args, **kwargs)
