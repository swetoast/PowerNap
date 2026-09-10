from __future__ import annotations

import logging
import os
import socket


def sd_notify(message: str) -> bool:
    address = os.getenv("NOTIFY_SOCKET")
    if not address:
        return False
    if address.startswith("@"):
        address = "\0" + address[1:]
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        sock.connect(address)
        sock.sendall(message.encode("utf-8"))
        return True
    except OSError as exc:
        logging.debug("systemd notification failed: %s", exc)
        return False
    finally:
        sock.close()
