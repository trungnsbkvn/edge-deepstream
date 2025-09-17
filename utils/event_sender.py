import os
import socket
import struct
from typing import Optional


class EventSender:
    """Unix domain socket sender for face recognition events.

    Packet format (Big Endian / network order), matching provided C++:
    - totalLen: uint32 (length of the remaining payload, not including this field)
    - strLen:   uint32 (length of UTF-8 string)
    - str:      bytes   (event text)
    - imgSize:  uint32 (length of image bytes)
    - imgData:  bytes   (encoded image, e.g., PNG/JPEG)
    """

    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        self._sock: Optional[socket.socket] = None

    def _ensure_conn(self) -> None:
        if self._sock is not None:
            return
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(0.2)
        s.connect(self.socket_path)
        # Optional: make it non-blocking for send
        try:
            s.settimeout(None)
            s.setblocking(True)
        except Exception:
            pass
        self._sock = s

    def close(self) -> None:
        try:
            if self._sock is not None:
                self._sock.close()
        finally:
            self._sock = None

    def send(self, event_text: str, img_bytes: Optional[bytes] = None) -> bool:
        if not isinstance(event_text, str):
            event_text = str(event_text)
        text_b = event_text.encode('utf-8')
        img_b = img_bytes or b''
        strLen = len(text_b)
        imgSize = len(img_b)
        totalLen = 4 + strLen + 4 + imgSize  # len(strLen+str) + len(imgSize+img)
        # Build packet
        buf = bytearray(4 + totalLen)
        # Write totalLen
        struct.pack_into('!I', buf, 0, totalLen)
        # Write strLen
        struct.pack_into('!I', buf, 4, strLen)
        # Write str
        if strLen:
            buf[8:8+strLen] = text_b
        # Write imgSize
        struct.pack_into('!I', buf, 8+strLen, imgSize)
        # Write img
        if imgSize:
            buf[12+strLen:12+strLen+imgSize] = img_b
        # Send
        try:
            self._ensure_conn()
            assert self._sock is not None
            sent = self._sock.send(buf)
            return sent == len(buf)
        except Exception:
            # Reset connection on failure; caller can retry later
            self.close()
            return False
