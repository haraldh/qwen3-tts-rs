#!/usr/bin/env python3
"""Verify TTS audio quality by transcribing via Wyoming STT and comparing to input text.

Usage:
    python3 scripts/verify_audio.py /tmp/test.wav "Hello, this is a test of the text to speech system."
    python3 scripts/verify_audio.py --host localhost --port 10300 /tmp/test.wav "expected text"

Exit code 0 if transcription is sufficiently similar, 1 otherwise.
"""

import argparse
import json
import socket
import sys
import wave


def clip_wav(path: str, max_seconds: float) -> bytes:
    """Read a WAV file clipped to max_seconds. Returns (frames, rate, width, channels)."""
    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        width = w.getsampwidth()
        channels = w.getnchannels()
        max_frames = int(max_seconds * rate)
        n = min(w.getnframes(), max_frames)
        frames = w.readframes(n)
    return frames, rate, width, channels


def estimate_duration(text: str, chars_per_second: float = 14.0, margin: float = 1.5) -> float:
    """Estimate expected audio duration from text length.

    Default ~14 chars/s is typical for TTS speech rate. Margin multiplier
    accounts for pauses and slower speech.
    """
    return max(5.0, len(text) / chars_per_second * margin)


def transcribe_wav(
    path: str, host: str = "localhost", port: int = 10300, max_seconds: float = 0
) -> str:
    frames, rate, width, channels = (
        clip_wav(path, max_seconds) if max_seconds > 0 else _read_wav(path)
    )

    timeout = max(30, int(len(frames) / rate / width * 0.5) + 30)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    sock.settimeout(timeout)

    def send_event(etype, data=None, payload=b""):
        data_bytes = json.dumps(data).encode() if data else b""
        evt = {
            "type": etype,
            "data_length": len(data_bytes),
            "payload_length": len(payload),
        }
        sock.sendall(json.dumps(evt).encode() + b"\n")
        if data_bytes:
            sock.sendall(data_bytes)
        if payload:
            sock.sendall(payload)

    def recv_events():
        buf = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                evt = json.loads(line)
                data_len = evt.get("data_length", 0)
                payload_len = evt.get("payload_length", 0)
                total = data_len + payload_len
                while len(buf) < total:
                    buf += sock.recv(4096)
                data = json.loads(buf[:data_len]) if data_len else None
                buf = buf[total:]
                yield evt["type"], data

    send_event("transcribe", {"language": "en"})
    send_event("audio-start", {"rate": rate, "width": width, "channels": channels})
    chunk_size = 16000
    for i in range(0, len(frames), chunk_size):
        send_event("audio-chunk", payload=frames[i : i + chunk_size])
    send_event("audio-stop")

    for etype, data in recv_events():
        if etype == "transcript":
            sock.close()
            return data.get("text", "")
        elif etype == "error":
            sock.close()
            raise RuntimeError(f"STT error: {data}")

    sock.close()
    raise RuntimeError("No transcript received")


def _read_wav(path: str):
    """Read full WAV file."""
    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        width = w.getsampwidth()
        channels = w.getnchannels()
        frames = w.readframes(w.getnframes())
    return frames, rate, width, channels


def normalize(text: str) -> str:
    """Normalize text for comparison: lowercase, strip punctuation, collapse whitespace."""
    import re

    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_overlap(a: str, b: str) -> float:
    """Fraction of words in `a` that appear in `b`."""
    words_a = set(normalize(a).split())
    words_b = set(normalize(b).split())
    if not words_a:
        return 1.0 if not words_b else 0.0
    return len(words_a & words_b) / len(words_a)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", help="Path to WAV file to transcribe")
    parser.add_argument("expected", help="Expected text content")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=10300)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Minimum word overlap ratio (default: 0.7)",
    )
    args = parser.parse_args()

    # Clip audio to expected duration to avoid STT timeouts on long/degenerate outputs
    max_seconds = estimate_duration(args.expected)
    transcript = transcribe_wav(args.wav, args.host, args.port, max_seconds=max_seconds)
    overlap = word_overlap(args.expected, transcript)

    print(f"Expected:    {args.expected}")
    print(f"Transcribed: {transcript}")
    print(f"Overlap:     {overlap:.0%}")

    if overlap >= args.threshold:
        print("PASS")
        sys.exit(0)
    else:
        print(f"FAIL (below {args.threshold:.0%} threshold)")
        sys.exit(1)


if __name__ == "__main__":
    main()
