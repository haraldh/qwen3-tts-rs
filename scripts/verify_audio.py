#!/usr/bin/env python3
"""Verify TTS audio quality by transcribing via whisper.cpp STT and comparing to input text.

Talks to whisper.cpp's OpenAI-compatible server (`/v1/audio/transcriptions`), which
runs on halo:8771 backed by large-v3 on the Strix Halo iGPU. Standard library only —
no openai, requests, or whisper package needed.

Usage:
    python3 scripts/verify_audio.py /var/tmp/test.wav "Hello, this is a test of the text to speech system."
    python3 scripts/verify_audio.py --url http://localhost:8771/v1/audio/transcriptions out.wav "expected text"

Override the default endpoint with the WHISPER_URL environment variable.

Exit code 0 if transcription is sufficiently similar, 1 otherwise.
"""

import argparse
import io
import json
import os
import re
import sys
import urllib.error
import urllib.request
import uuid
import wave

DEFAULT_URL = os.environ.get("WHISPER_URL", "http://halo:8771/v1/audio/transcriptions")


def estimate_duration(text: str, chars_per_second: float = 14.0, margin: float = 1.5) -> float:
    """Estimate expected audio duration from text length.

    Default ~14 chars/s is typical for TTS speech rate. Margin multiplier
    accounts for pauses and slower speech.
    """
    return max(5.0, len(text) / chars_per_second * margin)


def read_wav_clipped(path: str, max_seconds: float) -> bytes:
    """Return the WAV file as bytes, clipped to max_seconds.

    Clipping guards against degenerate generations that loop for minutes — the
    transcription only needs to cover the expected text.
    """
    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        width = w.getsampwidth()
        channels = w.getnchannels()
        total = w.getnframes()
        n = min(total, int(max_seconds * rate)) if max_seconds > 0 else total
        frames = w.readframes(n)

    if n == total:
        with open(path, "rb") as f:
            return f.read()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(width)
        out.setframerate(rate)
        out.writeframes(frames)
    return buf.getvalue()


def transcribe(wav_bytes: str, url: str, language: str, timeout: float) -> str:
    """POST the audio as multipart/form-data and return the transcript text."""
    boundary = uuid.uuid4().hex
    fields = {"language": language, "response_format": "json"}

    parts = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
        f"Content-Type: audio/wav\r\n\r\n".encode()
    )
    parts.append(wav_bytes)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    body = b"".join(parts)

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.URLError as e:
        raise RuntimeError(f"STT request to {url} failed: {e}") from e

    text = payload.get("text")
    if text is None:
        raise RuntimeError(f"No 'text' field in STT response: {payload}")
    return text.strip()


def normalize(text: str) -> str:
    """Normalize text for comparison: lowercase, strip punctuation, collapse whitespace."""
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
    parser.add_argument("--url", default=DEFAULT_URL, help=f"STT endpoint (default: {DEFAULT_URL})")
    parser.add_argument("--language", default="en", help="Language hint for STT (default: en)")
    parser.add_argument("--timeout", type=float, default=120.0, help="Request timeout in seconds")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Minimum word overlap ratio (default: 0.7)",
    )
    args = parser.parse_args()

    # Clip audio to expected duration so degenerate output can't stall transcription
    wav_bytes = read_wav_clipped(args.wav, estimate_duration(args.expected))
    transcript = transcribe(wav_bytes, args.url, args.language, args.timeout)
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
