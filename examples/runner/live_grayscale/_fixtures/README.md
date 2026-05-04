# Test fixtures

`0` — 51 KB synthetic MPEG-TS segment used by `test.sh`'s real-data smoke
(served by `_smoke_server.py`). 2 seconds, 320x240, H.264 yuv420p, video only.

Synthetic content from ffmpeg's `testsrc` lavfi generator — content
doesn't matter, only that it's valid MPEG-TS the runner can decode.

## Regenerate

```bash
ffmpeg -y \
    -f lavfi -i "testsrc=size=320x240:rate=30:duration=2" \
    -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p \
    -f mpegts _fixtures/0
```
