 # This repo is Reachy Mini Conversation App

  Voice conversation app for Reachy Mini robot using OpenAI
  Realtime API.

  ## Key Architecture
  - **Two modes**: `--gradio` (web UI via fastrtc Stream) or
  headless (LocalStream in `console.py`)
  - **Audio path in headless**: `LocalStream.play_loop()` handles
  audio output to robot speakers
  - **Handler**: `openai_realtime.py` - OpenaiRealtimeHandler
  manages the OpenAI websocket connection

  ## Important Files
  - `main.py` - Entry point, mode selection at line ~133
  - `console.py` - LocalStream class, audio playback loop, pitch
  shifting
  - `openai_realtime.py` - OpenAI realtime API handler

  ## Running
  ```bash
  python -m reachy_mini_conversation_app.main          # headless
  mode
  python -m reachy_mini_conversation_app.main --gradio # web UI
  mode
