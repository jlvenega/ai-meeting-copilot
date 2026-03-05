# Meeting Copilot

Real-time meeting transcription, translation, and summarization powered by OpenAI.

Captures system audio (via BlackHole), transcribes speech with OpenAI's Realtime API, translates each turn, and generates a structured meeting summary on demand.

## Features

- **Live transcription** using `gpt-4o-mini-transcribe` with server-side VAD
- **Per-turn translation** into your target language
- **Meeting summary** with sections: Summary, Decisions, Action items, Risks, Open questions
- **Meeting context** field to steer translation and summary quality
- **Multi-session safe**: each browser tab gets isolated queues and state

## Requirements

- macOS with [BlackHole](https://existential.audio/blackhole/) virtual audio driver
- Python 3.10+
- OpenAI API key with access to Realtime API

## Setup

```bash
# Clone the repo
git clone git@github.com:jlvenega/ai-meeting-copilot.git
cd ai-meeting-copilot

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure your API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Running

```bash
uvicorn app:app --reload
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

## Audio Setup

1. Install [BlackHole 2ch](https://existential.audio/blackhole/).
2. In **Audio MIDI Setup**, create a **Multi-Output Device** that includes both your speakers and BlackHole 2ch.
3. Set that Multi-Output Device as your system output so meeting audio is routed through BlackHole.
4. The app automatically picks the BlackHole device as its input.

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI API key |

## License

MIT — see [LICENSE](LICENSE).
