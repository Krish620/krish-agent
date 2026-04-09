"""
Krish's AI Phone Agent - Main Backend
Stack: Twilio (calls) + Deepgram (STT) + Claude (brain) + ElevenLabs (TTS)
"""

import os
import json
import asyncio
import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
from twilio.rest import Client as TwilioClient
from anthropic import Anthropic
import base64
from datetime import datetime

app = FastAPI()

# ── Clients ──────────────────────────────────────────────────────────────────
anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
twilio_client    = TwilioClient(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])

ELEVENLABS_API_KEY = os.environ["ELEVENLABS_API_KEY"]
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")  # default: Bella

DEEPGRAM_API_KEY = os.environ["DEEPGRAM_API_KEY"]

TWILIO_FROM_NUMBER = os.environ["TWILIO_FROM_NUMBER"]   # your Twilio number
KRISH_PHONE_NUMBER = os.environ["KRISH_PHONE_NUMBER"]   # your real phone for SMS

GOOGLE_CALENDAR_WEBHOOK = os.environ.get("GOOGLE_CALENDAR_WEBHOOK", "")  # optional

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a professional AI assistant answering calls on behalf of Krish.

Your job:
1. Greet the caller politely and ask who they are and the purpose of their call.
2. Answer any general questions about Krish professionally (he is unavailable right now).
3. If they want to leave a message, collect: their name, phone number, and message.
4. If they want to schedule a meeting, collect: their name, phone number, preferred date/time, and topic.
5. End every call politely, thanking the caller.

Rules:
- Be concise — this is a phone call, keep responses under 3 sentences.
- Never reveal personal information about Krish beyond what's needed.
- Always stay professional and friendly.
- When you have collected all info for a message or booking, include a JSON block at the END of your response (hidden from speech) in this format:

For messages:   [ACTION:message] {"name":"...","phone":"...","message":"..."}
For bookings:   [ACTION:booking] {"name":"...","phone":"...","datetime":"...","topic":"..."}
For ending:     [ACTION:end]

Speak naturally — no bullet points, no lists, just conversational sentences."""

# ── Active call sessions ──────────────────────────────────────────────────────
sessions: dict[str, dict] = {}


# ── 1. Twilio webhook: incoming call ─────────────────────────────────────────
@app.post("/incoming-call")
async def incoming_call(request: Request):
    """Twilio calls this when someone calls your Twilio number."""
    form = await request.form()
    call_sid = form.get("CallSid", "unknown")

    # Start a new session
    sessions[call_sid] = {
        "history": [],
        "caller_number": form.get("From", "Unknown"),
        "started_at": datetime.now().isoformat(),
    }

    # Tell Twilio to open a media stream (audio) to our WebSocket
    host = request.headers.get("host")
    response = VoiceResponse()
    connect = Connect()
    connect.stream(url=f"wss://{host}/media-stream/{call_sid}")
    response.append(connect)

    return Response(content=str(response), media_type="application/xml")


# ── 2. WebSocket: real-time audio stream ──────────────────────────────────────
@app.websocket("/media-stream/{call_sid}")
async def media_stream(websocket: WebSocket, call_sid: str):
    """Handles the real-time audio stream from Twilio via Deepgram."""
    await websocket.accept()
    session = sessions.get(call_sid, {"history": [], "caller_number": "Unknown"})

    # Greet the caller immediately
    greeting = "Hello! You've reached Krish's assistant. Krish is unavailable right now. May I ask who's calling and what's the purpose of your call?"
    await speak_and_send(websocket, greeting, call_sid)
    session["history"].append({"role": "assistant", "content": greeting})

    # Connect to Deepgram for real-time STT
    async with httpx.AsyncClient() as http:
        try:
            async with http.stream(
                "GET",
                "wss://api.deepgram.com/v1/listen?encoding=mulaw&sample_rate=8000&channels=1&model=nova-2&interim_results=false",
                headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
            ) as deepgram_ws:
                async def forward_audio():
                    """Forward audio from Twilio → Deepgram."""
                    async for message in websocket.iter_text():
                        data = json.loads(message)
                        if data.get("event") == "media":
                            audio_chunk = base64.b64decode(data["media"]["payload"])
                            await deepgram_ws.send(audio_chunk)
                        elif data.get("event") == "stop":
                            break

                async def receive_transcripts():
                    """Receive transcripts from Deepgram → feed to AI → speak response."""
                    async for chunk in deepgram_ws.aiter_text():
                        result = json.loads(chunk)
                        transcript = (
                            result.get("channel", {})
                                  .get("alternatives", [{}])[0]
                                  .get("transcript", "")
                            if result.get("is_final") else ""
                        )
                        if not transcript.strip():
                            continue

                        print(f"[STT] Caller said: {transcript}")
                        session["history"].append({"role": "user", "content": transcript})

                        # Get AI response
                        ai_reply, action = await get_ai_response(session["history"])
                        session["history"].append({"role": "assistant", "content": ai_reply})

                        # Speak it
                        await speak_and_send(websocket, ai_reply, call_sid)

                        # Handle action
                        if action:
                            await handle_action(action, session, call_sid)
                            if action.get("type") == "end":
                                await websocket.close()
                                return

                await asyncio.gather(forward_audio(), receive_transcripts())

        except WebSocketDisconnect:
            print(f"[INFO] Call {call_sid} ended.")
        except Exception as e:
            print(f"[ERROR] {e}")


# ── 3. AI brain ───────────────────────────────────────────────────────────────
async def get_ai_response(history: list) -> tuple[str, dict | None]:
    """Send conversation history to Claude, return (spoken_reply, action_dict)."""
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=history,
    )
    full_text = response.content[0].text

    # Parse hidden action tag
    action = None
    spoken = full_text

    for action_type in ["message", "booking", "end"]:
        tag = f"[ACTION:{action_type}]"
        if tag in full_text:
            parts = full_text.split(tag)
            spoken = parts[0].strip()
            if action_type != "end":
                try:
                    payload = json.loads(parts[1].strip())
                    action = {"type": action_type, **payload}
                except Exception:
                    action = {"type": action_type}
            else:
                action = {"type": "end"}
            break

    print(f"[AI] Reply: {spoken}")
    return spoken, action


# ── 4. Text-to-speech via ElevenLabs ─────────────────────────────────────────
async def text_to_speech(text: str) -> bytes:
    """Convert text to audio bytes using ElevenLabs."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
            headers={
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "text": text,
                "model_id": "eleven_turbo_v2",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
                "output_format": "ulaw_8000",  # Twilio-compatible format
            },
            timeout=15,
        )
        return response.content


async def speak_and_send(websocket: WebSocket, text: str, call_sid: str):
    """Convert text to speech and send audio back through Twilio stream."""
    audio_bytes = await text_to_speech(text)
    audio_b64 = base64.b64encode(audio_bytes).decode()

    # Send audio back to Twilio
    await websocket.send_json({
        "event": "media",
        "streamSid": call_sid,
        "media": {"payload": audio_b64},
    })


# ── 5. Handle post-call actions ───────────────────────────────────────────────
async def handle_action(action: dict, session: dict, call_sid: str):
    """Send SMS to Krish and/or book calendar based on action."""
    action_type = action.get("type")
    caller = session.get("caller_number", "Unknown")

    if action_type == "message":
        sms_body = (
            f"📞 New message from {action.get('name', caller)}\n"
            f"Phone: {action.get('phone', caller)}\n"
            f"Message: {action.get('message', 'No message')}\n"
            f"Time: {session.get('started_at', '')}"
        )
        send_sms(sms_body)

    elif action_type == "booking":
        sms_body = (
            f"📅 Meeting request from {action.get('name', caller)}\n"
            f"Phone: {action.get('phone', caller)}\n"
            f"When: {action.get('datetime', 'Not specified')}\n"
            f"Topic: {action.get('topic', 'Not specified')}"
        )
        send_sms(sms_body)

        # Optionally trigger Google Calendar webhook
        if GOOGLE_CALENDAR_WEBHOOK:
            async with httpx.AsyncClient() as client:
                await client.post(GOOGLE_CALENDAR_WEBHOOK, json=action, timeout=10)

    elif action_type == "end":
        # Send a full call summary
        history_text = "\n".join(
            f"{'Caller' if m['role']=='user' else 'Agent'}: {m['content']}"
            for m in session.get("history", [])
        )
        sms_body = f"📋 Call summary ({caller}):\n{history_text[:800]}"
        send_sms(sms_body)


def send_sms(body: str):
    """Send an SMS to Krish's real phone number."""
    try:
        twilio_client.messages.create(
            body=body,
            from_=TWILIO_FROM_NUMBER,
            to=KRISH_PHONE_NUMBER,
        )
        print(f"[SMS] Sent to Krish: {body[:80]}...")
    except Exception as e:
        print(f"[SMS ERROR] {e}")


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "Krish's AI agent is running 🤖"}