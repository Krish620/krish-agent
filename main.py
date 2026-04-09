"""
Krish's AI Phone Agent - Main Backend
Stack: Twilio (calls) + Deepgram (STT) + Groq FREE (brain) + ElevenLabs (TTS)
"""

import os
import json
import asyncio
import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Connect
from twilio.rest import Client as TwilioClient
from groq import Groq
import base64
from datetime import datetime
from pydantic import BaseModel

app = FastAPI()

# ── Clients ──────────────────────────────────────────────────────────────────
groq_client   = Groq(api_key=os.environ["GROQ_API_KEY"])
twilio_client = TwilioClient(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])

ELEVENLABS_API_KEY  = os.environ["ELEVENLABS_API_KEY"]
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
DEEPGRAM_API_KEY    = os.environ["DEEPGRAM_API_KEY"]
TWILIO_FROM_NUMBER  = os.environ["TWILIO_FROM_NUMBER"]
KRISH_PHONE_NUMBER  = os.environ["KRISH_PHONE_NUMBER"]
GOOGLE_CALENDAR_WEBHOOK = os.environ.get("GOOGLE_CALENDAR_WEBHOOK", "")

# ── In-memory store ───────────────────────────────────────────────────────────
call_logs: list[dict] = []
agent_settings: dict = {
    "name": "Krish",
    "greeting": "Hello! You've reached Krish's assistant. How can I help you?",
    "about": "Krish is a developer and entrepreneur.",
    "sms_enabled": True,
    "calendar_enabled": True,
}
sessions: dict[str, dict] = {}

# ── System prompt ─────────────────────────────────────────────────────────────
def get_system_prompt():
    return f"""You are a professional AI assistant answering calls on behalf of {agent_settings['name']}.

About {agent_settings['name']}: {agent_settings['about']}

Your job:
1. Greet the caller politely and ask who they are and the purpose of their call.
2. Answer any general questions about {agent_settings['name']} professionally.
3. If they want to leave a message, collect: their name, phone number, and message.
4. If they want to schedule a meeting, collect: their name, phone number, preferred date/time, and topic.
5. End every call politely, thanking the caller.

Rules:
- Be concise — this is a phone call, keep responses under 3 sentences.
- Always stay professional and friendly.
- When you have collected all info, include a JSON block at the END of your response in this format:

For messages:   [ACTION:message] {{"name":"...","phone":"...","message":"..."}}
For bookings:   [ACTION:booking] {{"name":"...","phone":"...","datetime":"...","topic":"..."}}
For ending:     [ACTION:end]

Speak naturally — no bullet points, just conversational sentences."""


# ── 1. Twilio webhook: incoming call ─────────────────────────────────────────
@app.post("/incoming-call")
async def incoming_call(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid", "unknown")

    sessions[call_sid] = {
        "history": [],
        "caller_number": form.get("From", "Unknown"),
        "started_at": datetime.now().isoformat(),
    }

    host = request.headers.get("host")
    response = VoiceResponse()
    connect = Connect()
    connect.stream(url=f"wss://{host}/media-stream/{call_sid}")
    response.append(connect)

    return Response(content=str(response), media_type="application/xml")


# ── 2. WebSocket: real-time audio stream ──────────────────────────────────────
@app.websocket("/media-stream/{call_sid}")
async def media_stream(websocket: WebSocket, call_sid: str):
    await websocket.accept()
    session = sessions.get(call_sid, {"history": [], "caller_number": "Unknown", "started_at": datetime.now().isoformat()})

    greeting = agent_settings["greeting"]
    await speak_and_send(websocket, greeting, call_sid)
    session["history"].append({"role": "assistant", "content": greeting})

    async with httpx.AsyncClient() as http:
        try:
            async with http.stream(
                "GET",
                "wss://api.deepgram.com/v1/listen?encoding=mulaw&sample_rate=8000&channels=1&model=nova-2&interim_results=false",
                headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
            ) as deepgram_ws:

                async def forward_audio():
                    async for message in websocket.iter_text():
                        data = json.loads(message)
                        if data.get("event") == "media":
                            audio_chunk = base64.b64decode(data["media"]["payload"])
                            await deepgram_ws.send(audio_chunk)
                        elif data.get("event") == "stop":
                            break

                async def receive_transcripts():
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

                        ai_reply, action = await get_ai_response(session["history"])
                        session["history"].append({"role": "assistant", "content": ai_reply})

                        await speak_and_send(websocket, ai_reply, call_sid)

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


# ── 3. AI brain (Groq — FREE) ─────────────────────────────────────────────────
async def get_ai_response(history: list) -> tuple[str, dict | None]:
    messages = [{"role": "system", "content": get_system_prompt()}] + history

    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",   # free and fast!
        messages=messages,
        max_tokens=300,
    )
    full_text = response.choices[0].message.content

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
                "output_format": "ulaw_8000",
            },
            timeout=15,
        )
        return response.content


async def speak_and_send(websocket: WebSocket, text: str, call_sid: str):
    audio_bytes = await text_to_speech(text)
    audio_b64 = base64.b64encode(audio_bytes).decode()
    await websocket.send_json({
        "event": "media",
        "streamSid": call_sid,
        "media": {"payload": audio_b64},
    })


# ── 5. Handle post-call actions ───────────────────────────────────────────────
async def handle_action(action: dict, session: dict, call_sid: str):
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

        if GOOGLE_CALENDAR_WEBHOOK:
            async with httpx.AsyncClient() as client:
                await client.post(GOOGLE_CALENDAR_WEBHOOK, json=action, timeout=10)

    elif action_type == "end":
        history_text = "\n".join(
            f"{'Caller' if m['role']=='user' else 'Agent'}: {m['content']}"
            for m in session.get("history", [])
        )
        sms_body = f"📋 Call summary ({caller}):\n{history_text[:800]}"
        send_sms(sms_body)

    # Save to call logs
    call_logs.append({
        "caller_number": session.get("caller_number"),
        "started_at": session.get("started_at"),
        "history": session.get("history", []),
        "action": action_type,
        "summary": action.get("message") or action.get("topic") or "Call ended",
    })


def send_sms(body: str):
    try:
        twilio_client.messages.create(
            body=body,
            from_=TWILIO_FROM_NUMBER,
            to=KRISH_PHONE_NUMBER,
        )
        print(f"[SMS] Sent: {body[:80]}...")
    except Exception as e:
        print(f"[SMS ERROR] {e}")


# ── 6. Mobile app API routes ──────────────────────────────────────────────────
@app.get("/calls")
def get_calls():
    return {"calls": list(reversed(call_logs))}

class Settings(BaseModel):
    name: str
    greeting: str
    about: str
    sms_enabled: bool = True
    calendar_enabled: bool = True

@app.post("/settings")
def update_settings(s: Settings):
    agent_settings.update(s.dict())
    return {"status": "saved"}

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": f"{agent_settings['name']}'s AI agent is running 🤖"}