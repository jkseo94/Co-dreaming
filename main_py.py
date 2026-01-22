import streamlit as st
from openai import OpenAI
from supabase import create_client
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
import random
import re

# ======================================================
# CONFIG
# ======================================================

st.set_page_config(page_title="Saving for the Future", layout="centered")

SYSTEM_PROMPT = """
You are a friendly, realistic AI guiding users through a retirement imagination exercise.
Constraints:
- Each assistant turn must be under 50 words.
- Follow stages strictly.
- Never skip a stage.
- Never output empty messages.
"""

# ======================================================
# ENUMS & STATE
# ======================================================

class Stage(Enum):
    INTRO = 0
    SMALL_TALK = 1
    SIMULATION = 2
    PRE_EXPERIENCE = 3
    CALL_TO_ACTION = 4
    COMPLETE = 5

@dataclass
class SimulationState:
    stage: Stage = Stage.INTRO
    small_talk_answers: dict = field(default_factory=dict)
    stage4_turns: int = 0
    finish_code: str | None = None
    saved: bool = False

# ======================================================
# SERVICES
# ======================================================

class AIService:
    def __init__(self):
        self.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    def respond(self, messages, instruction):
        api_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": instruction},
        ]
        api_messages.extend(messages)

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=api_messages
        )

        text = response.choices[0].message.content.strip()
        return text if text else "Let’s keep going."

class DatabaseService:
    def __init__(self):
        self.supabase = create_client(
            st.secrets["SUPABASE_URL"],
            st.secrets["SUPABASE_SERVICE_KEY"]
        )

    def save(self, finish_code, messages):
        self.supabase.table("full_conversations").insert({
            "finish_code": finish_code,
            "full_conversation": messages,
            "finished_at": datetime.utcnow().isoformat()
        }).execute()

# ======================================================
# UTILITIES
# ======================================================

def generate_finish_code():
    return str(random.randint(100000, 999999))

def is_valid_stage3(text: str) -> bool:
    return bool(re.match(r"^I am ", text.strip(), re.IGNORECASE))

# ======================================================
# APP
# ======================================================

class SimulationApp:
    def __init__(self):
        self.ai = AIService()
        self.db = DatabaseService()
        self.init_state()

    def init_state(self):
        if "state" not in st.session_state:
            st.session_state.state = SimulationState()
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def assistant(self, text):
        st.session_state.messages.append({"role": "assistant", "content": text})
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(text)

    def user(self, text):
        st.session_state.messages.append({"role": "user", "content": text})

    def run(self):
        st.title("Saving for the Future")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        state = st.session_state.state

        if state.stage == Stage.INTRO and not st.session_state.messages:
            self.assistant(
                "Hello! I’d like to invite you to a short session designed to think about retirement.\n\n"
                "It can sometimes feel very distant, but exploring it now helps clarify what matters to you.\n\n"
                "Are you ready?"
            )

        if prompt := st.chat_input("Type your message here"):
            self.user(prompt)

            # ---------------- Stage 1 → 2 ----------------
            if state.stage == Stage.INTRO:
                if any(w in prompt.lower() for w in ["yes", "ready", "ok", "sure"]):
                    state.stage = Stage.SMALL_TALK
                    self.assistant(
                        "Great. I’d like to get to know you a little better.\n\nHow old are you right now?"
                    )
                return

            # ---------------- Stage 2 ----------------
            if state.stage == Stage.SMALL_TALK:
                if "age" not in state.small_talk_answers:
                    state.small_talk_answers["age"] = prompt
                    self.assistant("Thank you. How do you describe your gender?")
                    return

                if "gender" not in state.small_talk_answers:
                    state.small_talk_answers["gender"] = prompt
                    self.assistant("Got it. How many family members do you currently have?")
                    return

                if "family" not in state.small_talk_answers:
                    state.small_talk_answers["family"] = prompt

                    # ✅ 요청한 자연스러운 전환 + Stage 3 메시지 (같은 assistant, 다른 paragraph)
                    self.assistant(
                        "Thank you for sharing that. Having family often shapes how we think about the future and what truly matters to us.\n\n"
                        "Now, let’s try to think about a specific future event after many years of saving for retirement. "
                        "Please describe it using the phrase “I am,” as if you are there right now."
                    )
                    state.stage = Stage.SIMULATION
                    return

            # ---------------- Stage 3 ----------------
            if state.stage == Stage.SIMULATION:
                if not is_valid_stage3(prompt):
                    self.assistant(
                        "Please start your description with the words “I am,” as if you are experiencing it right now."
                    )
                    return

                state.stage = Stage.PRE_EXPERIENCE
                self.assistant("Where are you in this moment?")
                return

            # ---------------- Stage 4 ----------------
            if state.stage == Stage.PRE_EXPERIENCE:
                state.stage4_turns += 1

                if state.stage4_turns == 1:
                    self.assistant("Who are you with?")
                    return
                if state.stage4_turns == 2:
                    self.assistant("What can you see around you?")
                    return
                if state.stage4_turns == 3:
                    self.assistant("What sounds do you hear?")
                    return
                if state.stage4_turns == 4:
                    self.assistant("How does your body feel in this moment?")
                    return

                if 5 <= state.stage4_turns <= 7:
                    state.stage = Stage.CALL_TO_ACTION

            # ---------------- Stage 5 ----------------
            if state.stage == Stage.CALL_TO_ACTION:
                state.finish_code = generate_finish_code()

                self.assistant(
                    "You imagined a future shaped by preparation and care. "
                    "It may not be easy to think so far ahead, but doing so is a powerful step toward financial readiness.\n\n"
                    "Your tomorrow is built on what you do today. Why not invest in a brighter future by saving a small amount for retirement now?\n\n"
                    "I hope this short conversation provided you with a meaningful perspective on your retirement.\n\n"
                    f"Your finish code is **{state.finish_code}**."
                )

                self.db.save(state.finish_code, st.session_state.messages)
                state.saved = True
                state.stage = Stage.COMPLETE
                return


# ======================================================
# RUN
# ======================================================

if __name__ == "__main__":
    SimulationApp().run()
