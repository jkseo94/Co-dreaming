"""
Retirement Planning Chatbot
A Streamlit application that guides users through a structured conversation
about retirement planning using AI-powered dialogue.

Author: Refactored Version - Planning Condition
Date: 2026-01-22
"""

import streamlit as st
from openai import OpenAI
from supabase import create_client
from datetime import datetime
from enum import IntEnum
from typing import List, Dict, Optional
from dataclasses import dataclass
import random
import time
import re


# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================

class Stage(IntEnum):
    """Conversation stages for retirement planning."""
    INITIAL = 0
    INTRODUCTION = 1
    SMALL_TALK = 2
    PLANNING = 3
    PREDETERMINATION = 4
    CALL_TO_ACTION = 5
    COMPLETE = 6


FINISH_CODE_MIN = 10000
FINISH_CODE_MAX = 99999
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

SYSTEM_PROMPT = """
Role: You are an AI agent designed to help users generate simulations of future-oriented thinking that involves imaginatively placing oneself in a hypothetical scenario. Your ultimate purpose is to help users mentally pre-experience the future and decision-making in intertemporal choices, such as saving.

Constraints:
- Make sure each conversation response is less than 80 words.
- Please follow the stages strictly in the order provided.
- Tone: Friendly, realistic, empathetic

Dialogue Stages:
Follow this sequence strictly. Do not skip steps.

Stage 1 — Introduction:
- Introduce yourself as follows: "Hello! Thank you for joining. I'm here to be your thinking partner for a quick session on your life after your main career—whether that means traditional retirement or simply having the financial freedom to work less. Looking that far ahead can be challenging on your own, but exploring it together can help clarify what truly matters to you. There are no right or wrong answers. Ready to look ahead?"

Stage 2 — Small Talk:
- Smoothly transition by letting users know you want to get to know them better.
- Ask ALL of the following questions ONE BY ONE (wait for response between each):
  1. "How old are you right now?"
  2. "How do you describe your gender?"
  3. "How many family members do you currently have?"
  4. "At what age do you expect to retire or start significantly cutting back on work?"
- INTERNAL LOGIC (Perform after Question 4):
  - IF the user provides a number: Calculate [Answer to Q4] minus [Answer to Q1] = X. Remember "X years" as the timeframe for the final cue.
  - IF the user is unsure: Do not force a number. Set the timeframe as "your future retirement" for the final cue.
  - IF the user is already retired: Set the timeframe as "a few years".

Stage 3 — Planning:
- Guide users to identify and organize the concrete steps needed to arrive at their future where work is optional or done on their own terms.
- Help them construct a future plan to achieve financial security through saving.
- CRITICAL: Ask them to describe specific future plans by using the following opening question: "What are the steps to retire in a financially prepared state through years of saving? I'd like you to think about three main steps to achieve that retirement. These can be big or small. What is the first step?"
- Wait for their response about the first step before proceeding.

Stage 4 — Predetermination:
- Help users make their plans vivid with detailed and specific questions.
- Ask about specific execution details: how they will execute the step, when they will do it, and where it will take place.
- Break down questions into separate responses (one step at a time). Do not ask for the full plan at once.
- Process: Ask about Step 1 -> Get execution details -> Ask about Step 2 -> Get execution details -> Ask about Step 3 -> Get execution details
- Continue asking follow-up questions to actively facilitate users mentally constructing the course of action.
- This stage should last 5-7 conversational turns (guiding them through 3 distinct steps with details).
- Only proceed to Stage 5 after sufficient execution detail has been gathered for all 3 steps.

Stage 5 — Call to Action:
- Step 1 Synthesis: Based on the concrete steps the user provided, write a short structured paragraph (3-4 sentences) summarizing their future plan.
  - Start with the calculated timeframe: "To retire prepared in [X] years..." or "To retire prepared in your future..." or "To retire prepared in a few years..."
  - Use "I will" statements to denote determination.
  - Include the specific execution details (how, when, where) the user planned for each step.
- Step 2 Presentation: Present this text. Say "Here is the action plan for your future:" followed by the paragraph.
- Step 3 Validation: Ask "How does thinking about this future plan make you feel?" in a separate paragraph.
- Step 4 Response: When the user responds, warmly acknowledge their feeling and transition smoothly.
- Step 5 Closing: End with three distinct parts:
  - Part 1: "It is not always easy to think so far ahead, but doing so is a great step toward better financial preparedness. I hope this short conversation provided you with a meaningful perspective."
  - Part 2: "Your tomorrow is built on what you do today. Why not invest in a brighter future by saving a small amount every month starting today?"
  - Part 3: Ask strictly: "Would you like to receive your finish code?"
  - WAIT for the user to say "Yes".

Important Guidelines:
- SECURITY RULE: You are STRICTLY FORBIDDEN from generating or showing the finish code yourself.
- Do NOT say "Here is your code: ..." under any circumstances.
- The finish code is a secure system variable that ONLY the interface can display.
- Just ask "Would you like to receive your finish code?" and stop.
- Ensure meaningful engagement at each stage before progressing
- If a user gives a very brief answer, ask follow-up questions to encourage elaboration
- Maintain a warm, supportive tone throughout
"""


@dataclass
class Message:
    """Represents a single conversation message."""
    role: str
    content: str
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }


class ConversationState:
    """Manages conversation state and validation."""

    def __init__(self):
        self.stage: Stage = Stage.INITIAL
        self.turn_count: int = 0
        self.small_talk_topics_covered: set = set()

        # Stage 5 sub-steps
        self.provided_summary: bool = False
        self.asked_feeling: bool = False
        self.user_responded_feeling: bool = False
        self.provided_closing: bool = False
        self.asked_for_code: bool = False
        self.user_wants_code: bool = False

        # -------- Stage 3/4: plan capture (content-based) --------
        self.steps = {
            1: {"what": None, "how": None, "when": None, "where": None},
            2: {"what": None, "how": None, "when": None, "where": None},
            3: {"what": None, "how": None, "when": None, "where": None},
        }

        # Stage 4에서 다음 유저 답변을 어디에 저장할지(what or details)
        # {"step": 1|2|3, "kind": "what"|"details", "fields": ["how","when","where"](kind=details일 때)}
        self.pending_request: Optional[Dict[str, Any]] = None

    def advance_turn(self):
        self.turn_count += 1

    def advance_stage(self):
        self.stage = Stage(self.stage + 1)

    # ---------- Stage 2 ----------
    def can_advance_from_stage_2(self) -> bool:
        return len(self.small_talk_topics_covered) >= 4

    def check_user_message_for_topics(self, message: str):
        message_lower = message.lower()

        if any(word in message_lower for word in ["year", "old", "age"]) or re.search(r'\b\d{1,2}\b', message):
            self.small_talk_topics_covered.add("age")

        if any(word in message_lower for word in ["male", "female", "man", "woman", "gender", "non-binary", "they", "he", "she"]):
            self.small_talk_topics_covered.add("gender")

        if any(word in message_lower for word in ["family", "member", "people", "person", "wife", "husband", "child", "parent", "sibling", "alone", "single"]) \
            or re.search(r'\b\d+\b', message):
            self.small_talk_topics_covered.add("family")

        if any(word in message_lower for word in ["retire", "retirement", "quit", "stop working"]) or re.search(r'\b\d{2}\b', message):
            self.small_talk_topics_covered.add("retirement_age")

    # ---------- Stage 3/4 data helpers ----------
    def set_step_what(self, step_num: int, text: str):
        text = (text or "").strip()
        if text:
            self.steps[step_num]["what"] = text

    def update_step_details(self, step_num: int, details: Dict[str, Optional[str]]):
        for k in ["how", "when", "where"]:
            v = details.get(k)
            if v:
                self.steps[step_num][k] = v.strip()

    def missing_detail_fields(self, step_num: int) -> List[str]:
        missing = []
        for k in ["how", "when", "where"]:
            if not self.steps[step_num][k]:
                missing.append(k)
        return missing

    def plan_complete_for_stage_5(self) -> bool:
        for i in [1, 2, 3]:
            s = self.steps[i]
            if not s["what"]:
                return False
            if not s["how"] or not s["when"] or not s["where"]:
                return False
        return True

    def next_stage4_request(self) -> Optional[Dict[str, Any]]:
        """
        Stage 4에서 다음에 물어볼 것을 결정.
        - Step1: details(how/when/where) 부족분을 한 번에 묻기
        - Step2: what 없으면 what 먼저, 그다음 details
        - Step3: what 없으면 what 먼저, 그다음 details
        """
        # Step 1 details
        m1 = self.missing_detail_fields(1)
        if m1:
            return {"step": 1, "kind": "details", "fields": m1}

        # Step 2 what / details
        if not self.steps[2]["what"]:
            return {"step": 2, "kind": "what"}
        m2 = self.missing_detail_fields(2)
        if m2:
            return {"step": 2, "kind": "details", "fields": m2}

        # Step 3 what / details
        if not self.steps[3]["what"]:
            return {"step": 3, "kind": "what"}
        m3 = self.missing_detail_fields(3)
        if m3:
            return {"step": 3, "kind": "details", "fields": m3}

        return None

    # ---------- Stage 5 substeps ----------
    def mark_summary_provided(self):
        self.provided_summary = True
        self.asked_feeling = True

    def mark_feeling_response(self):
        self.user_responded_feeling = True

    def mark_closing_provided(self):
        self.provided_closing = True
        self.asked_for_code = True

    def check_for_code_request(self, message: str) -> bool:
        message_lower = message.lower().strip()
        # "yes" 단독/유사 응답 처리
        affirmative = ["yes", "yeah", "yep", "sure", "ok", "okay", "please", "code"]
        if any(word == message_lower or word in message_lower for word in affirmative):
            self.user_wants_code = True
            return True
        return False


# ==========================================
# SERVICES
# ==========================================

class DatabaseService:
    """Handles all database operations."""

    def __init__(self):
        self._validate_secrets()
        try:
            self.supabase = create_client(
                st.secrets["SUPABASE_URL"],
                st.secrets["SUPABASE_SERVICE_KEY"]
            )
        except Exception as error:
            st.error(f"❌ Database connection failed: {error}")
            st.stop()

    @staticmethod
    def _validate_secrets():
        required = ["SUPABASE_URL", "SUPABASE_SERVICE_KEY", "OPENAI_API_KEY"]
        missing = [key for key in required if key not in st.secrets]
        if missing:
            st.error(f"❌ Missing required secrets: {', '.join(missing)}")
            st.stop()

    def is_finish_code_unique(self, code: str) -> bool:
        try:
            result = self.supabase.table("full_conversations_planning")\
                .select("finish_code")\
                .eq("finish_code", code)\
                .execute()
            return len(result.data) == 0
        except Exception as error:
            # 유니크 체크 실패 시에도 세션 진행은 허용
            st.warning(f"⚠️ Could not verify finish code uniqueness: {error}")
            return True

    def save_conversation(self, finish_code: str, messages: List[Message]) -> bool:
        try:
            messages_data = [msg.to_dict() for msg in messages]

            data = {
                "finish_code": finish_code,
                "full_conversation": messages_data,
                "finished_at": datetime.utcnow().isoformat()
            }

            self.supabase.table("full_conversations_planning").insert(data).execute()
            return True
        except Exception as error:
            st.error(f"❌ Failed to save conversation: {error}")
            return False


class AIService:
    """Handles AI model interactions."""

    def __init__(self):
        try:
            self.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        except Exception as error:
            st.error(f"❌ Failed to initialize AI service: {error}")
            st.stop()

    def generate_response(
        self,
        messages: List[Message],
        current_stage: Stage,
        state: ConversationState
    ) -> Optional[str]:
        stage_context = self._build_stage_context(current_stage, state)

        api_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": stage_context}
        ]

        for msg in messages[-10:]:
            api_messages.append({"role": msg.role, "content": msg.content})

        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=api_messages,
                    temperature=0.7,
                    max_tokens=220
                )
                text = response.choices[0].message.content.strip()
                if not text:
                    st.warning("⚠️ AI generated empty response, retrying...")
                    continue
                return text
            except Exception as error:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                st.error(f"❌ AI service error after {MAX_RETRIES} attempts: {error}")
                return None

        return None

    def extract_details(self, user_text: str, fields_needed: List[str]) -> Dict[str, Optional[str]]:
        """
        유저의 답변에서 how/when/where를 구조화해서 뽑아냄.
        fields_needed에 포함된 항목만 의미있게 채우고, 못 뽑으면 null.
        """
        fields_needed = [f for f in fields_needed if f in ["how", "when", "where"]]
        if not fields_needed:
            return {"how": None, "when": None, "where": None}

        extractor_system = (
            "You are an information extraction engine. "
            "Extract the requested fields from the user's message. "
            "Return ONLY valid JSON with keys how, when, where. "
            "If a field is not present or unclear, set it to null. "
            "Do not add extra keys."
        )
        extractor_user = (
            f"Requested fields: {fields_needed}\n\n"
            f"User message:\n{user_text}"
        )

        for attempt in range(MAX_RETRIES):
            try:
                # response_format이 지원되는 환경이면 JSON 강제
                response = self.client.chat.completions.create(
                    model=EXTRACT_MODEL,
                    messages=[
                        {"role": "system", "content": extractor_system},
                        {"role": "user", "content": extractor_user}
                    ],
                    temperature=0.0,
                    max_tokens=180,
                    response_format={"type": "json_object"}  # 미지원이면 예외 가능
                )
                raw = response.choices[0].message.content.strip()
                data = json.loads(raw)

                out = {"how": None, "when": None, "where": None}
                for k in ["how", "when", "where"]:
                    v = data.get(k)
                    if isinstance(v, str) and v.strip():
                        out[k] = v.strip()
                    else:
                        out[k] = None
                return out

            except Exception:
                # fallback: response_format 미지원/JSON 파싱 실패 시 일반 호출 후 정규식 JSON 추출 시도
                try:
                    response = self.client.chat.completions.create(
                        model=EXTRACT_MODEL,
                        messages=[
                            {"role": "system", "content": extractor_system},
                            {"role": "user", "content": extractor_user}
                        ],
                        temperature=0.0,
                        max_tokens=220
                    )
                    raw = response.choices[0].message.content.strip()
                    # JSON 객체만 뽑기(가장 바깥 {})
                    m = re.search(r"\{.*\}", raw, re.DOTALL)
                    if not m:
                        continue
                    data = json.loads(m.group(0))
                    out = {"how": None, "when": None, "where": None}
                    for k in ["how", "when", "where"]:
                        v = data.get(k)
                        out[k] = v.strip() if isinstance(v, str) and v.strip() else None
                    return out
                except Exception:
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY * (attempt + 1))
                        continue
                    return {"how": None, "when": None, "where": None}

        return {"how": None, "when": None, "where": None}

    @staticmethod
    def _build_stage_context(stage: Stage, state: ConversationState) -> str:
        parts = [f"Current Stage: {stage.name} (Stage {stage.value})"]

        if stage == Stage.SMALL_TALK:
            covered = ", ".join(sorted(state.small_talk_topics_covered)) if state.small_talk_topics_covered else "none"
            parts.append(f"Topics covered so far: {covered}")
            parts.append("Ask the next missing small-talk question (ONE at a time) among: age, gender, family members, retirement age.")

        elif stage == Stage.PLANNING:
            parts.append("Ask the Stage 3 opening question exactly as instructed and WAIT for the user's first step.")

        elif stage == Stage.PREDETERMINATION:
            req = state.pending_request or state.next_stage4_request()
            if not req:
                parts.append("All steps and execution details have been collected. You may proceed to Stage 5.")
            else:
                step_num = req["step"]
                kind = req["kind"]
                if kind == "what":
                    parts.append(f"Ask ONLY for Step {step_num} itself (what the step is). One question.")
                else:
                    fields = req.get("fields", ["how", "when", "where"])
                    # 한 메시지에서 how/when/where를 같이 묻되, 빠진 항목만 묻기
                    pretty = ", ".join(fields)
                    parts.append(
                        f"Ask for Step {step_num} execution details in ONE message: request {pretty}. "
                        "If the user previously missed any of these, ask ONLY the missing ones."
                    )

        elif stage == Stage.CALL_TO_ACTION:
            if not state.provided_summary:
                parts.append("CRITICAL: Provide synthesis paragraph + ask 'How does thinking about this future plan make you feel?' (separate paragraph).")
            elif not state.user_responded_feeling:
                parts.append("Wait for user to respond to the feeling question.")
            elif not state.provided_closing:
                parts.append("CRITICAL: Acknowledge feeling + provide the 3-part closing and ask: 'Would you like to receive your finish code?' Then stop.")
            else:
                parts.append("Wait for user to confirm they want the finish code (Yes).")

        return " | ".join(parts)


# ==========================================
# APPLICATION CONTROLLER
# ==========================================

class PlanningApp:
    """Main application controller."""

    def __init__(self):
        self.db = DatabaseService()
        self.ai = AIService()
        self.initialize_session_state()

    def initialize_session_state(self):
        defaults = {
            "messages": [],
            "finish_code": self._generate_unique_finish_code(),
            "planning_complete": False,
            "data_saved": False,
            "state": ConversationState()
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def _generate_unique_finish_code(self) -> str:
        max_attempts = 10
        for _ in range(max_attempts):
            code = str(random.randint(FINISH_CODE_MIN, FINISH_CODE_MAX))
            if self.db.is_finish_code_unique(code):
                return code
        return str(int(time.time() * 1000) % 100000)

    def run(self):
        self._render_ui()
        self._handle_initial_message()
        self._render_chat_history()
        self._handle_user_input()

    def _render_ui(self):
        st.set_page_config(
            page_title="Saving for the Future",
            page_icon="💰",
            layout="centered"
        )
        st.markdown("""
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
        """, unsafe_allow_html=True)

        st.title("💰 Saving for the Future")

    def _handle_initial_message(self):
        if not st.session_state.messages:
            welcome_message = Message(
                role="assistant",
                content=(
                    "Hello! Thank you for joining.\n\n"
                    "I'm here to be your thinking partner for a quick session on your life after your main career—"
                    "whether that means traditional retirement or simply having the financial freedom to work less.\n\n"
                    "Looking that far ahead can be challenging on your own, but exploring it together can help "
                    "clarify what truly matters to you. There are no right or wrong answers.\n\n"
                    "Ready to look ahead?"
                )
            )
            st.session_state.messages.append(welcome_message)
            st.session_state.state.stage = Stage.INTRODUCTION

    def _render_chat_history(self):
        for message in st.session_state.messages:
            avatar = "🤖" if message.role == "assistant" else "👤"
            with st.chat_message(message.role, avatar=avatar):
                st.markdown(message.content)

    def _handle_user_input(self):
        # COMPLETE이면 UI로 finish code 보여주고 종료
        if st.session_state.planning_complete:
            st.success(f"✅ Planning session complete! Your finish code: **{st.session_state.finish_code}**")
            st.info("Please save this code and return to the survey.")
            return

        user_input = st.chat_input("Type your message here...")
        if not user_input:
            return

        # 1) 유저 메시지 저장/표시
        user_message = Message(role="user", content=user_input)
        st.session_state.messages.append(user_message)
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        # 2) 유저 입력 기반으로 state 업데이트 + stage 전환(핵심: AI 호출 전에 전환)
        self._process_user_input_and_maybe_advance(user_input)

        # 3) Stage 4라면 이번 assistant 질문이 무엇인지 pending_request를 확정
        state = st.session_state.state
        if state.stage == Stage.PREDETERMINATION:
            state.pending_request = state.next_stage4_request()

        # 4) assistant 응답 생성
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                response_text = self.ai.generate_response(
                    st.session_state.messages,
                    st.session_state.state.stage,
                    st.session_state.state
                )
                if not response_text:
                    st.error("Failed to generate response. Please try again.")
                    return

                # Stage 5 substeps 트래킹 (your/a finish code 문구 둘 다 처리)
                self._check_stage_5_substeps(response_text)

                st.markdown(response_text)

                # assistant 메시지 저장
                assistant_message = Message(role="assistant", content=response_text)
                st.session_state.messages.append(assistant_message)
                st.session_state.state.advance_turn()

        # 5) COMPLETE가 되었으면 UI로 코드 표시 + 저장
        if st.session_state.state.stage == Stage.COMPLETE:
            st.session_state.planning_complete = True

        if st.session_state.planning_complete and not st.session_state.data_saved:
            success = self.db.save_conversation(
                st.session_state.finish_code,
                st.session_state.messages
            )
            if success:
                st.session_state.data_saved = True

    def _process_user_input_and_maybe_advance(self, user_input: str):
        state = st.session_state.state

        # Stage 1: readiness
        if state.stage == Stage.INTRODUCTION:
            affirmative_words = ["yes", "ready", "sure", "ok", "start", "yeah", "yep", "let's", "lets", "go"]
            if any(word in user_input.lower() for word in affirmative_words):
                state.advance_stage()  # -> SMALL_TALK
            return

        # Stage 2: track topics & advance when done
        if state.stage == Stage.SMALL_TALK:
            state.check_user_message_for_topics(user_input)
            if state.can_advance_from_stage_2():
                state.advance_stage()  # -> PLANNING
            return

        # Stage 3: capture Step 1 "what" then advance to Stage 4
        if state.stage == Stage.PLANNING:
            # 유저가 첫 번째 step을 말하면 Step1 what 저장 후 바로 Stage4로 이동
            if user_input.strip():
                state.set_step_what(1, user_input)
                state.advance_stage()  # -> PREDETERMINATION
            return

        # Stage 4: pending_request 기반으로 what 또는 details 저장 + 완료되면 Stage 5로
        if state.stage == Stage.PREDETERMINATION:
            req = state.pending_request
            # 만약 pending_request가 없으면(예외), 다음 요청을 계산
            if not req:
                req = state.next_stage4_request()

            if req:
                step_num = req["step"]
                kind = req["kind"]

                if kind == "what":
                    state.set_step_what(step_num, user_input)

                else:
                    fields = req.get("fields", ["how", "when", "where"])
                    details = self.ai.extract_details(user_input, fields_needed=fields)
                    state.update_step_details(step_num, details)

                # 처리 후 pending_request 해제
                state.pending_request = None

            # 모든 step+details가 모이면 Stage 5로
            if state.plan_complete_for_stage_5():
                state.advance_stage()  # -> CALL_TO_ACTION
            return

        # Stage 5: feeling 응답 / code 요청 처리, Yes면 COMPLETE로
        if state.stage == Stage.CALL_TO_ACTION:
            if state.asked_feeling and not state.user_responded_feeling:
                state.mark_feeling_response()
                return

            if state.asked_for_code and not state.user_wants_code:
                if state.check_for_code_request(user_input):
                    state.advance_stage()  # -> COMPLETE
            return

    def _check_stage_5_substeps(self, assistant_response: str):
        state = st.session_state.state
        if state.stage != Stage.CALL_TO_ACTION:
            return

        # Summary + feeling question detection
        if ("Here is the action plan" in assistant_response or "action plan for your future" in assistant_response):
            if re.search(r"how does.*feel\??", assistant_response, re.IGNORECASE):
                state.mark_summary_provided()

        # Closing + code question detection: "your finish code" / "a finish code" 둘 다 인식
        if re.search(r"would you like to receive (your|a) finish code\??", assistant_response, re.IGNORECASE):
            state.mark_closing_provided()


# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":
    app = PlanningApp()
    app.run()
