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
Role: You are an AI agent designed to help users identify and organize future plans needed to arrive at a financially prepared retirement. Your ultimate purpose is to help users predetermine a course of action aimed at achieving some goals and decision-making in intertemporal choices, such as saving.

Constraints:
- Make sure each conversation response is less than 80 words.
- Please follow the stages strictly in the order provided.
- Tone: Friendly, realistic, empathetic

Dialogue Stages:
Follow this sequence strictly. Do not skip steps.

Stage 1 — Introduction:
- Introduce yourself as follows: "Hello! Thank you for joining.\n\n
I'm here to be your thinking partner for a quick session on your life after your main career—whether that means traditional retirement or simply having the financial freedom to work less.\n\n
Looking that far ahead can be challenging on your own, but exploring it together can help clarify what truly matters to you. There are no right or wrong answers.\n\n
Ready to look ahead?"

Stage 2 — Small Talk:
- Smoothly transition by letting users know you want to get to know them better.
- Ask ALL of the following questions ONE BY ONE (wait for response between each):
  1. "How old are you right now?"
  2. "How do you describe your gender?"
  3. "How many family members do you currently have?"
  4. "At what age do you expect to retire or start significantly cutting back on work?"
- **INTERNAL LOGIC (Perform after Question 4):**
  - **IF the user provides a number:** Calculate [Answer to Q4] minus [Answer to Q1] = X. (e.g., 65 - 30 = 35). Remember "35 years" as the timeframe for the final cue.
  - **IF the user is unsure (e.g., "I don't know"):** Do not force a number. Simply accept it and set the timeframe as "your future retirement" for the final cue.
  - **IF the user is already retired:** Set the timeframe as "a few years".

Stage 3 — Planning:
- Guide users to identify and organize the concrete steps needed to arrive at their future where work is optional or done on their own terms.
- Help them construct a future plan to achieve financial security through saving—whether that means fully retiring, working fewer hours to pursue passions, or simply having the freedom to design their days exactly as they wish.
- CRITICAL: Ask them to describe specific future plans by using the following opening questions: "What are the steps to retire in a financially prepared state through years of saving?\n\n
I'd like you to think about two main steps to achieve that retirement. These can be big or small.\n\n 
What is the first step?"

Stage 4 — Predetermination:
- Help users make their plans vividly with detailed and specific questions.
- Ask about: specific execution details, such as how they will execute the step, when they will do it, and where it will take place.
- Break down questions into separate responses (one step at a time). Do not ask for the full plan at once. (e.g., "What is the first step?" -> asking execution details -> "What is the second step?" …)
- Continue asking follow-up questions to actively facilitate users mentally constructing the course of action.
- This stage should last 5-7 conversational turns (guiding them through 2 distinct steps with detailed follow-ups).
- Only proceed to Stage 5 after sufficient execution detail has been gathered.

Stage 5 — Call to Action (Do not show this title): 
- **Step 1: Synthesis.** Based on the concrete steps the user provided in the previous turns, write a short, structured paragraph (3-4 sentences) summarizing their future plan. 
- You MUST follow this format strictly: 
 1. Say: "Here is the action plan for your future: [Insert the paragraph you wrote following below format]" 
 2. **Start with the calculated timeframe:** - **If you calculated X in Stage 2:** Start with "**To retire prepared in [X] years**..." (e.g., "To retire prepared in 20 years..."). - **If X was unknown:** Start with "**To retire prepared in the future**..." 
 3. Use "I will" statements to denote determination (e.g., "First, I will...", "Then, I will..."). 
 4. Include the specific execution details (how, when, where) the user planned for each step. 
- **Step 2: Presentation & Validation.** Present this text to the user naturally. 
- Say: "How does thinking about this future plan make you feel?" in a separate paragraph. 
- **Step 3: Validation.** When the user responds with their feeling: 
- FIRST, warmly acknowledge.
- THEN, smoothly transition using a bridge.
- **Step 4: Closing.** End on a hopeful note
 - You must output exactly three distinct parts.
 Part 1: "It is not always easy to think so far ahead, but doing so is a great step toward better financial preparedness. I hope this short conversation provided you with a meaningful perspective.\n\n"
 Part 2: "Your tomorrow is built on what you do today. Why not invest in a brighter future by **saving a small amount every month starting today**?\n\n"
 Part 3: Ask them if they want to receive a finish code.

    
Important Guidelines:
- Never generate or mention a finish code - the system will provide this automatically
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

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
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
        self.stage_turn_count: int = 0

        # Stage-specific counters (kept for prompt context / analytics)
        self.stage_4_turns: int = 0
        self.call_to_action_turns: int = 0

        # Stage 2 tracking
        self.small_talk_topics_covered: set = set()

        # ✅ Event-based progression (Stage 3 -> 4 -> 5)
        # Stage 3 (Planning): require user to provide a meaningful first step
        self.planning_step1_provided: bool = False
        self.step1_text: str = ""

        # Stage 4 (Predetermination): require both steps + execution detail (how/when/where)
        self.active_step: int = 1  # 1 or 2, driven by assistant prompts
        self.step2_provided: bool = False
        self.step2_text: str = ""

        self.step1_details = {"how": False, "when": False, "where": False}
        self.step2_details = {"how": False, "when": False, "where": False}

    def advance_turn(self):
        """Increment turn counters."""
        self.turn_count += 1
        self.stage_turn_count += 1
        if self.stage == Stage.PREDETERMINATION:
            self.stage_4_turns += 1
        if self.stage == Stage.CALL_TO_ACTION:
            self.call_to_action_turns += 1

    def advance_stage(self):
        """Move to next stage and reset stage-specific counters."""
        self.stage = Stage(self.stage + 1)
        self.stage_turn_count = 0

        # Reset / initialize stage-local state where helpful
        if self.stage == Stage.PREDETERMINATION:
            self.active_step = 1
        if self.stage == Stage.CALL_TO_ACTION:
            # Call-to-action stage uses simple count-based exit (stage_turn_count >= 2)
            pass

    def can_advance_from_stage_2(self) -> bool:
        """Check if Stage 2 (Small Talk) requirements are met - need 4 topics."""
        return len(self.small_talk_topics_covered) >= 4

    def can_advance_from_stage_3(self) -> bool:
        """Event-based: advance when user provided a meaningful first step."""
        return self.planning_step1_provided

    def can_advance_from_stage_4(self) -> bool:
        """Event-based: advance when both steps have execution detail (how/when/where)."""

        def enough_details(d: dict) -> bool:
            # Require at least 2 of 3 (how/when/where) to avoid getting stuck,
            # while still enforcing specificity.
            return sum(1 for v in d.values() if v) >= 2

        return (
            self.planning_step1_provided
            and self.step2_provided
            and enough_details(self.step1_details)
            and enough_details(self.step2_details)
        )

    def can_advance_from_stage_5(self) -> bool:
        """Count-based within Stage 5 only: after 2 assistant turns, advance to COMPLETE."""
        return self.stage_turn_count >= 2

    def check_user_message_for_topics(self, message: str):
        """Extract topics from user message for stage 2."""
        message_lower = message.lower()

        # Detect age-related responses
        if any(word in message_lower for word in ["year", "old", "age"]) or re.search(r'\b\d{1,2}\b', message):
            self.small_talk_topics_covered.add("age")

        # Detect gender-related responses
        if any(word in message_lower for word in ["male", "female", "man", "woman", "gender", "non-binary", "they", "he", "she"]):
            self.small_talk_topics_covered.add("gender")

        # Detect family-related responses
        if any(word in message_lower for word in ["family", "member", "people", "person", "wife", "husband", "child", "parent", "sibling", "alone", "single"]) or re.search(r'\b\d+\b', message):
            self.small_talk_topics_covered.add("family")

        # Detect retirement age responses
        if any(word in message_lower for word in ["retire", "retirement", "quit", "stop working"]) or re.search(r'\b\d{2}\b', message):
            self.small_talk_topics_covered.add("retirement_age")

    # --------------------------
    # ✅ Event detectors / recorders
    # --------------------------

    def record_planning_step1(self, user_text: str):
        """Record Stage 3 (Planning) first-step response and mark completion if meaningful."""
        text = (user_text or "").strip()
        text_lower = text.lower()

        # If user explicitly indicates they don't know, do not advance
        unsure_markers = ["i don't know", "dont know", "not sure", "no idea", "idk", "모르", "잘 모르", "모르겠"]
        if any(m in text_lower for m in unsure_markers) and len(text) < 30:
            return

        # Require minimal substance (avoid accidental advancement on 'ok', 'yes', etc.)
        if len(text.split()) < 3 and len(text) < 20:
            return

        self.planning_step1_provided = True
        self.step1_text = text

    def check_ai_message_for_step_prompt(self, assistant_text: str):
        """Update which step the user is likely answering next, based on assistant prompt text."""
        if not assistant_text:
            return
        t = assistant_text.lower()

        # If the assistant explicitly asks for the second step, switch active step to 2
        if re.search(r"\bsecond\s+step\b", t) or "두 번째" in t or "두번째" in t:
            self.active_step = 2
            return

        # If the assistant explicitly asks for the first step (rare in Stage 4), ensure step 1 active
        if re.search(r"\bfirst\s+step\b", t) or "첫 번째" in t or "첫번째" in t:
            self.active_step = 1

    def record_predetermination_reply(self, user_text: str):
        """Record Stage 4 user reply as either step definition or execution details for the active step."""
        text = (user_text or "").strip()
        if not text:
            return
        t = text.lower()

        # Heuristics for execution-detail signals
        how_markers = ["how", "plan", "use", "set up", "automate", "contribute", "save", "invest", "budget", "enroll", "increase", "track",
                       "방법", "계획", "설정", "자동", "저축", "투자", "예산", "가입", "늘리", "기록"]
        when_markers = ["when", "every", "weekly", "monthly", "yearly", "each", "by", "before", "after", "starting", "next", "tomorrow",
                        "주", "월", "매", "매달", "매월", "매주", "매년", "부터", "다음", "이번", "내일"]
        where_markers = ["where", "at", "in", "from", "bank", "app", "online", "work", "employer", "broker", "account",
                         "에서", "은행", "앱", "온라인", "회사", "직장", "계좌", "브로커"]

        def has_any(markers):
            return any(m in t for m in markers)

        # Step identification: capture a second step when it is requested or explicitly mentioned
        explicit_second = bool(re.search(r"\bsecond\s+step\b", t)) or ("두 번째" in t) or ("두번째" in t)
        if (self.active_step == 2 or explicit_second) and not self.step2_provided:
            if len(text.split()) >= 3:
                self.step2_provided = True
                self.step2_text = text

        # Details:
        # Details: assign to active step
        details = self.step1_details if self.active_step == 1 else self.step2_details
        if has_any(how_markers) or len(text.split()) >= 6:
            details["how"] = True
        if has_any(when_markers) or re.search(r"\b\d{1,4}\b", text):
            details["when"] = True
        if has_any(where_markers):
            details["where"] = True


# ==========================================
# SERVICES
# ==========================================

class DatabaseService:
    """Handles all database operations."""

    def __init__(self):
        """Initialize Supabase client with error handling."""
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
        """Ensure required secrets are present."""
        required = ["SUPABASE_URL", "SUPABASE_SERVICE_KEY", "OPENAI_API_KEY"]
        missing = [key for key in required if key not in st.secrets]
        if missing:
            st.error(f"❌ Missing required secrets: {', '.join(missing)}")
            st.stop()

    def is_finish_code_unique(self, code: str) -> bool:
        """Check if finish code already exists in database."""
        try:
            result = self.supabase.table("full_conversations_planning")\
                .select("finish_code")\
                .eq("finish_code", code)\
                .execute()
            return len(result.data) == 0
        except Exception as error:
            st.warning(f"⚠️ Could not verify finish code uniqueness: {error}")
            return True  # Proceed anyway

    def save_conversation(self, finish_code: str, messages: List[Message]) -> bool:
        """
        Save the complete conversation to database.
        
        Args:
            finish_code: Unique identifier for this conversation
            messages: List of Message objects
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert messages to serializable format
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
        """Initialize OpenAI client."""
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
        """
        Generate AI response with retry logic.
        
        Args:
            messages: Conversation history
            current_stage: Current stage of conversation
            state: Current conversation state for context
            
        Returns:
            AI response string or None if failed
        """
        # Build context-aware system message
        stage_context = self._build_stage_context(current_stage, state)

        api_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": stage_context}
        ]

        # Add conversation history (last 8 messages for faster response)
        for msg in messages[-8:]:
            api_messages.append({
                "role": msg.role,
                "content": msg.content
            })

        # Retry logic with shorter delays
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=api_messages,
                    temperature=0.7,
                    max_tokens=200,  # Slightly higher for planning details
                    stream=False  # Set to True for streaming responses
                )

                response_text = response.choices[0].message.content.strip()

                # Validate response is not empty
                if not response_text:
                    if attempt < MAX_RETRIES - 1:
                        continue
                    st.warning("⚠️ AI generated empty response")
                    return None

                return response_text

            except Exception as error:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(0.5 * (attempt + 1))  # Shorter delays
                    continue
                else:
                    st.error(f"❌ AI service error: {error}")
                    return None

        return None

    @staticmethod
    def _build_stage_context(stage: Stage, state: ConversationState) -> str:
        """Build stage-specific context for AI."""
        context_parts = [f"Current Stage: {stage.name} (Stage {stage.value})"]

        if stage == Stage.SMALL_TALK:
            covered = ", ".join(state.small_talk_topics_covered) if state.small_talk_topics_covered else "none"
            context_parts.append(f"Topics covered: {covered}")
            context_parts.append("You must ask about: age, gender, family members, and retirement age (one question at a time)")

        elif stage == Stage.PLANNING:
            context_parts.append("Ask the opening question about two main steps to retire financially prepared")
            context_parts.append("Wait for user's response about first step, then proceed to Stage 4")

        elif stage == Stage.PREDETERMINATION:
            context_parts.append(f"Turn {state.stage_4_turns + 1} of minimum 5 required turns")
            if state.stage_4_turns < 5:
                context_parts.append("Guide user through BOTH 2 STEPS with execution details (how, when, where)")
                context_parts.append("ONE step at a time: Step 1 (what) -> Step 1 details (how/when/where) -> Step 2 (what) -> Step 2 details (how/when/where)")
                context_parts.append("Ask multiple follow-up questions per step to reach 5+ turns total")
            else:
                context_parts.append("You completed 5+ turns. Verify both steps have details before moving to Stage 5")

        elif stage == Stage.CALL_TO_ACTION:
            context_parts.append(f"Call to Action turn {state.call_to_action_turns + 1}")
            if state.call_to_action_turns == 0:
                context_parts.append("Provide synthesis paragraph + ask 'How does thinking about this future plan make you feel?'")
            elif state.call_to_action_turns == 1:
                context_parts.append("User responded with feeling. Acknowledge + provide 3-part closing + ask for finish code")
            else:
                context_parts.append("User should be confirming code request. Conversation should complete soon")

        return " | ".join(context_parts)


# ==========================================
# APPLICATION CONTROLLER
# ==========================================

class PlanningApp:
    """Main application controller."""

    def __init__(self):
        """Initialize services and session state."""
        self.db = DatabaseService()
        self.ai = AIService()
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize or restore session state."""
        defaults = {
            "messages": [],
            "finish_code": None,  # Generate only when needed
            "planning_complete": False,
            "data_saved": False,
            "state": ConversationState()
        }

        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
        
        # Generate finish code only once, when first needed
        if st.session_state.finish_code is None:
            st.session_state.finish_code = self._generate_unique_finish_code()

    def _generate_unique_finish_code(self) -> str:
        """Generate a unique finish code."""
        max_attempts = 10
        for _ in range(max_attempts):
            code = str(random.randint(FINISH_CODE_MIN, FINISH_CODE_MAX))
            if self.db.is_finish_code_unique(code):
                return code

        # Fallback: use timestamp-based code
        return str(int(time.time() * 1000) % 100000)

    def run(self):
        """Main application loop."""
        self._render_ui()
        self._handle_initial_message()
        self._render_chat_history()
        self._handle_user_input()

    def _render_ui(self):
        """Render page configuration and styling."""
        st.set_page_config(
            page_title="Saving for the Future",
            page_icon="💰",
            layout="centered"
        )

        # Hide Streamlit branding
        st.markdown("""
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
        """, unsafe_allow_html=True)

        st.title("💰 Saving for the Future")

    def _handle_initial_message(self):
        """Send initial greeting if conversation just started."""
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
        """Display all messages in conversation."""
        # Only render messages that haven't been rendered yet
        # Streamlit automatically handles chat message display
        for message in st.session_state.messages:
            avatar = "🤖" if message.role == "assistant" else "👤"
            with st.chat_message(message.role, avatar=avatar):
                st.markdown(message.content)

    def _handle_user_input(self):
        """Process user input and generate response."""
        # Show finish code if planning complete
        if st.session_state.planning_complete:
            st.success(f"✅ Planning session complete! Your finish code: **{st.session_state.finish_code}**")
            st.info("Please save this code and return to the survey.")
            return

        # Chat input
        user_input = st.chat_input("Type your message here...")
        if not user_input:
            return

        # Display user message
        user_message = Message(role="user", content=user_input)
        st.session_state.messages.append(user_message)

        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        # Update state based on user input
        self._process_user_input(user_input)

        # Generate and display assistant response
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
                
                # Update state based on the assistant's prompt (used for Stage 4 step tracking)
                state = st.session_state.state
                if state.stage == Stage.PREDETERMINATION:
                    state.check_ai_message_for_step_prompt(response_text)

                # Check if we should advance stages BEFORE appending message
                self._check_stage_progression()
                
                # If just completed, append finish code to response
                if st.session_state.state.stage == Stage.COMPLETE:
                    response_text += f"\n\n---\n\n✅ **Your finish code is: {st.session_state.finish_code}**\n\nPlease save this code to continue with the survey."
                    st.session_state.planning_complete = True
                
                st.markdown(response_text)

                # Save assistant message
                assistant_message = Message(role="assistant", content=response_text)
                st.session_state.messages.append(assistant_message)
                st.session_state.state.advance_turn()

        # Save to database if complete
        if st.session_state.planning_complete and not st.session_state.data_saved:
            success = self.db.save_conversation(
                st.session_state.finish_code,
                st.session_state.messages
            )
            if success:
                st.session_state.data_saved = True

    def _process_user_input(self, user_input: str):
        """Process user input and update state accordingly."""
        state: ConversationState = st.session_state.state
        user_text = user_input or ""
        user_lower = user_text.lower()

        # Stage 1: readiness gate
        if state.stage == Stage.INTRODUCTION:
            affirmative_words = ["yes", "ready", "sure", "ok", "start", "yeah", "yep", "let's", "lets", "go", "네", "예", "응", "좋아", "시작"]
            if any(word in user_lower for word in affirmative_words):
                state.advance_stage()
            return

        # Stage 2: small talk topic tracking (age/gender/family/retirement age)
        if state.stage == Stage.SMALL_TALK:
            state.check_user_message_for_topics(user_text)
            return

        # Stage 3 -> 4 (event-based): user provides a meaningful first step
        if state.stage == Stage.PLANNING:
            state.record_planning_step1(user_text)
            if state.can_advance_from_stage_3():
                state.advance_stage()  # -> PREDETERMINATION
            return

        # Stage 4 -> 5 (event-based): both steps + execution detail captured
        if state.stage == Stage.PREDETERMINATION:
            state.record_predetermination_reply(user_text)
            if state.can_advance_from_stage_4():
                state.advance_stage()  # -> CALL_TO_ACTION
            return

    def _check_stage_progression(self):
        """Determine if stage should advance based on completion criteria."""
        state: ConversationState = st.session_state.state

        # Stage 2 -> 3: topic coverage (still fine as count/topic-based)
        if state.stage == Stage.SMALL_TALK and state.can_advance_from_stage_2():
            state.advance_stage()
            return

        # Stage 5 -> 6: keep count-based exit within Stage 5 only
        if state.stage == Stage.CALL_TO_ACTION and state.can_advance_from_stage_5():
            state.advance_stage()
            return


# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":
    app = PlanningApp()
    app.run()
