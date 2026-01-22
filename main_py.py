"""
Retirement Simulation Chatbot
A Streamlit application that guides users through a structured conversation
about retirement planning using AI-powered dialogue.

Author: Refactored Version
Date: 2026-01-22
"""

import streamlit as st
from openai import OpenAI
from supabase import create_client
from datetime import datetime
from enum import IntEnum
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import random
import time
import re


# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================

class Stage(IntEnum):
    """Conversation stages for retirement simulation."""
    INITIAL = 0
    INTRODUCTION = 1
    SMALL_TALK = 2
    SIMULATION = 3
    PRE_EXPERIENCE = 4
    CALL_TO_ACTION = 5
    COMPLETE = 6


FINISH_CODE_MIN = 10000
FINISH_CODE_MAX = 99999
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

SYSTEM_PROMPT = """
Role: You are an AI agent designed to help users generate simulations of future-oriented thinking that involves imaginatively placing oneself in a hypothetical scenario. Your ultimate purpose is to help users mentally pre-experience the future and decision-making in intertemporal choices, such as saving.

Constraints:
- Make sure each conversation response is less than 50 words.
- Please follow the stages strictly in the order provided.
- Tone: Friendly, realistic, empathetic

Dialogue Stages:
Follow this sequence strictly. Do not skip steps.

Stage 1 — Introduction:
- Introduce yourself as follows: "Hello! I'd like to invite you to a short session designed to think about retirement. It can sometimes feel very distant, but exploring it now helps clarify what matters to you. Are you ready?"

Stage 2 — Small Talk:
- Smoothly transition by letting users know you want to get to know them better.
- Ask ALL of the following questions ONE BY ONE (wait for response between each):
  1. "How old are you right now?"
  2. "How do you describe your gender?"
  3. "How many family members do you currently have?"

Stage 3 — Simulation:
- Guide users to mentally pre-experience what life would be like after years of saving.
- Help them think about retirement in a financially prepared state.
- CRITICAL: Ask them to describe a specific future event using the phrase "I am" as if they are there right now (e.g., "I am at the beach swimming" or "I am making coffee in my kitchen").
- Wait for their "I am" response before proceeding.

Stage 4 — Pre-experience (Extended Exploration):
- Help users simulate more vividly with detailed and specific questions.
- Ask about: where they are, who they are with, what they are hearing, what they are seeing.
- Break down questions into separate responses. Do not ask everything at once.
- Continue asking follow-up questions to deepen the visualization.
- This stage should last 5-7 conversational turns (your responses).
- Only proceed to Stage 5 after sufficient detail has been gathered.

Stage 5 — Call to Action:
- Recap and synthesize the future event the user constructed during the conversation.
- Ask how they feel about it.
- End on a hopeful note: "It is not always easy to think so far ahead, but doing so is a great step toward better financial preparedness."
- Suggest saving now can help: "Your tomorrow is built on what you do today. Why not invest in a brighter future by saving a small amount for your retirement now?"
- Send the final message: "I hope this short conversation provided you with a meaningful perspective on your retirement."

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
        self.stage_4_turns: int = 0
        self.questions_asked: List[str] = []
        self.user_provided_i_am: bool = False
        self.small_talk_topics_covered: set = set()
        
    def advance_turn(self):
        """Increment turn counters."""
        self.turn_count += 1
        self.stage_turn_count += 1
        if self.stage == Stage.PRE_EXPERIENCE:
            self.stage_4_turns += 1
    
    def advance_stage(self):
        """Move to next stage and reset stage-specific counters."""
        self.stage = Stage(self.stage + 1)
        self.stage_turn_count = 0
        
    def can_advance_from_stage_2(self) -> bool:
        """Check if Stage 2 (Small Talk) requirements are met."""
        return len(self.small_talk_topics_covered) >= 3
    
    def can_advance_from_stage_3(self) -> bool:
        """Check if Stage 3 (Simulation) requirements are met."""
        return self.user_provided_i_am
    
    def can_advance_from_stage_4(self) -> bool:
        """Check if Stage 4 (Pre-experience) requirements are met."""
        return self.stage_4_turns >= 5
    
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
    
    def check_for_i_am_phrase(self, message: str) -> bool:
        """Check if user used 'I am' phrase in their message."""
        pattern = r'\bI\s+am\b'
        if re.search(pattern, message, re.IGNORECASE):
            self.user_provided_i_am = True
            return True
        return False


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
            result = self.supabase.table("full_conversations")\
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
            
            self.supabase.table("full_conversations").insert(data).execute()
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
        
        # Add conversation history (last 10 messages for context window management)
        for msg in messages[-10:]:
            api_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Retry logic
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=api_messages,
                    temperature=0.7,
                    max_tokens=150  # Enforce brevity
                )
                
                response_text = response.choices[0].message.content.strip()
                
                # Validate response is not empty
                if not response_text:
                    st.warning("⚠️ AI generated empty response, retrying...")
                    continue
                    
                return response_text
                
            except Exception as error:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                else:
                    st.error(f"❌ AI service error after {MAX_RETRIES} attempts: {error}")
                    return None
        
        return None
    
    @staticmethod
    def _build_stage_context(stage: Stage, state: ConversationState) -> str:
        """Build stage-specific context for AI."""
        context_parts = [f"Current Stage: {stage.name} (Stage {stage.value})"]
        
        if stage == Stage.SMALL_TALK:
            covered = ", ".join(state.small_talk_topics_covered) if state.small_talk_topics_covered else "none"
            context_parts.append(f"Topics covered: {covered}")
            context_parts.append("You must ask about: age, gender, and family members (one question at a time)")
            
        elif stage == Stage.SIMULATION:
            context_parts.append("CRITICAL: User must respond with an 'I am' statement describing a future event")
            context_parts.append("Do not proceed until you receive this")
            
        elif stage == Stage.PRE_EXPERIENCE:
            context_parts.append(f"Turn {state.stage_4_turns + 1} of minimum 5 required turns in this stage")
            if state.stage_4_turns < 5:
                context_parts.append("Continue asking detailed follow-up questions about their visualization")
            else:
                context_parts.append("You have completed 5 turns. You may wrap up this stage if sufficient detail gathered")
                
        elif stage == Stage.CALL_TO_ACTION:
            context_parts.append("Provide recap, ask about feelings, give call to action, then final message")
        
        return " | ".join(context_parts)


# ==========================================
# APPLICATION CONTROLLER
# ==========================================

class SimulationApp:
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
            "finish_code": self._generate_unique_finish_code(),
            "simulation_complete": False,
            "data_saved": False,
            "state": ConversationState()
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
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
            .stChatMessage {padding: 1rem;}
            </style>
        """, unsafe_allow_html=True)
        
        st.title("💰 Saving for the Future")
        st.caption("A guided conversation about retirement planning")
    
    def _handle_initial_message(self):
        """Send initial greeting if conversation just started."""
        if not st.session_state.messages:
            welcome_message = Message(
                role="assistant",
                content=(
                    "Hello! I'd like to invite you to a short session designed to "
                    "think about retirement. It can sometimes feel very distant, but "
                    "exploring it now helps clarify what matters to you.\n\n"
                    "Are you ready?"
                )
            )
            st.session_state.messages.append(welcome_message)
            st.session_state.state.stage = Stage.INTRODUCTION
    
    def _render_chat_history(self):
        """Display all messages in conversation."""
        for message in st.session_state.messages:
            avatar = "🤖" if message.role == "assistant" else "👤"
            with st.chat_message(message.role, avatar=avatar):
                st.markdown(message.content)
    
    def _handle_user_input(self):
        """Process user input and generate response."""
        # Show finish code if simulation complete
        if st.session_state.simulation_complete:
            st.success(f"✅ Simulation complete! Your finish code: **{st.session_state.finish_code}**")
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
                
                # Check if we should advance stages BEFORE appending message
                self._check_stage_progression()
                
                # If just completed, append finish code
                if st.session_state.state.stage == Stage.COMPLETE:
                    response_text += f"\n\n---\n\n✅ **Your finish code is: {st.session_state.finish_code}**\n\nPlease save this code to continue with the survey."
                    st.session_state.simulation_complete = True
                
                st.markdown(response_text)
                
                # Save assistant message
                assistant_message = Message(role="assistant", content=response_text)
                st.session_state.messages.append(assistant_message)
                st.session_state.state.advance_turn()
        
        # Save to database if complete
        if st.session_state.simulation_complete and not st.session_state.data_saved:
            success = self.db.save_conversation(
                st.session_state.finish_code,
                st.session_state.messages
            )
            if success:
                st.session_state.data_saved = True
    
    def _process_user_input(self, user_input: str):
        """Process user input and update state accordingly."""
        state = st.session_state.state
        
        # Check for readiness to start (from Stage 1)
        if state.stage == Stage.INTRODUCTION:
            affirmative_words = ["yes", "ready", "sure", "ok", "start", "yeah", "yep", "let's", "lets"]
            if any(word in user_input.lower() for word in affirmative_words):
                state.advance_stage()
        
        # Track topics in Stage 2
        elif state.stage == Stage.SMALL_TALK:
            state.check_user_message_for_topics(user_input)
        
        # Check for "I am" phrase in Stage 3
        elif state.stage == Stage.SIMULATION:
            state.check_for_i_am_phrase(user_input)
    
    def _check_stage_progression(self):
        """Determine if stage should advance based on completion criteria."""
        state = st.session_state.state
        
        if state.stage == Stage.SMALL_TALK and state.can_advance_from_stage_2():
            state.advance_stage()
            
        elif state.stage == Stage.SIMULATION and state.can_advance_from_stage_3():
            state.advance_stage()
            
        elif state.stage == Stage.PRE_EXPERIENCE and state.can_advance_from_stage_4():
            state.advance_stage()
            
        elif state.stage == Stage.CALL_TO_ACTION and state.stage_turn_count >= 2:
            # After recap and call to action (typically 2-3 turns)
            state.advance_stage()


# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":
    app = SimulationApp()
    app.run()
