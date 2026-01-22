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
  - Part 3: Ask them if they want to receive a finish code.

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
        self.steps_mentioned: int = 0  # Track how many steps user has mentioned
        self.small_talk_topics_covered: set = set()
        
    def advance_turn(self):
        """Increment turn counters."""
        self.turn_count += 1
        self.stage_turn_count += 1
        if self.stage == Stage.PREDETERMINATION:
            self.stage_4_turns += 1
    
    def advance_stage(self):
        """Move to next stage and reset stage-specific counters."""
        self.stage = Stage(self.stage + 1)
        self.stage_turn_count = 0
        
    def can_advance_from_stage_2(self) -> bool:
        """Check if Stage 2 (Small Talk) requirements are met - need 4 topics."""
        return len(self.small_talk_topics_covered) >= 4
    
    def can_advance_from_stage_3(self) -> bool:
        """Check if Stage 3 (Planning) requirements are met - user mentioned first step."""
        return self.stage_turn_count >= 1  # At least one response to planning question
    
    def can_advance_from_stage_4(self) -> bool:
        """Check if Stage 4 (Predetermination) requirements are met - 5+ turns."""
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

        # Detect retirement age responses
        if any(word in message_lower for word in ["retire", "retirement", "quit", "stop working"]) or re.search(r'\b\d{2}\b', message):
            self.small_talk_topics_covered.add("retirement_age")
    
    def check_for_step_mention(self, message: str):
        """Check if user mentioned a step in their planning."""
        message_lower = message.lower()
        step_keywords = ["step", "first", "second", "third", "start", "begin", "will", "plan", "going to"]
        if any(keyword in message_lower for keyword in step_keywords):
            self.steps_mentioned += 1


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
                    max_tokens=200  # Slightly higher for planning details
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
            context_parts.append("You must ask about: age, gender, family members, and retirement age (one question at a time)")
            
        elif stage == Stage.PLANNING:
            context_parts.append("CRITICAL: User must respond with their first step for retirement planning")
            context_parts.append("Ask the opening question about three main steps to retire in a financially prepared state")
            
        elif stage == Stage.PREDETERMINATION:
            context_parts.append(f"Turn {state.stage_4_turns + 1} of minimum 5 required turns in this stage")
            if state.stage_4_turns < 5:
                context_parts.append("Continue asking detailed execution questions about their steps (how, when, where)")
                context_parts.append("Guide them through all 3 steps with specific details")
            else:
                context_parts.append("You have completed 5 turns. You may wrap up this stage if all 3 steps have sufficient detail")
                
        elif stage == Stage.CALL_TO_ACTION:
            context_parts.append("Provide synthesis of their plan, ask about feelings, give call to action, then final message")
        
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
            "finish_code": self._generate_unique_finish_code(),
            "planning_complete": False,
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
                
                # Check if we should advance stages BEFORE appending message
                self._check_stage_progression()
                
                # If just completed, append finish code
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
        state = st.session_state.state
        
        # Check for readiness to start (from Stage 1)
        if state.stage == Stage.INTRODUCTION:
            affirmative_words = ["yes", "ready", "sure", "ok", "start", "yeah", "yep", "let's", "lets", "go"]
            if any(word in user_input.lower() for word in affirmative_words):
                state.advance_stage()
        
        # Track topics in Stage 2
        elif state.stage == Stage.SMALL_TALK:
            state.check_user_message_for_topics(user_input)
        
        # Track step mentions in Stage 3
        elif state.stage == Stage.PLANNING:
            state.check_for_step_mention(user_input)
    
    def _check_stage_progression(self):
        """Determine if stage should advance based on completion criteria."""
        state = st.session_state.state
        
        if state.stage == Stage.SMALL_TALK and state.can_advance_from_stage_2():
            state.advance_stage()
            
        elif state.stage == Stage.PLANNING and state.can_advance_from_stage_3():
            state.advance_stage()
            
        elif state.stage == Stage.PREDETERMINATION and state.can_advance_from_stage_4():
            state.advance_stage()
            
        elif state.stage == Stage.CALL_TO_ACTION and state.stage_turn_count >= 3:
            # After synthesis, validation response, and closing (3+ turns)
            state.advance_stage()


# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":
    app = PlanningApp()
    app.run()
