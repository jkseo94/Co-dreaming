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
    INITIAL = 0
    INTRODUCTION = 1
    SMALL_TALK = 2
    PLANNING = 3        # Introduction to planning (Ask first step)
    PREDETERMINATION = 4 # Looping through Steps 1, 2, 3 and details
    CALL_TO_ACTION = 5   # Summary -> Feeling -> Closing
    COMPLETE = 6

class Stage5SubState(IntEnum):
    READY_FOR_SUMMARY = 0
    WAITING_FOR_FEELING = 1
    READY_FOR_CLOSING = 2
    WAITING_FOR_CODE_CONFIRMATION = 3

FINISH_CODE_MIN = 10000
FINISH_CODE_MAX = 99999
MAX_RETRIES = 3
RETRY_DELAY = 1

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
I’m here to be your thinking partner for a quick session on your life after your main career—whether that means traditional retirement or simply having the financial freedom to work less.\n\n
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
I’d like you to think about three main steps to achieve that retirement. These can be big or small.\n\n 
What is the first step?"

Stage 4 — Predetermination:
- Help users make their plans vividly with detailed and specific questions.
- Ask about: specific execution details, such as how they will execute the step, when they will do it, and where it will take place.
- Break down questions into separate responses (one step at a time). Do not ask for the full plan at once. (e.g., “What is the first step?” -> asking execution details -> “What is the second step?” …)
- Continue asking follow-up questions to actively facilitate users mentally constructing the course of action.
- This stage should last 5-7 conversational turns (guiding them through 3 distinct steps).
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
    role: str
    content: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict:
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
        
        # Planning Stage Trackers (Stage 4)
        self.planning_step_number: int = 1  # 1, 2, or 3
        self.step_phase: str = "ask_action" # 'ask_action' or 'ask_detail'
        
        # Stage 5 Sub-state tracker
        self.stage_5_substate: Stage5SubState = Stage5SubState.READY_FOR_SUMMARY
        
        # User Data
        self.retirement_timeframe: str = "your future retirement"

    def update_timeframe(self, age: int, retire_age: int):
        if age and retire_age:
            years = retire_age - age
            self.retirement_timeframe = f"{years} years"
        else:
            self.retirement_timeframe = "the future"

    def check_user_message_for_topics(self, message: str):
        """Extract topics from user message for stage 2."""
        msg = message.lower()
        if any(w in msg for w in ["year", "old", "age"]) or re.search(r'\b\d{1,2}\b', msg):
            self.small_talk_topics_covered.add("age")
        if any(w in msg for w in ["male", "female", "man", "woman", "gender", "he", "she"]):
            self.small_talk_topics_covered.add("gender")
        if any(w in msg for w in ["family", "member", "alone", "single", "married", "child"]):
            self.small_talk_topics_covered.add("family")
        if any(w in msg for w in ["retire", "work", "stop"]) or re.search(r'\b\d{2}\b', msg):
            self.small_talk_topics_covered.add("retirement_age")

# ==========================================
# SERVICES
# ==========================================

class DatabaseService:
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
            st.error(f"❌ Missing secrets: {', '.join(missing)}")
            st.stop()
    
    def is_finish_code_unique(self, code: str) -> bool:
        try:
            result = self.supabase.table("full_conversations_planning").select("finish_code").eq("finish_code", code).execute()
            return len(result.data) == 0
        except:
            return True
    
    def save_conversation(self, finish_code: str, messages: List[Message]) -> bool:
        try:
            data = {
                "finish_code": finish_code,
                "full_conversation": [msg.to_dict() for msg in messages],
                "finished_at": datetime.utcnow().isoformat()
            }
            self.supabase.table("full_conversations_planning").insert(data).execute()
            return True
        except Exception as e:
            st.error(f"Save failed: {e}")
            return False

class AIService:
    def __init__(self):
        self.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    
    def generate_response(self, messages: List[Message], state: ConversationState) -> Optional[str]:
        # Build strict context based on exact state
        context = self._build_context(state)
        
        api_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": context}
        ]
        
        for msg in messages[-10:]:
            api_messages.append({"role": msg.role, "content": msg.content})
            
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=api_messages,
                temperature=0.7,
                max_tokens=250
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"AI Error: {e}")
            return None

    def _build_context(self, state: ConversationState) -> str:
        """
        Dynamically constructs instructions based on the granular state.
        This effectively acts as the 'Director' of the conversation.
        """
        
        # --- Stage 1: Introduction ---
        if state.stage == Stage.INTRODUCTION:
            return "Current Task: Wait for the user to say they are ready. Once they agree, acknowledge it briefly and transition to asking specifically about their Age."

        # --- Stage 2: Small Talk ---
        elif state.stage == Stage.SMALL_TALK:
            topics = state.small_talk_topics_covered
            remaining = []
            if "age" not in topics: remaining.append("Age")
            if "gender" not in topics: remaining.append("Gender")
            if "family" not in topics: remaining.append("Family size")
            if "retirement_age" not in topics: remaining.append("Expected retirement age")
            
            if not remaining:
                return "Current Task: Thank the user for sharing. Now transition to the Planning Phase. Ask: 'What are the steps to retire in a financially prepared state through years of saving? I’d like you to think about three main steps. What is the first step?'"
            else:
                return f"Current Task: Small Talk. You must ask about {remaining[0]}. Ask ONE question only."

        # --- Stage 3: Planning Start (Transition handled above, this catches the response) ---
        elif state.stage == Stage.PLANNING:
             return "Current Task: The user just gave their First Step. Ask for specific execution details (How, When, Where) for this first step."

        # --- Stage 4: Predetermination (Looping) ---
        elif state.stage == Stage.PREDETERMINATION:
            step = state.planning_step_number
            phase = state.step_phase
            
            if phase == "ask_detail":
                return f"Current Task: The user just stated Step {step}. Now ask for specific execution details (How, When, Where) for Step {step}."
            elif phase == "ask_action":
                # Logic to ask for next step
                if step == 2:
                    return "Current Task: Ask for the Second Step to achieve their retirement goal."
                elif step == 3:
                    return "Current Task: Ask for the Third (and final) Step to achieve their retirement goal."
        
        # --- Stage 5: Call to Action (Strict Sequence) ---
        elif state.stage == Stage.CALL_TO_ACTION:
            sub = state.stage_5_substate
            
            if sub == Stage5SubState.READY_FOR_SUMMARY:
                return f"""
                Current Task: Synthesis & Feeling Check.
                1. Summarize the user's 3-step plan in a structured paragraph using 'To retire prepared in {state.retirement_timeframe}...' and 'I will...' statements.
                2. After the summary, in a new paragraph, ask exactly: "How does thinking about this future plan make you feel?"
                """
            
            elif sub == Stage5SubState.READY_FOR_CLOSING:
                return """
                Current Task: Validation & Closing.
                1. Warmly acknowledge the user's feeling (they just shared it).
                2. Say: "It is not always easy to think so far ahead, but doing so is a great step toward better financial preparedness. I hope this short conversation provided you with a meaningful perspective."
                3. Say: "Your tomorrow is built on what you do today. Why not invest in a brighter future by **saving a small amount every month starting today**?"
                4. Finally, ask: "Would you like to receive a finish code?"
                """
            
            elif sub == Stage5SubState.WAITING_FOR_CODE_CONFIRMATION:
                return "Current Task: The user wants the code. Simply say 'Here is your code:' and stop. The system will display the code."

        return "Continue conversation naturally."

# ==========================================
# APPLICATION CONTROLLER
# ==========================================

class PlanningApp:
    def __init__(self):
        self.db = DatabaseService()
        self.ai = AIService()
        self._init_session()
        
    def _init_session(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.session_state.state = ConversationState()
            st.session_state.finish_code = self._generate_code()
            st.session_state.completed = False
            
    def _generate_code(self):
        while True:
            code = str(random.randint(FINISH_CODE_MIN, FINISH_CODE_MAX))
            if self.db.is_finish_code_unique(code): return code

    def run(self):
        self._render_ui()
        self._handle_initial()
        self._render_history()
        self._handle_input()
        
    def _render_ui(self):
        st.set_page_config(page_title="Future Planning", page_icon="💰")
        st.markdown("<style>#MainMenu, footer, header {visibility: hidden;}</style>", unsafe_allow_html=True)
        st.title("💰 Saving for the Future")
        
    def _handle_initial(self):
        if not st.session_state.messages:
            intro = (
                "Hello! Thank you for joining.\n\n"
                "I’m here to be your thinking partner for a quick session on your life after your main career. "
                "Looking that far ahead can be challenging on your own, but exploring it together can help clarify what truly matters.\n\n"
                "Ready to look ahead?"
            )
            self._add_message("assistant", intro)
            st.session_state.state.stage = Stage.INTRODUCTION

    def _add_message(self, role, content):
        st.session_state.messages.append(Message(role=role, content=content))
        
    def _render_history(self):
        for msg in st.session_state.messages:
            with st.chat_message(msg.role, avatar="🤖" if msg.role == "assistant" else "👤"):
                st.markdown(msg.content)
                
    def _handle_input(self):
        if st.session_state.completed:
            st.success(f"✅ Planning complete! Code: **{st.session_state.finish_code}**")
            return

        user_input = st.chat_input("Type your message...")
        if not user_input: return
        
        # 1. User Message
        self._add_message("user", user_input)
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
            
        # 2. Logic Update BEFORE AI generation (Analyze user input)
        self._update_state_pre_generation(user_input)
        
        # 3. AI Generation
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                response = self.ai.generate_response(st.session_state.messages, st.session_state.state)
                
                # Check if we need to append the finish code manually (if confirmed)
                if st.session_state.state.stage == Stage.COMPLETE:
                    response += f"\n\n---\n\n✅ **Your finish code is: {st.session_state.finish_code}**\n\nPlease save this code to continue with the survey."
                    st.session_state.completed = True
                    self.db.save_conversation(st.session_state.finish_code, st.session_state.messages)
                
                st.markdown(response)
                self._add_message("assistant", response)
        
        # 4. Logic Update AFTER AI generation (Prepare state for NEXT turn)
        self._update_state_post_generation()

    def _update_state_pre_generation(self, user_input: str):
        state = st.session_state.state
        msg = user_input.lower()
        
        # Introduction -> Small Talk
        if state.stage == Stage.INTRODUCTION:
            if any(w in msg for w in ["yes", "ready", "sure", "ok", "start"]):
                state.stage = Stage.SMALL_TALK
                
        # Small Talk Logic
        elif state.stage == Stage.SMALL_TALK:
            state.check_user_message_for_topics(user_input)
            # Try to calculate timeframe
            if "age" in state.small_talk_topics_covered and "retirement_age" in state.small_talk_topics_covered:
                # Minimal parsing logic for demo purposes
                pass 
                
        # Small Talk -> Planning
        if state.stage == Stage.SMALL_TALK and len(state.small_talk_topics_covered) >= 4:
            state.stage = Stage.PLANNING
            
        # Planning -> Stage 4 (User just gave step 1)
        elif state.stage == Stage.PLANNING:
            state.stage = Stage.PREDETERMINATION
            state.planning_step_number = 1
            state.step_phase = "ask_detail" # AI needs to ask detail now
            
        # Stage 4 Logic (Ping Pong: Action -> Detail -> Action -> Detail)
        elif state.stage == Stage.PREDETERMINATION:
            if state.step_phase == "ask_detail":
                # User just gave the Action. State was "ask_detail" so AI context told it to ask details.
                # Now we wait for AI to ask.
                pass 
            elif state.step_phase == "ask_action":
                # User just gave details. State was "ask_action" so AI context told it to ask next step.
                pass
                
        # Stage 5 Logic
        elif state.stage == Stage.CALL_TO_ACTION:
            if state.stage_5_substate == Stage5SubState.WAITING_FOR_FEELING:
                # User just answered feeling
                state.stage_5_substate = Stage5SubState.READY_FOR_CLOSING
            elif state.stage_5_substate == Stage5SubState.WAITING_FOR_CODE_CONFIRMATION:
                if any(w in msg for w in ["yes", "yeah", "code", "please"]):
                    state.stage = Stage.COMPLETE

    def _update_state_post_generation(self):
        """
        Advance the logical state markers after the AI has successfully asked its question.
        This sets up the expectations for the user's NEXT response.
        """
        state = st.session_state.state
        
        # If AI just asked for details, next input will be details.
        # After that input, we need to ask for next action.
        if state.stage == Stage.PREDETERMINATION:
            if state.step_phase == "ask_detail":
                # AI just asked details. Next, user speaks details. Then we switch to ask_action.
                state.step_phase = "ask_action"
            elif state.step_phase == "ask_action":
                # AI just asked for next Step. Next user speaks step. Then we switch to ask_detail.
                # Also increment step number
                if state.planning_step_number < 3:
                    state.planning_step_number += 1
                    state.step_phase = "ask_detail"
                else:
                    # We just finished Step 3 Details. Move to Stage 5
                    state.stage = Stage.CALL_TO_ACTION
                    state.stage_5_substate = Stage5SubState.READY_FOR_SUMMARY

        # Stage 5 Progressions
        elif state.stage == Stage.CALL_TO_ACTION:
            if state.stage_5_substate == Stage5SubState.READY_FOR_SUMMARY:
                # AI just gave summary and asked feeling. Next we wait for feeling.
                state.stage_5_substate = Stage5SubState.WAITING_FOR_FEELING
            elif state.stage_5_substate == Stage5SubState.READY_FOR_CLOSING:
                # AI just gave closing and asked "Want code?". Next we wait for Yes.
                state.stage_5_substate = Stage5SubState.WAITING_FOR_CODE_CONFIRMATION

if __name__ == "__main__":
    app = PlanningApp()
    app.run()
