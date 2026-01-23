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
    PLANNING = 3        
    PREDETERMINATION = 4 
    CALL_TO_ACTION = 5   
    COMPLETE = 6

class Stage5SubState(IntEnum):
    READY_FOR_SUMMARY = 0
    WAITING_FOR_FEELING = 1
    READY_FOR_INVESTMENT_PROMPT = 2  # [수정] 저축 권유 멘트 준비
    WAITING_FOR_INVESTMENT_RESPONSE = 3 # [수정] 유저의 저축 다짐 대기
    READY_FOR_FINAL_CODE = 4 # [수정] 마지막 반응 및 코드 발급

FINISH_CODE_MIN = 10000
FINISH_CODE_MAX = 99999

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
    def __init__(self):
        self.stage: Stage = Stage.INITIAL
        self.turn_count: int = 0
        self.small_talk_topics_covered: set = set()
        
        # Stage 4 Trackers
        self.planning_step_number: int = 1 
        self.step_phase: str = "ask_action" # ask_action <-> ask_detail
        
        # Stage 5 Trackers
        self.stage_5_substate: Stage5SubState = Stage5SubState.READY_FOR_SUMMARY
        self.retirement_timeframe: str = "your future retirement"

    def check_user_message_for_topics(self, message: str):
        msg = message.lower()
        if any(w in msg for w in ["year", "old", "age"]) or re.search(r'\b\d{1,2}\b', msg):
            self.small_talk_topics_covered.add("age")
        if any(w in msg for w in ["male", "female", "man", "woman", "gender", "he", "she"]):
            self.small_talk_topics_covered.add("gender")
        if any(w in msg for w in ["family", "member", "alone", "single", "married", "child"]):
            self.small_talk_topics_covered.add("family")
        if any(w in msg for w in ["retire", "work", "stop"]) or re.search(r'\b\d{2}\b', msg):
            self.small_talk_topics_covered.add("retirement_age")

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
        # --- Stage 1 ---
        if state.stage == Stage.INTRODUCTION:
            return "Current Task: Wait for user readiness. Transition to asking Age."

        # --- Stage 2 ---
        elif state.stage == Stage.SMALL_TALK:
            topics = state.small_talk_topics_covered
            remaining = []
            if "age" not in topics: remaining.append("Age")
            if "gender" not in topics: remaining.append("Gender")
            if "family" not in topics: remaining.append("Family size")
            if "retirement_age" not in topics: remaining.append("Expected retirement age")
            
            if not remaining:
                return "Current Task: Transition to Planning. Ask: 'What are the steps to retire in a financially prepared state through years of saving? I’d like you to think about three main steps. What is the first step?'"
            else:
                return f"Current Task: Ask about {remaining[0]}. Ask ONE question only."

        # --- Stage 3 ---
        elif state.stage == Stage.PLANNING:
             return "Current Task: Ask for specific execution details (How, When, Where) for the first step."

        # --- Stage 4 (Ping Pong) ---
        elif state.stage == Stage.PREDETERMINATION:
            step = state.planning_step_number
            phase = state.step_phase
            
            if phase == "ask_detail":
                return f"Current Task: User just provided Step {step}. Ask for specific execution details (How, When, Where) for this step."
            elif phase == "ask_action":
                if step == 2:
                    return "Current Task: User provided details for Step 1. Now ask for the Second Step."
                elif step == 3:
                    return "Current Task: User provided details for Step 2. Now ask for the Third (and final) Step."
        
        # --- Stage 5 (Strict Scripting) ---
        elif state.stage == Stage.CALL_TO_ACTION:
            sub = state.stage_5_substate
            
            if sub == Stage5SubState.READY_FOR_SUMMARY:
                return f"""
                Current Task: Synthesis.
                1. Summarize the user's plan.
                2. Ask: "How does thinking about this future plan make you feel?"
                """
            
            elif sub == Stage5SubState.READY_FOR_INVESTMENT_PROMPT:
                # [수정] 저축 권유 단계 (코드 질문 X)
                return """
                Current Task: Acknowledge & Invest Prompt.
                1. Acknowledge the user's feeling warmly.
                2. Say: "It is not always easy to think so far ahead, but doing so is a great step toward better financial preparedness. I hope this short conversation provided you with a meaningful perspective."
                3. Say: "Your tomorrow is built on what you do today. Why not invest in a brighter future by **saving a small amount every month starting today**?"
                4. STOP here. Wait for the user's response.
                """
            
            elif sub == Stage5SubState.READY_FOR_FINAL_CODE:
                # [수정] 마지막 인사 (시스템이 코드를 붙임)
                return """
                Current Task: Final Goodbye.
                1. The user just responded to your suggestion about saving (e.g., 'Okay', 'I will').
                2. Respond positively to their commitment.
                3. Say a brief goodbye. 
                (Do NOT generate the code yourself. The system will add it.)
                """

        return "Continue conversation."

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
            intro = "Hello! Thank you for joining. I’m here to be your thinking partner for a quick session on your life after your main career. Ready to look ahead?"
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
        
        self._add_message("user", user_input)
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
            
        self._update_state_pre_generation(user_input)
        
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                response = self.ai.generate_response(st.session_state.messages, st.session_state.state)
                
                # [핵심] 마지막 단계(READY_FOR_FINAL_CODE)에서 AI 응답 뒤에 코드 자동 부착
                if st.session_state.state.stage == Stage.COMPLETE:
                    response += f"\n\n---\n\n✅ **Your finish code is: {st.session_state.finish_code}**\n\nPlease save this code to continue with the survey."
                    st.session_state.completed = True
                    self.db.save_conversation(st.session_state.finish_code, st.session_state.messages)
                
                st.markdown(response)
                self._add_message("assistant", response)
        
        self._update_state_post_generation()

    def _update_state_pre_generation(self, user_input: str):
        """
        AI가 대답하기 전에 유저의 입력을 보고 상태를 업데이트 (Trigger logic)
        """
        state = st.session_state.state
        msg = user_input.lower()
        
        if state.stage == Stage.INTRODUCTION:
            if any(w in msg for w in ["yes", "ready", "sure", "ok", "start"]):
                state.stage = Stage.SMALL_TALK
                
        elif state.stage == Stage.SMALL_TALK:
            state.check_user_message_for_topics(user_input)
            if len(state.small_talk_topics_covered) >= 4:
                state.stage = Stage.PLANNING
            
        elif state.stage == Stage.PLANNING:
            state.stage = Stage.PREDETERMINATION
            state.planning_step_number = 1
            state.step_phase = "ask_detail"
            
        elif state.stage == Stage.CALL_TO_ACTION:
            # 유저가 감정을 말함 -> 저축 권유로 이동
            if state.stage_5_substate == Stage5SubState.WAITING_FOR_FEELING:
                state.stage_5_substate = Stage5SubState.READY_FOR_INVESTMENT_PROMPT
                
            # 유저가 저축 권유에 대답함 -> 종료 및 코드 발급으로 이동
            elif state.stage_5_substate == Stage5SubState.WAITING_FOR_INVESTMENT_RESPONSE:
                # [수정] 무조건 넘어감 (사용자가 No라고 해도 마무리는 해야 하므로)
                state.stage_5_substate = Stage5SubState.READY_FOR_FINAL_CODE
                # 여기서 COMPLETE로 바꾸지 않고, post_generation에서 바꿉니다 (AI가 마지막 인사는 해야 하니까)

    def _update_state_post_generation(self):
        """
        AI가 대답한 후에 다음 턴을 위한 상태 설정 (Transition logic)
        """
        state = st.session_state.state
        
        if state.stage == Stage.PREDETERMINATION:
            if state.step_phase == "ask_detail":
                state.step_phase = "ask_action"
            elif state.step_phase == "ask_action":
                if state.planning_step_number < 3:
                    state.planning_step_number += 1
                    state.step_phase = "ask_detail"
                else:
                    state.stage = Stage.CALL_TO_ACTION
                    state.stage_5_substate = Stage5SubState.READY_FOR_SUMMARY

        elif state.stage == Stage.CALL_TO_ACTION:
            if state.stage_5_substate == Stage5SubState.READY_FOR_SUMMARY:
                state.stage_5_substate = Stage5SubState.WAITING_FOR_FEELING
            elif state.stage_5_substate == Stage5SubState.READY_FOR_INVESTMENT_PROMPT:
                state.stage_5_substate = Stage5SubState.WAITING_FOR_INVESTMENT_RESPONSE
            elif state.stage_5_substate == Stage5SubState.READY_FOR_FINAL_CODE:
                state.stage = Stage.COMPLETE # 다음 렌더링 때 완료 처리

if __name__ == "__main__":
    app = PlanningApp()
    app.run()
