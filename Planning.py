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

FINISH_CODE_MIN = 10000
FINISH_CODE_MAX = 99999
MAX_RETRIES = 3

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
 - You must output exactly two distinct parts.
 Part 1: "It is not always easy to think so far ahead, but doing so is a great step toward better financial preparedness. I hope this short conversation provided you with a meaningful perspective.\n\n"
 Part 2: "Your tomorrow is built on what you do today. Why not invest in a brighter future by **saving a small amount every month starting today**?\n\n"
    
Important Guidelines:
- Do not generate the numeric code - the system will provide finish code automatically
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
        self.stage_turn_count: int = 0  # 현재 스테이지에서 몇 번째 말하는지 카운트
        self.stage_4_turns: int = 0
        self.user_provided_i_am: bool = False
        self.small_talk_topics_covered: set = set()
        
    def advance_turn(self):
        self.turn_count += 1
        self.stage_turn_count += 1
        if self.stage == Stage.PRE_EXPERIENCE:
            self.stage_4_turns += 1

    def advance_stage(self):
        self.stage = Stage(self.stage + 1)
        self.stage_turn_count = 0 # 스테이지 바뀌면 턴 카운트 초기화
    
    def check_user_message_for_topics(self, message: str):
        msg = message.lower()
        if any(w in msg for w in ["year", "old", "age"]) or re.search(r'\b\d{1,2}\b', msg):
            self.small_talk_topics_covered.add("age")
        if any(w in msg for w in ["male", "female", "man", "woman", "gender"]) or "non-binary" in msg:
            self.small_talk_topics_covered.add("gender")
        if any(w in msg for w in ["family", "member", "alone", "single", "child"]):
            self.small_talk_topics_covered.add("family")
        if any(w in msg for w in ["retire", "work", "stop"]) or re.search(r'\b\d{2}\b', msg):
            self.small_talk_topics_covered.add("retirement_age")
    
    def check_for_i_am_phrase(self, message: str) -> bool:
        if re.search(r'\bI\s+am\b', message, re.IGNORECASE):
            self.user_provided_i_am = True
            return True
        return False

# ==========================================
# SERVICES
# ==========================================

class DatabaseService:
    def __init__(self):
        self._validate_secrets()
        try:
            self.supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_SERVICE_KEY"])
        except:
            st.error("Database connection failed")
            st.stop()
    
    @staticmethod
    def _validate_secrets():
        required = ["SUPABASE_URL", "SUPABASE_SERVICE_KEY", "OPENAI_API_KEY"]
        if any(k not in st.secrets for k in required):
            st.error("Missing secrets")
            st.stop()
    
    def is_finish_code_unique(self, code: str) -> bool:
        try:
            result = self.supabase.table("full_conversations").select("finish_code").eq("finish_code", code).execute()
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
            self.supabase.table("full_conversations").insert(data).execute()
            return True
        except:
            return False

class AIService:
    def __init__(self):
        self.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    
    def generate_response(self, messages: List[Message], current_stage: Stage, state: ConversationState) -> Optional[str]:
        stage_context = self._build_stage_context(current_stage, state)
        api_messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "system", "content": stage_context}]
        for msg in messages[-10:]:
            api_messages.append({"role": msg.role, "content": msg.content})
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview", messages=api_messages, temperature=0.7, max_tokens=250
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return None
    
    @staticmethod
    def _build_stage_context(stage: Stage, state: ConversationState) -> str:
        context_parts = [f"Current Stage: {stage.name}"]
        
        if stage == Stage.SMALL_TALK:
            covered = ", ".join(state.small_talk_topics_covered) if state.small_talk_topics_covered else "none"
            context_parts.append(f"Topics covered: {covered}. Ask about remaining: age, gender, family (one by one).")
            
        elif stage == Stage.SIMULATION:
            context_parts.append("CRITICAL: User must respond with 'I am...' describing a future event. Wait for this.")
            
        elif stage == Stage.PRE_EXPERIENCE:
            context_parts.append(f"Turn {state.stage_4_turns + 1} of min 5. Ask detailed sensory questions (sights, sounds, people).")
                
        elif stage == Stage.CALL_TO_ACTION:
            # [핵심] 턴 수에 따라 AI가 할 말을 기계적으로 지정 (Turn-based Logic)
            turn = state.stage_turn_count
            
            if turn == 0:
                return """
                Task: Synthesis & Feeling Check.
                1. Summarize user's future event vividly.
                2. Ask: "How does thinking about this future make you feel?"
                """
            elif turn == 1:
                return """
                Task: Investment Prompt.
                1. Acknowledge the user's feeling warmly.
                2. Say: "Your tomorrow is built on what you do today. Why not invest in a brighter future by **saving a small amount every month starting today**?"
                """
            else:
                return """
                Task: Final Goodbye.
                1. Respond positively to the user's answer.
                2. Say goodbye. (System will append the code).
                """
        
        return " | ".join(context_parts)

# ==========================================
# APPLICATION CONTROLLER
# ==========================================

class SimulationApp:
    def __init__(self):
        self.db = DatabaseService()
        self.ai = AIService()
        self._init_session()
    
    def _init_session(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.session_state.state = ConversationState()
            st.session_state.finish_code = self._generate_code()
            st.session_state.simulation_complete = False
            
    def _generate_code(self):
        while True:
            code = str(random.randint(FINISH_CODE_MIN, FINISH_CODE_MAX))
            if self.db.is_finish_code_unique(code): return code
    
    def run(self):
        self._render_ui()
        self._handle_initial_message()
        self._render_chat_history()
        self._handle_user_input()
    
    def _render_ui(self):
        st.set_page_config(page_title="Future Simulation", page_icon="🧘", layout="centered")
        st.markdown("<style>#MainMenu, footer, header {visibility: hidden;}</style>", unsafe_allow_html=True)
        st.title("🧘 Future Simulation")
    
    def _handle_initial_message(self):
        if not st.session_state.messages:
            msg = "Hello! I’m here to help you simulate your future retirement. Ready to look ahead?"
            st.session_state.messages.append(Message(role="assistant", content=msg))
            st.session_state.state.stage = Stage.INTRODUCTION
    
    def _render_chat_history(self):
        for msg in st.session_state.messages:
            with st.chat_message(msg.role, avatar="🤖" if msg.role == "assistant" else "👤"):
                st.markdown(msg.content)
    
    def _handle_user_input(self):
        if st.session_state.simulation_complete:
            st.success(f"✅ Simulation complete! Code: **{st.session_state.finish_code}**")
            return
        
        user_input = st.chat_input("Type here...")
        if not user_input: return
        
        st.session_state.messages.append(Message(role="user", content=user_input))
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        
        # 1. AI 생성 전 상태 업데이트 (Trigger)
        self._check_stage_progression_pre_ai(user_input)
        
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                response = self.ai.generate_response(st.session_state.messages, st.session_state.state.stage, st.session_state.state)
                
                # [핵심] 마지막 Stage의 마지막 턴(턴 2)이 끝나면 무조건 코드 부착
                # AI가 방금 Turn 2에 해당하는 답변(작별인사)을 생성했으므로, 여기에 코드를 붙임
                if st.session_state.state.stage == Stage.CALL_TO_ACTION and st.session_state.state.stage_turn_count >= 2:
                    response += f"\n\n---\n\n✅ **Your finish code is: {st.session_state.finish_code}**\n\nPlease save this code to continue."
                    st.session_state.simulation_complete = True
                    self.db.save_conversation(st.session_state.finish_code, st.session_state.messages)
                    st.session_state.state.stage = Stage.COMPLETE # 완료 처리
                
                st.markdown(response)
                st.session_state.messages.append(Message(role="assistant", content=response))
                
                # 턴 카운트 증가
                st.session_state.state.advance_turn()

    def _check_stage_progression_pre_ai(self, user_input: str):
        """
        AI가 말하기 '직전'에 사용자의 입력이나 상황을 보고 스테이지를 넘길지 결정
        """
        state = st.session_state.state
        msg = user_input.lower()
        
        # Intro -> Small Talk
        if state.stage == Stage.INTRODUCTION and any(w in msg for w in ["yes", "ok", "ready"]):
            state.advance_stage()
            
        # Small Talk -> Simulation
        elif state.stage == Stage.SMALL_TALK:
            state.check_user_message_for_topics(user_input)
            if len(state.small_talk_topics_covered) >= 4:
                state.advance_stage()
        
        # Simulation -> Pre-experience
        elif state.stage == Stage.SIMULATION:
            if state.check_for_i_am_phrase(user_input):
                state.advance_stage()
                
        # Pre-experience -> Call to Action (5턴 지나면)
        elif state.stage == Stage.PRE_EXPERIENCE and state.stage_4_turns >= 5:
            state.advance_stage()
            
        # [핵심] Call to Action 내부는 자동으로 advance_turn()에 의해 턴이 올라가므로
        # 별도의 Trigger 조건이 필요 없음. (무조건 순서대로 Turn 0 -> Turn 1 -> Turn 2 진행)

if __name__ == "__main__":
    app = SimulationApp()
    app.run()
