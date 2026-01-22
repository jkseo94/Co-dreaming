import streamlit as st
from openai import OpenAI
from supabase import create_client
from datetime import datetime
import random
import time

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================

SYSTEM_PROMPT_CONTENT = """
Role: You are an AI agent designed to help users generate simulations or a subset of future-oriented thinking that involves imaginatively placing oneself in a hypothetical scenario. Your ultimate purpose is to help users mentally pre-experience the future and decision-making in intertemporal choices, such as saving.
Constraints:
- Make sure each conversation thread is less than 50 words.
- Tone: Friendly, realistic

Dialogue Steps:
Follow this sequence strictly. Do not skip steps.
1. Stage 1 — Introduction (don't show this): 
- Introduce yourself briefly as follows: "Hello! I’d like to invite you to a short session designed to think about retirement. It can sometimes feel very distant, but exploring it now helps clarify what matters to you. 
Are you ready?" 

2. Stage 2 — Small Talk (don't show this): 
- Smoothly transition from Turn 1 to Turn 2 by letting users know that you are trying to get to know them better.
- THEN ask all of the following questions one by one to get to know the user better: "How old are you right now?," "How do you describe your gender?," "How many family members do you currently have?" 

3. Stage 3 — Simulation (don't show this): 
- Guide users to mentally pre-experience what life would be like after years of saving. 
- You should help users think about what life would be like, having retired in a financially prepared state through years of saving. THEN ask them to think about a specific future event that could occur then, using the phrase “I am” as if they are there right now (e.g., “I am at the beach swimming” or “I am making coffee”).

4. Stage 4 — Pre-experience (don't show this): 
- Please help users simulate more vividly and in detailed and specific ways. 
- Ask them where they are, who they are with, and what they will be hearing and seeing at that future event they are asked to think about in Turn 3, one by one.
- Please further expand this stage 4 to actively facilitate users mentally pre-experiencing the future event.
- Break down questions into separate turns. Do not ask everything at once.
- Ensure this stage lasts for a minimum of 5 turns and a maximum of 7 turns.

5. Stage 5 — Call to Action (don't show this): 
- Recap and synthesize the future event that the user has constructed with you during the conversation. Then ask the user how they feel about it.
- End on a hopeful note that it is not always easy to think so far ahead, but doing so is a great step toward better financial preparedness. 
- Suggest to users that saving now can help them reach a financially prepared retirement in the future, such as "Your tomorrow is built on what you do today. Why not invest in a brighter future by saving a small amount for your retirement now?"
- Please send the final message, "I hope this short conversation provided you with a meaningful perspective on your retirement."

Concluding Remarks: 
Once the users want to end the conversation after going through all five turns, provide them with a randomized finish code to proceed with the survey questionnaire.
"""

# AI가 파이썬 코드와 통신하기 위한 제어 지침 (수정됨: 연결성 강화)
SYSTEM_CONTROL_INSTRUCTIONS = """
[SYSTEM CONTROL INSTRUCTIONS - CRITICAL]
You are interacting with a Python script that controls the Stage number based on your output.

Rules for "Step Transitions":
1. IF you have NOT completed the current stage's goals:
   - Respond normally. DO NOT use [[NEXT]].

2. IF you HAVE completed the current stage's goals and are ready to move on:
   - **CRITICAL:** You MUST **combine** the closing of the current stage AND the **opening question/intro** of the NEXT stage in the SAME response.
   - Append `[[NEXT]]` at the very end.

Example (Stage 2 -> Stage 3):
- User: "I have 3 family members." (Last question of Stage 2)
- You: "Thank you for sharing that. Now, let's try to think about... [Stage 3 Intro Question] [[NEXT]]"
(This ensures the user immediately sees the new topic without typing again.)
"""

FULL_SYSTEM_PROMPT = SYSTEM_PROMPT_CONTENT + "\n\n" + SYSTEM_CONTROL_INSTRUCTIONS

# ==========================================
# SERVICES (Model & Database)
# ==========================================

class DatabaseService:
    def __init__(self):
        try:
            self.supabase = create_client(
                st.secrets["SUPABASE_URL"],
                st.secrets["SUPABASE_SERVICE_KEY"]
            )
        except Exception as e:
            st.error(f"Database connection failed: {e}")
            self.supabase = None

    def save_full_conversation(self, finish_code, messages):
        if not self.supabase:
            return False
            
        data = {
            "finish_code": finish_code,
            "full_conversation": messages,
            "finished_at": datetime.utcnow().isoformat()
        }
        try:
            self.supabase.table("full_conversations").insert(data).execute()
            return True
        except Exception as e:
            st.error(f"Failed to save conversation: {e}")
            return False

class AIService:
    def __init__(self):
        self.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    def generate_response(self, messages, current_step):
        api_messages = [
            {"role": "system", "content": FULL_SYSTEM_PROMPT},
            {"role": "system", "content": f"[SYSTEM STATUS: You are currently in STAGE {current_step}. Follow the Dialogue Steps for Stage {current_step}. Remember: When finishing a stage, combine the closing + next stage intro + [[NEXT]] tag in one message.]"}
        ]
        
        for msg in messages:
            api_messages.append({"role": msg["role"], "content": msg["content"]})

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=api_messages
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"AI Error: {e}")
            return "Connection interrupted."

# ==========================================
# APP LOGIC (Controller)
# ==========================================

class SimulationApp:
    def __init__(self):
        self.db = DatabaseService()
        self.ai = AIService()
        self.initialize_session_state()

    def initialize_session_state(self):
        # Reset Logic
        if "reset_trigger" not in st.session_state:
            st.session_state.reset_trigger = False

        defaults = {
            "messages": [],
            "current_step": 0,
            "finish_code": str(random.randint(10000, 99999)),
            "simulation_complete": False,
            "data_saved": False
        }
        
        # Reset이 트리거되었거나 초기 상태일 때 초기화
        if st.session_state.reset_trigger:
            for key, value in defaults.items():
                st.session_state[key] = value
            st.session_state.reset_trigger = False
            st.rerun()

        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def run(self):
        self.render_ui()
        self.handle_initial_message()
        self.render_chat_history()
        self.handle_user_input()

    def render_ui(self):
        st.set_page_config(page_title="Saving for the future", layout="centered")
        st.markdown("""
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """, unsafe_allow_html=True)
        st.title("Saving for the future")
        
        # [NEW] 개발/테스트용 리셋 버튼 (사이드바)
        with st.sidebar:
            st.write("Debug Options")
            if st.button("Reset Conversation"):
                st.session_state.reset_trigger = True
                st.rerun()

    def handle_initial_message(self):
        if not st.session_state.messages:
            welcome_msg = (
                "Hello! I’d like to invite you to a short session designed to think about retirement. "
                "It can sometimes feel very distant, but exploring it now helps clarify what matters to you.\n\n"
                "Are you ready?"
            )
            st.session_state.messages.append({"role": "assistant", "content": welcome_msg})

    def render_chat_history(self):
        for msg in st.session_state.messages:
            avatar = "🤖" if msg["role"] == "assistant" else None
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

    def handle_user_input(self):
        # 시뮬레이션 종료 시
        if st.session_state.simulation_complete:
            st.info(f"Simulation ended. Your code: {st.session_state.finish_code}")
            return

        if prompt := st.chat_input("Type your message here"):
            # 1. User Message Display
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # 2. Stage Logic: Intro -> Small Talk
            if st.session_state.current_step == 0:
                st.session_state.current_step = 2
            
            # 3. Generate AI Response
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("..."):
                    raw_response = self.ai.generate_response(
                        st.session_state.messages, 
                        st.session_state.current_step
                    )

                # [[NEXT]] 태그 처리
                move_to_next_stage = False
                clean_response = raw_response

                if "[[NEXT]]" in raw_response:
                    move_to_next_stage = True
                    clean_response = raw_response.replace("[[NEXT]]", "").strip()

                # Stage 5 (종료) 처리
                if st.session_state.current_step == 5 and move_to_next_stage:
                    clean_response += f"\n\nYour finish code is **{st.session_state.finish_code}**."
                    st.session_state.simulation_complete = True
                
                st.markdown(clean_response)
                st.session_state.messages.append({"role": "assistant", "content": clean_response})

                # 태그가 있을 때만 단계 증가
                if move_to_next_stage and not st.session_state.simulation_complete:
                    st.session_state.current_step += 1

            # 4. Save Logic (완료 시 즉시 저장)
            if st.session_state.simulation_complete and not st.session_state.data_saved:
                success = self.db.save_full_conversation(
                    st.session_state.finish_code,
                    st.session_state.messages
                )
                if success:
                    st.session_state.data_saved = True
                    st.success("Conversation saved successfully!")

if __name__ == "__main__":
    app = SimulationApp()
    app.run()
