"""
Child Context Manager
Manages screening sessions, conversation history, and symptom severity tracking
"""

from datetime import datetime
from typing import Dict, List, Optional
import uuid

class ChildContextManager:
    """Manages child screening context and conversation history"""
    
    def __init__(self):
        self.sessions = {}  # In-memory storage (use Redis/DB for production)
    
    def create_session(self, initial_data: Dict) -> str:
        """Create new session with initial screening data"""
        session_id = str(uuid.uuid4())
        
        self.sessions[session_id] = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "child_profile": {
                "age": initial_data.get("age"),
                "sex": initial_data.get("sex", "Not specified"),
                "urban_rural": initial_data.get("urban_rural", "Not specified"),
                "siblings_asd": initial_data.get("siblings_asd") == '1',
                "speech_delay": initial_data.get("speech_delay") == '1',
                "parental_concern": initial_data.get("parental_concern", "Not specified")
            },
            "initial_screening": {
                "timestamp": datetime.now().isoformat(),
                "questions": {
                    f"Q{i}": int(initial_data.get(f"Q{i}", 0))
                    for i in range(1, 31)
                },
                "total_score": sum(int(initial_data.get(f"Q{i}", 0)) for i in range(1, 31)),
                "result": None  # Will be filled after prediction
            },
            "symptom_severity": {},
            "conversation_history": [],
            "clarification_needed": [],
            "ai_insights": {},
            
            # NEW: Conversation state with follow-up tracking
            "conversation_state": {
                "mode": "follow_up",  # "follow_up" or "qa"
                "total_followups_asked": 0,
                "max_followups": 12,  # Ask up to 12 follow-ups
                "questions_per_area": {
                    "social_communication": 0,
                    "verbal_nonverbal_communication": 0,
                    "behaviour_routine": 0,
                    "sensory_processing": 0,
                    "motor_skills": 0,
                    "emotional_understanding": 0
                },
                "suggested_followups": [],  # Current suggested follow-ups
                "last_clicked_followup": None,
                "last_clicked_area": None
            },
            
            # NEW: Extracted insights (updated after each user response)
            "child_insights": {
                "key_behaviors": [],  # Observable behaviors
                "specific_challenges": [],  # Difficulties faced
                "strengths": [],  # Positive traits
                "triggers": [],  # Things that cause distress
                "preferences": [],  # Likes/dislikes
                "social_patterns": [],  # Social interaction patterns
                "communication_style": [],  # How child communicates
                "coping_mechanisms": []  # How child manages stress
            }
        }
        
        return session_id
    
    def update_prediction(self, session_id: str, prediction_data: Dict):
        """Update session with prediction results"""
        if session_id in self.sessions:
            self.sessions[session_id]["initial_screening"]["result"] = prediction_data
            self.sessions[session_id]["symptom_severity"] = self.analyze_symptom_severity(
                self.sessions[session_id]["initial_screening"]["questions"]
            )
    
    def analyze_symptom_severity(self, questions: Dict) -> Dict:
        """Categorize and assess symptom severity based on new 30-question structure"""
        
        # Symptom categories mapping (6 categories, 30 questions total)
        categories = {
            "social_communication": ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"],  # 6 questions
            "verbal_nonverbal_communication": ["Q7", "Q8", "Q9", "Q10", "Q11"],  # 5 questions
            "behaviour_routine": ["Q12", "Q13", "Q14", "Q15", "Q16", "Q17"],  # 6 questions
            "sensory_processing": ["Q18", "Q19", "Q20", "Q21", "Q22"],  # 5 questions
            "motor_skills": ["Q23", "Q24", "Q25", "Q26"],  # 4 questions
            "emotional_understanding": ["Q27", "Q28", "Q29", "Q30"]  # 4 questions
        }
        
        severity = {}
        for category, question_ids in categories.items():
            score = sum(questions.get(q, 0) for q in question_ids)
            total_questions = len(question_ids)
            
            # Calculate severity based on percentage of "Yes" responses
            if score >= total_questions * 0.7:
                severity[category] = "high"
            elif score >= total_questions * 0.4:
                severity[category] = "moderate"
            else:
                severity[category] = "low"
        
        return severity
    
    def add_conversation(self, session_id: str, role: str, message: str, 
                        question_type: Optional[str] = None,
                        symptom_area: Optional[str] = None):
        """Add message to conversation history"""
        if session_id in self.sessions:
            self.sessions[session_id]["conversation_history"].append({
                "timestamp": datetime.now().isoformat(),
                "role": role,
                "message": message,
                "question_type": question_type,
                "symptom_area": symptom_area
            })
    
    def get_context(self, session_id: str) -> Optional[Dict]:
        """Retrieve full session context"""
        return self.sessions.get(session_id)
    
    def get_context_summary(self, session_id: str) -> str:
        """Generate text summary of context for AI prompting"""
        context = self.get_context(session_id)
        if not context:
            return ""
        
        profile = context["child_profile"]
        screening = context["initial_screening"]
        severity = context["symptom_severity"]
        
        result_info = screening.get('result', {})
        
        summary = f"""
CHILD PROFILE:
- Age: {profile['age']} years old
- Sex: {profile['sex']}
- Residential Area: {profile['urban_rural']}
- Siblings with ASD: {'Yes' if profile['siblings_asd'] else 'No'}
- Speech Delay: {'Yes' if profile['speech_delay'] else 'No'}
- Parental Concern Level: {profile['parental_concern']}

INITIAL SCREENING RESULTS:
- Prediction: {result_info.get('prediction', 'Pending')}
- Confidence: {result_info.get('confidence', 'N/A')}
- Total Score: {sum(screening['questions'].values())}/30

SYMPTOM SEVERITY ANALYSIS:
- Social Communication: {severity.get('social_communication', 'Unknown').upper()}
- Verbal & Non-Verbal Communication: {severity.get('verbal_nonverbal_communication', 'Unknown').upper()}
- Behaviour & Routine Patterns: {severity.get('behaviour_routine', 'Unknown').upper()}
- Sensory Processing: {severity.get('sensory_processing', 'Unknown').upper()}
- Motor Skills: {severity.get('motor_skills', 'Unknown').upper()}
- Emotional Understanding & Social Behaviour: {severity.get('emotional_understanding', 'Unknown').upper()}

EXTRACTED CHILD INSIGHTS:
{self.get_child_insights_summary(session_id)}

CONVERSATION HISTORY ({len(context['conversation_history'])} messages):
"""
        # Include last 5 messages for context
        for msg in context["conversation_history"][-5:]:
            summary += f"\n{msg['role'].upper()}: {msg['message'][:150]}..."
        
        return summary
    
    def get_conversation_count(self, session_id: str) -> int:
        """Get number of conversation messages"""
        context = self.get_context(session_id)
        return len(context["conversation_history"]) if context else 0
    
    def session_exists(self, session_id: str) -> bool:
        """Check if session exists"""
        return session_id in self.sessions
    
    def set_suggested_followups(self, session_id: str, followups: List[Dict]):
        """Set the current suggested follow-up questions"""
        if session_id in self.sessions:
            self.sessions[session_id]["conversation_state"]["suggested_followups"] = followups
    
    def mark_followup_clicked(self, session_id: str, followup_text: str, symptom_area: str):
        """Mark that user clicked on a suggested follow-up"""
        if session_id in self.sessions:
            state = self.sessions[session_id]["conversation_state"]
            state["last_clicked_followup"] = followup_text
            state["last_clicked_area"] = symptom_area
            state["questions_per_area"][symptom_area] = state["questions_per_area"].get(symptom_area, 0) + 1
            state["total_followups_asked"] += 1
            
            # Check if we should switch to Q&A mode
            if state["total_followups_asked"] >= state["max_followups"]:
                state["mode"] = "qa"
    
    def should_suggest_followups(self, session_id: str) -> bool:
        """Determine if we should still suggest follow-ups"""
        if session_id not in self.sessions:
            return False
        
        state = self.sessions[session_id]["conversation_state"]
        return state["total_followups_asked"] < state["max_followups"]
    
    def add_child_insight(self, session_id: str, category: str, insight: str):
        """Add extracted insight about the child"""
        if session_id in self.sessions and category in self.sessions[session_id]["child_insights"]:
            insights_list = self.sessions[session_id]["child_insights"][category]
            # Avoid duplicates
            if insight not in insights_list:
                insights_list.append(insight)
    
    def get_child_insights_summary(self, session_id: str) -> str:
        """Get formatted summary of child insights"""
        if session_id not in self.sessions:
            return ""
        
        insights = self.sessions[session_id]["child_insights"]
        summary_parts = []
        
        for category, items in insights.items():
            if items:
                category_name = category.replace('_', ' ').title()
                summary_parts.append(f"{category_name}: {'; '.join(items)}")
        
        if not summary_parts:
            return "No specific insights extracted yet."
        
        return "\n".join(summary_parts)
