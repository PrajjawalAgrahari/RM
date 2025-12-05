"""
Follow-up Question Generator
Generates mixed-domain follow-up questions based on symptom severity
"""

from typing import List, Dict, Tuple
import random


class FollowUpGenerator:
    """Generates intelligent follow-up questions balanced across all symptom domains"""
    
    # Comprehensive question bank organized by domain and severity (6 categories)
    QUESTIONS = {
        "social_communication": {
            "high": [
                "Can you describe a recent situation where your child avoided making eye contact during a conversation?",
                "How does your child typically respond when peers try to talk to them?",
                "Does your child initiate conversations or wait for others to start?",
                "How does your child handle taking turns in conversations or games?",
                "Does your child understand sarcasm, jokes, or figures of speech?"
            ],
            "moderate": [
                "How often does your child initiate conversations with family members?",
                "Does your child prefer playing alone or with other children?",
                "How does your child react when someone doesn't understand what they're saying?"
            ],
            "low": [
                "Does your child enjoy talking about their favorite topics?",
                "How does your child communicate when they need help?"
            ]
        },
        "verbal_nonverbal_communication": {
            "high": [
                "Does your child use gestures like pointing or waving when communicating?",
                "Can you give examples of when your child repeats phrases or sentences?",
                "How does your child express their feelings or needs when they're upset?",
                "Does your child speak in a monotone voice or have an unusual speech rhythm?",
                "How does your child handle multi-step instructions?"
            ],
            "moderate": [
                "How well does your child explain what they want or need?",
                "Does your child combine verbal and non-verbal communication effectively?",
                "How does your child respond to complex verbal instructions?"
            ],
            "low": [
                "How does your child typically communicate their daily needs?",
                "Does your child use facial expressions appropriately?"
            ]
        },
        "behaviour_routine": {
            "high": [
                "How does your child react when their daily routine is changed unexpectedly?",
                "Does your child insist on doing tasks in a very specific order or way?",
                "Can you describe any repetitive movements your child makes (hand-flapping, rocking, spinning)?",
                "What topics or objects is your child particularly fixated on?",
                "Does your child line up toys or arrange objects in specific patterns?"
            ],
            "moderate": [
                "How flexible is your child when it comes to changes in routine?",
                "Does your child have specific rituals or routines they must follow?",
                "How does your child react when interrupted during a focused activity?"
            ],
            "low": [
                "Does your child have any preferred routines or activities?",
                "How easily can your child transition between activities?"
            ]
        },
        "sensory_processing": {
            "high": [
                "What specific sounds, lights, or textures cause strong reactions in your child?",
                "Does your child frequently cover their ears even when sounds seem normal to others?",
                "Are there certain clothing textures or tags that your child refuses to wear?",
                "Does your child seek intense sensory experiences like jumping, spinning, or crashing?",
                "Are there specific food textures or smells that your child strongly avoids?"
            ],
            "moderate": [
                "How does your child handle crowded or noisy environments?",
                "Does your child notice sensory details that others might miss?",
                "Are there activities your child avoids due to sensory discomfort?"
            ],
            "low": [
                "Does your child have any specific sensory preferences?",
                "How does your child react to trying new foods or textures?"
            ]
        },
        "motor_skills": {
            "high": [
                "What fine motor tasks does your child find particularly challenging (buttoning, writing, scissors)?",
                "How would you describe your child's coordination compared to peers of the same age?",
                "Can you describe any delays in motor milestones (walking, running, climbing)?",
                "What unusual motor movements does your child display (finger flicking, pacing)?"
            ],
            "moderate": [
                "How well does your child perform age-appropriate motor tasks?",
                "Does your child have difficulty with activities requiring hand-eye coordination?",
                "How is your child's balance and body awareness?"
            ],
            "low": [
                "Are there any motor activities your child particularly enjoys or avoids?",
                "How does your child perform with everyday motor tasks?"
            ]
        },
        "emotional_understanding": {
            "high": [
                "Can your child identify basic emotions in others by looking at their faces?",
                "Does your child understand when someone else is upset or hurt without being told?",
                "How does your child respond to minor changes or small frustrations?",
                "Does your child accept or avoid physical affection like hugs?",
                "How easy is it for your child to make and maintain friendships?"
            ],
            "moderate": [
                "Does your child pick up on subtle emotional cues from others?",
                "How does your child handle social conflicts or disagreements?",
                "Can your child understand other people's perspectives?"
            ],
            "low": [
                "How does your child express their own emotions?",
                "Does your child show empathy when others are upset?"
            ]
        }
    }
    
    def __init__(self):
        self.asked_questions = {}  # Track questions per session
    
    def generate_mixed_followups(self, session_id: str, symptom_severity: Dict, 
                                 questions_per_area: Dict, num_suggestions: int = 4) -> List[Dict]:
        """
        Generate mixed follow-up questions across all domains
        Prioritizes high-severity areas but ensures variety
        
        Args:
            session_id: Session identifier
            symptom_severity: Dict mapping symptom areas to severity levels
            questions_per_area: Dict tracking how many questions asked per area
            num_suggestions: Number of questions to generate
            
        Returns:
            List of dicts with question, symptom_area, severity, priority
        """
        if session_id not in self.asked_questions:
            self.asked_questions[session_id] = set()
        
        # Create priority list: (area, severity_level, priority_score)
        priority_areas = []
        for area, severity in symptom_severity.items():
            if severity in ["high", "moderate"]:
                asked_count = questions_per_area.get(area, 0)
                target_count = 4 if severity == "high" else 2
                
                if asked_count < target_count:
                    # Higher severity and fewer questions = higher priority
                    severity_weight = 10 if severity == "high" else 5
                    balance_weight = (target_count - asked_count) * 2
                    priority_score = severity_weight + balance_weight
                    
                    priority_areas.append((area, severity, priority_score, asked_count, target_count))
        
        # Sort by priority (highest first)
        priority_areas.sort(key=lambda x: x[2], reverse=True)
        
        # Generate questions with domain mixing
        suggested_followups = []
        used_areas = set()
        
        for area, severity, priority, asked, target in priority_areas:
            if len(suggested_followups) >= num_suggestions:
                break
            
            # Get available questions for this area
            available_questions = self.QUESTIONS.get(area, {}).get(severity, [])
            
            # Filter out already-asked questions
            available_questions = [
                q for q in available_questions 
                if f"{area}:{q}" not in self.asked_questions[session_id]
            ]
            
            if available_questions:
                # For high priority area, include 2 questions; otherwise 1
                num_from_area = 2 if priority >= 15 and len(suggested_followups) < num_suggestions - 1 else 1
                
                selected = random.sample(available_questions, min(num_from_area, len(available_questions)))
                
                for question in selected:
                    suggested_followups.append({
                        "question": question,
                        "symptom_area": area,
                        "severity": severity,
                        "priority": priority
                    })
                    
                    # Mark as asked
                    self.asked_questions[session_id].add(f"{area}:{question}")
                
                used_areas.add(area)
        
        # If we don't have enough questions, add from low-severity areas
        if len(suggested_followups) < num_suggestions:
            for area, severity in symptom_severity.items():
                if len(suggested_followups) >= num_suggestions:
                    break
                
                if area not in used_areas and severity == "low":
                    available_questions = self.QUESTIONS.get(area, {}).get(severity, [])
                    available_questions = [
                        q for q in available_questions 
                        if f"{area}:{q}" not in self.asked_questions[session_id]
                    ]
                    
                    if available_questions:
                        question = random.choice(available_questions)
                        suggested_followups.append({
                            "question": question,
                            "symptom_area": area,
                            "severity": severity,
                            "priority": 1
                        })
                        self.asked_questions[session_id].add(f"{area}:{question}")
        
        return suggested_followups[:num_suggestions]
    
    def clear_session(self, session_id: str):
        """Clear asked questions for a session"""
        if session_id in self.asked_questions:
            del self.asked_questions[session_id]
