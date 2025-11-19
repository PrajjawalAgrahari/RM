"""
Insight Extractor
Extracts key behavioral insights from user responses using Gemini
"""

import re
import json
from typing import Dict
import google.generativeai as genai
import logging

logger = logging.getLogger(__name__)


class InsightExtractor:
    """Extracts structured insights from user responses using AI"""
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.5-flash')
    
    def extract_from_response(self, user_response: str, symptom_area: str) -> Dict:
        """
        Extract structured insights from user's response
        
        Args:
            user_response: The user's text response
            symptom_area: The symptom area being discussed
            
        Returns:
            Dict with categories and extracted short phrases
        """
        try:
            prompt = f"""You are analyzing a parent's response about their child's behavior. Extract KEY INSIGHTS in SHORT PHRASES (2-5 words each).

Symptom Area: {symptom_area.replace('_', ' ').title()}
Parent's Response: "{user_response}"

Extract insights into these categories:
- key_behaviors: Observable specific behaviors
- specific_challenges: Difficulties or struggles
- strengths: Positive abilities or traits
- triggers: Things that cause distress
- preferences: Likes, dislikes, or patterns
- social_patterns: Social interaction styles
- communication_style: How child communicates
- coping_mechanisms: How child manages stress

Return ONLY valid JSON with SHORT PHRASES (max 5 words each):
{{
  "key_behaviors": ["phrase1", "phrase2"],
  "specific_challenges": ["phrase1"],
  "strengths": [],
  "triggers": [],
  "preferences": ["phrase1"],
  "social_patterns": [],
  "communication_style": [],
  "coping_mechanisms": []
}}

Only include categories with content. Be specific and concise.

JSON:"""
            
            response = self.model.generate_content(prompt)
            
            if response and response.text:
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                if json_match:
                    insights = json.loads(json_match.group())
                    
                    # Validate structure
                    valid_insights = {}
                    for category in ['key_behaviors', 'specific_challenges', 'strengths', 
                                   'triggers', 'preferences', 'social_patterns', 
                                   'communication_style', 'coping_mechanisms']:
                        if category in insights and isinstance(insights[category], list):
                            valid_insights[category] = insights[category]
                    
                    logger.info(f"✓ Extracted {sum(len(v) for v in valid_insights.values())} insights")
                    return valid_insights
            
            return {}
            
        except Exception as e:
            logger.warning(f"⚠️ Insight extraction failed: {e}")
            return {}
