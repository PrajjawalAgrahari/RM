from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
import google.generativeai as genai
from typing import Dict, Any
import json
import faiss
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from context_manager import ChildContextManager
from followup_generator import FollowUpGenerator
from insight_extractor import InsightExtractor
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configure Gemini API
api_key = os.getenv('GEMINI_API_KEY')
if api_key:
    genai.configure(api_key=api_key)
    logger.info("✓ Gemini API configured")
else:
    logger.warning("⚠️ GEMINI_API_KEY not found")

# Load the trained model and preprocessing objects
try:
    model = joblib.load('svm_child_asd_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    logger.info("✓ ML model loaded")
except Exception as e:
    logger.error(f"❌ Model loading failed: {e}")
    model = None
    scaler = None
    label_encoders = None

# Initialize managers
context_manager = ChildContextManager()
followup_generator = FollowUpGenerator()
insight_extractor = InsightExtractor()
logger.info("✓ Managers initialized")

# Initialize RAG components
rag_initialized = False
embedding_model = None
faiss_index = None
chunk_metadata = None
chunks_with_text = []  # Store chunks with actual text content

def initialize_rag():
    """Initialize RAG pipeline components"""
    global rag_initialized, embedding_model, faiss_index, chunk_metadata, chunks_with_text
    
    if rag_initialized:
        return True
    
    try:
        logger.info("🔄 Initializing RAG components...")
        
        # Define paths - Project folder is in RM directory, not autism-screening-app
        # Path structure: RM/autism-screening-app/backend/app.py -> RM/Project/
        project_path = Path(__file__).parent.parent.parent / "Project"
        
        logger.info(f"📂 Loading from: {project_path}")
        
        # Load embedding model
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        logger.info("✓ Embedding model loaded")
        
        # Load FAISS index
        faiss_index = faiss.read_index(str(project_path / 'faiss_index.bin'))
        logger.info(f"✓ FAISS index loaded ({faiss_index.ntotal} vectors)")
        
        # Load chunk metadata
        with open(project_path / 'chunk_metadata.json', 'r', encoding='utf-8') as f:
            chunk_metadata = json.load(f)
        
        # Load actual chunks with text from JSONL
        with open(project_path / 'processed_chunks.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                chunk = json.loads(line.strip())
                chunks_with_text.append(chunk)
        
        logger.info(f"✓ Loaded {len(chunks_with_text)} chunks with text")
        
        rag_initialized = True
        return True
        
    except Exception as e:
        logger.error(f"❌ RAG initialization failed: {e}")
        return False

def rag_search(query: str, top_k: int = 3) -> tuple:
    """
    Search vector database for relevant chunks using RAG
    
    Args:
        query: User query text
        top_k: Number of results to return
    
    Returns:
        tuple: (scores, retrieved_chunks_text)
    """
    try:
        if not rag_initialized:
            if not initialize_rag():
                return [], ""
        
        # Generate query embedding
        query_embedding = embedding_model.encode([query], normalize_embeddings=True)
        
        # Search FAISS index
        scores, indices = faiss_index.search(query_embedding.astype('float32'), top_k)
        
        # Retrieve chunk texts
        retrieved_texts = []
        for idx in indices[0]:
            if idx < len(chunks_with_text):
                chunk_text = chunks_with_text[idx].get('text', '')
                if chunk_text:
                    retrieved_texts.append(chunk_text)
        
        # Combine context
        context = "\n\n".join([f"Context {i+1}:\n{text[:800]}..." if len(text) > 800 else f"Context {i+1}:\n{text}" 
                              for i, text in enumerate(retrieved_texts)])
        
        return scores[0].tolist(), context
        
    except Exception as e:
        logger.warning(f"⚠️ RAG search failed: {e}")
        return [], ""

def get_gemini_insights(prediction_data: Dict[str, Any], form_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Get AI-powered insights using Gemini for autism screening results
    """
    try:
        # Check if API key is available
        if not os.getenv('GEMINI_API_KEY'):
            print("No Gemini API key found, using fallback insights")
            raise ValueError("No API key configured")
        
        # Initialize Gemini model
        print("Initializing Gemini model...")
        model_gemini = genai.GenerativeModel('gemini-2.5-flash')
        print("Gemini model initialized successfully")
        
        # Prepare the prompt with screening data
        prompt = f"""
        You are a clinical AI assistant specializing in autism spectrum disorder (ASD) assessment. 
        Based on the following autism screening results for a child, provide professional insights:

        SCREENING RESULTS:
        - Prediction: {prediction_data['prediction']}
        - Confidence: {prediction_data['confidence']}
        - ASD Probability: {prediction_data['probability_asd']}
        - No ASD Probability: {prediction_data['probability_no_asd']}

        CHILD DEMOGRAPHICS:
        - Age: {form_data['age']} years
        - Gender: {'Male' if form_data['gender'] == 1 else 'Female'}
        - Born with jaundice: {'Yes' if form_data['jundice'] == 1 else 'No'}
        - Family history of ASD: {'Yes' if form_data['austim'] == 1 else 'No'}

        SCREENING QUESTIONS RESPONSES:
        - Q1 (Notices small sounds): {'Yes' if form_data['A1_Score'] == 1 else 'No'}
        - Q2 (Concentrates on whole picture): {'Yes' if form_data['A2_Score'] == 1 else 'No'}
        - Q3 (Keeps track of conversations): {'Yes' if form_data['A3_Score'] == 1 else 'No'}
        - Q4 (Easy to switch activities): {'Yes' if form_data['A4_Score'] == 1 else 'No'}
        - Q5 (Doesn't know how to keep conversation): {'Yes' if form_data['A5_Score'] == 1 else 'No'}
        - Q6 (Good at social chit-chat): {'Yes' if form_data['A6_Score'] == 1 else 'No'}
        - Q7 (Difficult to work out character intentions): {'Yes' if form_data['A7_Score'] == 1 else 'No'}
        - Q8 (Enjoyed pretend play in preschool): {'Yes' if form_data['A8_Score'] == 1 else 'No'}
        - Q9 (Easy to work out emotions from faces): {'Yes' if form_data['A9_Score'] == 1 else 'No'}
        - Q10 (Hard to make new friends): {'Yes' if form_data['A10_Score'] == 1 else 'No'}

        Please provide:
        1. SEVERITY ASSESSMENT: Based on the screening results, assess the severity level (Low Risk, Moderate Risk, High Risk)
        2. KEY FINDINGS: Highlight the most significant behavioral indicators from the responses
        3. RECOMMENDATIONS: Provide specific, actionable next steps for parents/caregivers
        4. FOLLOW-UP: Suggest appropriate professional consultations or further assessments

        Format your response as a JSON with the following keys:
        - severity: string (Low Risk/Moderate Risk/High Risk)
        - key_findings: string (2-3 bullet points)
        - recommendations: string (3-4 specific actionable recommendations)
        - follow_up: string (professional consultation suggestions)

        Keep responses professional, compassionate, and evidence-based. Emphasize that this is a screening tool and professional evaluation is needed for diagnosis.
        """

        print("Sending request to Gemini...")
        response = model_gemini.generate_content(prompt)
        print("Received response from Gemini")
        
        if not response or not response.text:
            print("Empty response from Gemini")
            raise ValueError("Empty response from Gemini")
            
        response_text = response.text.strip()
        print("Full response text:", response_text)
        print("Response length:", len(response_text))
        
        # Try to extract JSON, if not possible, create structured response
        try:
            import json
            # Clean the response to extract JSON
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                print("Extracted JSON string:", json_str)
                insights = json.loads(json_str)
                print("Successfully parsed JSON:", insights)
            else:
                print("No JSON braces found in response")
                raise ValueError("No JSON found")
        except Exception as parse_error:
            # Fallback: parse manually or create default structure
            print(f"JSON parsing failed: {parse_error}")
            print("Using fallback insights")
            insights = {
                "severity": "Moderate Risk" if "ASD Traits Detected" in prediction_data['prediction'] else "Low Risk",
                "key_findings": response_text[:200] + "..." if len(response_text) > 200 else response_text,
                "recommendations": "Consult with a pediatric developmental specialist for comprehensive evaluation.",
                "follow_up": "Schedule appointment with healthcare provider for further assessment."
            }
        
        return insights
        
    except Exception as e:
        print(f"Error getting Gemini insights: {e}")
        # Fallback insights based on basic rules
        return {
            "severity": "Moderate Risk" if "ASD Traits Detected" in prediction_data['prediction'] else "Low Risk",
            "key_findings": "Based on screening responses, further evaluation recommended.",
            "recommendations": "Consult with a healthcare professional for comprehensive developmental assessment.",
            "follow_up": "Contact your pediatrician or a developmental specialist for detailed evaluation."
        }

@app.route('/')
def index():
    return jsonify({"message": "Autism Screening API with AI Insights", "status": "running"})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded properly'}), 500

    try:
        # Get form data
        data = request.json
        
        logger.info(f"\n{'='*60}")
        logger.info("📋 NEW SCREENING REQUEST")
        
        # Create session BEFORE prediction
        session_id = context_manager.create_session(data)
        logger.info(f"✓ Session created: {session_id[:8]}...")
        
        # Extract features in the correct order
        features = [
            float(data.get('A1_Score', 0)),
            float(data.get('A2_Score', 0)),
            float(data.get('A3_Score', 0)),
            float(data.get('A4_Score', 0)),
            float(data.get('A5_Score', 0)),
            float(data.get('A6_Score', 0)),
            float(data.get('A7_Score', 0)),
            float(data.get('A8_Score', 0)),
            float(data.get('A9_Score', 0)),
            float(data.get('A10_Score', 0)),
            float(data.get('age', 0)),
            float(data.get('gender', 0)),  # Assuming encoded: 0=Female, 1=Male
            float(data.get('jundice', 0)),  # 0=No, 1=Yes
            float(data.get('austim', 0)),   # 0=No, 1=Yes
            float(data.get('used_app_before', 0)),  # 0=No, 1=Yes
            float(data.get('result', 0))    # Sum of A1-A10 scores
        ]
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)

        # Scale features
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        
        # Convert prediction to readable format
        result = "ASD Traits Detected" if prediction == 1 else "No ASD Traits Detected"
        confidence = max(prediction_proba) * 100
        
        prediction_data = {
            'prediction': result,
            'confidence': round(confidence, 2),
            'probability_asd': round(prediction_proba[1]*100, 2),
            'probability_no_asd': round(prediction_proba[0]*100, 2)
        }
        
        logger.info(f"✓ Prediction: {result} ({confidence:.1f}%)")
        
        # Update session with prediction results
        context_manager.update_prediction(session_id, prediction_data)
        
        # Get AI insights using Gemini
        ai_insights = get_gemini_insights(prediction_data, data)
        
        # Get symptom severity and generate mixed follow-up questions
        session_context = context_manager.get_context(session_id)
        symptom_severity = session_context["symptom_severity"]
        
        # Generate mixed follow-ups across domains
        suggested_followups = followup_generator.generate_mixed_followups(
            session_id,
            symptom_severity,
            session_context["conversation_state"]["questions_per_area"],
            num_suggestions=4
        )
        
        context_manager.set_suggested_followups(session_id, suggested_followups)
        
        logger.info(f"✓ Generated {len(suggested_followups)} mixed follow-ups")
        logger.info(f"   Domains: {[f['symptom_area'] for f in suggested_followups]}")
        logger.info(f"{'='*60}\n")
        
        # Combine prediction results with AI insights
        response_data = {
            **prediction_data,
            'ai_insights': ai_insights,
            'session_id': session_id,
            'symptom_severity': symptom_severity,
            'suggested_followups': suggested_followups,
            'followup_mode': True
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 400

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'gemini_configured': os.getenv('GEMINI_API_KEY') is not None
    })

@app.route('/chat', methods=['POST'])
def chat():
    """
    Context-aware chat with suggested follow-ups and insight extraction
    """
    try:
        data = request.json
        user_query = data.get('query', '').strip()
        session_id = data.get('session_id', '')
        clicked_followup = data.get('clicked_followup')  # NEW: Which follow-up was clicked
        
        # Log incoming request payload
        logger.info(f"Incoming request payload: {data}")

        # Validate required fields
        if not user_query or not session_id:
            logger.error("Missing 'query' or 'session_id' in request payload.")
            return jsonify({'error': 'Query and session_id required'}), 400

        if not context_manager.session_exists(session_id):
            logger.error(f"Invalid session_id: {session_id}")
            return jsonify({'error': 'Invalid session'}), 400
        
        logger.info(f"\n{'='*60}")
        logger.info(f"💬 USER: {user_query[:100]}...")
        
        # Get context
        session_context = context_manager.get_context(session_id)
        
        # If user clicked a suggested follow-up, mark it
        if clicked_followup:
            context_manager.mark_followup_clicked(
                session_id, 
                clicked_followup['question'],
                clicked_followup['symptom_area']
            )
            logger.info(f"✓ Clicked follow-up: {clicked_followup['symptom_area']}")
        
        # Add user message
        context_manager.add_conversation(
            session_id, "user", user_query,
            question_type="followup_response" if clicked_followup else "general_query",
            symptom_area=clicked_followup['symptom_area'] if clicked_followup else None
        )
        
        # Extract insights from user's response
        if len(user_query) > 20:
            area = clicked_followup['symptom_area'] if clicked_followup else 'general'
            insights = insight_extractor.extract_from_response(user_query, area)
            
            # Add insights to context
            for category, items in insights.items():
                for item in items:
                    context_manager.add_child_insight(session_id, category, item)
        
        # Initialize RAG
        if not rag_initialized:
            initialize_rag()
        
        # RAG search
        scores, rag_context = rag_search(user_query, top_k=3)
        if scores:
            logger.info(f"📚 RAG: {len(scores)} contexts (relevance: {scores[0]:.3f})")
        
        # Build comprehensive context
        context_summary = context_manager.get_context_summary(session_id)
        child_insights = context_manager.get_child_insights_summary(session_id)
        
        # Log child insights
        if child_insights and child_insights != "No specific insights extracted yet.":
            logger.info(f"\n{'='*60}")
            logger.info("📋 CHILD INSIGHTS:")
            logger.info(f"{'-'*60}")
            logger.info(child_insights)
            logger.info(f"{'='*60}\n")
        
        # Build prompt
        prompt = f"""You are a compassionate autism assessment assistant.

{context_summary}

EXTRACTED INSIGHTS ABOUT CHILD:
{child_insights}

RESEARCH CONTEXT:
{rag_context if rag_context else "No specific research context available."}

USER'S QUESTION/RESPONSE: "{user_query}"

Provide a helpful, empathetic response that:
1. Acknowledges their input
2. Provides evidence-based information from research
3. Relates to the child's specific situation
4. Is clear and actionable

Keep response concise (2-3 paragraphs max).

RESPONSE:"""
        
        # Log prompt (truncated)
        logger.info(f"\n{'='*60}")
        logger.info("📝 PROMPT TO GEMINI:")
        logger.info(f"{'-'*60}")
        logger.info(prompt[:600] + "..." if len(prompt) > 600 else prompt)
        logger.info(f"{'='*60}\n")
        
        # Generate response
        model_gemini = genai.GenerativeModel('gemini-2.5-flash')
        response = model_gemini.generate_content(prompt)
        
        if not response or not response.text:
            raise ValueError("Empty response")
        
        assistant_message = response.text.strip()
        logger.info(f"🤖 ASSISTANT: {assistant_message[:120]}...")
        
        # Add assistant response
        context_manager.add_conversation(session_id, "assistant", assistant_message)
        
        # Generate NEW suggested follow-ups (mixed domains)
        should_suggest = context_manager.should_suggest_followups(session_id)
        
        new_followups = []
        if should_suggest:
            new_followups = followup_generator.generate_mixed_followups(
                session_id,
                session_context["symptom_severity"],
                session_context["conversation_state"]["questions_per_area"],
                num_suggestions=4
            )
            context_manager.set_suggested_followups(session_id, new_followups)
            
            if new_followups:
                domains = [f['symptom_area'] for f in new_followups]
                logger.info(f"✓ Generated {len(new_followups)} new follow-ups")
                logger.info(f"   Domains: {domains}")
        else:
            logger.info("✓ Follow-up limit reached - Q&A mode")
        
        state = session_context["conversation_state"]
        logger.info(f"📊 Progress: {state['total_followups_asked']}/{state['max_followups']}")
        logger.info(f"{'='*60}\n")
        
        return jsonify({
            'response': assistant_message,
            'session_id': session_id,
            'suggested_followups': new_followups,
            'followup_mode': should_suggest,
            'progress': {
                'answered': state['total_followups_asked'],
                'total': state['max_followups'],
                'percentage': int((state['total_followups_asked'] / state['max_followups']) * 100)
            },
            'questions_per_area': state['questions_per_area']
        })
        
    except Exception as e:
        logger.error(f"❌ Chat error: {e}", exc_info=True)
        return jsonify({'error': 'Failed to process query'}), 500

def chat_without_rag(user_query: str, context: Dict) -> Any:
    """
    Fallback chat function without RAG
    """
    try:
        model_gemini = genai.GenerativeModel('gemini-2.5-flash')
        
        context_info = ""
        if context:
            context_info = f"""
            Context from previous screening:
            - Prediction: {context.get('prediction', 'N/A')}
            - Child Age: {context.get('age', 'N/A')}
            - Gender: {context.get('gender', 'N/A')}
            """
        
        prompt = f"""
        You are a compassionate and knowledgeable AI assistant specializing in autism spectrum disorder (ASD) 
        and child development. Answer the following question professionally and empathetically.
        
        {context_info}
        
        User Question: {user_query}
        
        Provide a clear, helpful, and evidence-based response. If the question is about diagnosis or medical advice, 
        remind the user to consult with healthcare professionals. Keep the response concise but informative.
        """
        
        response = model_gemini.generate_content(prompt)
        
        if not response or not response.text:
            raise ValueError("Empty response from AI")
        
        return jsonify({
            'response': response.text.strip(),
            'rag_enabled': False
        })
        
    except Exception as e:
        print(f"Error in fallback chat: {e}")
        raise

if __name__ == '__main__':
    logger.info("🚀 Starting Autism Screening API...")
    app.run(debug=True, host='0.0.0.0', port=5000)