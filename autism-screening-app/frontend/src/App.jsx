import { useState } from 'react';
import axios from 'axios';
import './App.css';

const QUESTION_CATEGORIES = {
  'Social Communication': [
    { id: 'Q1', text: 'Does the child avoid making eye contact during conversations?' },
    { id: 'Q2', text: 'Does the child struggle to understand social cues (e.g., facial expressions, tone)?' },
    { id: 'Q3', text: 'Does the child rarely initiate conversation or social interaction with peers?' },
    { id: 'Q4', text: 'Does the child prefer playing alone rather than with other children?' },
    { id: 'Q5', text: 'Does the child have difficulty maintaining turn-taking in conversations or games?' },
    { id: 'Q6', text: 'Does the child take language very literally and struggle with jokes or sarcasm?' }
  ],
  'Verbal & Non-Verbal Communication': [
    { id: 'Q7', text: 'Does the child use limited gestures (pointing, waving) when communicating?' },
    { id: 'Q8', text: 'Does the child repeat phrases or sentences (echolalia)?' },
    { id: 'Q9', text: 'Does the child struggle to explain what they feel or want?' },
    { id: 'Q10', text: 'Does the child speak in a monotone or unusual rhythm?' },
    { id: 'Q11', text: 'Does the child have difficulty understanding multi-step verbal instructions?' }
  ],
  'Behaviour & Routine Patterns': [
    { id: 'Q12', text: 'Does the child get upset when routine or familiar patterns change?' },
    { id: 'Q13', text: 'Does the child insist on doing tasks in a very specific way?' },
    { id: 'Q14', text: 'Does the child show repetitive movements (hand-flapping, rocking, spinning)?' },
    { id: 'Q15', text: 'Does the child have strong fixations on certain topics or objects?' },
    { id: 'Q16', text: 'Does the child line up toys or arrange objects in a specific order repeatedly?' },
    { id: 'Q17', text: 'Does the child get overly focused on one task and struggle to shift attention?' }
  ],
  'Sensory Processing': [
    { id: 'Q18', text: 'Does the child react strongly to loud sounds, bright lights, or specific textures?' },
    { id: 'Q19', text: 'Does the child cover their ears frequently even when sounds are normal?' },
    { id: 'Q20', text: 'Does the child avoid certain clothes due to texture discomfort?' },
    { id: 'Q21', text: 'Does the child seek strong sensory input (jumping, spinning, crashing into things)?' },
    { id: 'Q22', text: 'Does the child have unusual food preferences based mainly on texture or smell?' }
  ],
  'Motor Skills': [
    { id: 'Q23', text: 'Does the child have difficulty with fine motor tasks (buttoning, writing, using scissors)?' },
    { id: 'Q24', text: 'Does the child appear clumsy or have poor coordination compared to peers?' },
    { id: 'Q25', text: 'Does the child show delayed development in basic motor milestones?' },
    { id: 'Q26', text: 'Does the child exhibit unusual motor mannerisms (finger flicking, pacing)?' }
  ],
  'Emotional Understanding & Social Behaviour': [
    { id: 'Q27', text: 'Does the child struggle to understand other people\'s feelings or perspectives?' },
    { id: 'Q28', text: 'Does the child overreact to minor changes or small frustrations?' },
    { id: 'Q29', text: 'Does the child avoid physical affection (hugs, touch) even with family?' },
    { id: 'Q30', text: 'Does the child have difficulty forming or maintaining friendships?' }
  ]
};

const QUESTIONS = Object.values(QUESTION_CATEGORIES).flat();

function App() {
  const [formData, setFormData] = useState({
    age: '',
    sex: '',
    urban_rural: '',
    siblings_asd: '',
    speech_delay: '',
    parental_concern: '',
    Q1: '',
    Q2: '',
    Q3: '',
    Q4: '',
    Q5: '',
    Q6: '',
    Q7: '',
    Q8: '',
    Q9: '',
    Q10: '',
    Q11: '',
    Q12: '',
    Q13: '',
    Q14: '',
    Q15: '',
    Q16: '',
    Q17: '',
    Q18: '',
    Q19: '',
    Q20: '',
    Q21: '',
    Q22: '',
    Q23: '',
    Q24: '',
    Q25: '',
    Q26: '',
    Q27: '',
    Q28: '',
    Q29: '',
    Q30: ''
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [chatQuery, setChatQuery] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [chatResponse, setChatResponse] = useState('');
  
  // New state for context management
  const [sessionId, setSessionId] = useState(null);
  const [symptomSeverity, setSymptomSeverity] = useState(null);
  const [suggestedQuestions, setSuggestedQuestions] = useState([]);
  const [conversationHistory, setConversationHistory] = useState([]);
  
  // NEW: State for refined follow-up system
  const [suggestedFollowups, setSuggestedFollowups] = useState([]);
  const [followupMode, setFollowupMode] = useState(true);
  const [progress, setProgress] = useState({ answered: 0, total: 12, percentage: 0 });
  const [clickedFollowup, setClickedFollowup] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      // Calculate the sum of all A scores (result field required by backend)
      const resultScore = Object.keys(formData)
        .filter(key => key.startsWith('A') && key.endsWith('_Score'))
        .reduce((sum, key) => sum + parseInt(formData[key] || 0), 0);
      
      const submissionData = {
        ...formData,
        result: resultScore
      };
      
      console.log('Submitting form data:', submissionData);
      const response = await axios.post('http://localhost:5000/predict', submissionData);
      console.log('Received response:', response.data);
      
      if (response.data) {
        setResult(response.data);
        setSessionId(response.data.session_id); // Store session ID
        setSymptomSeverity(response.data.symptom_severity); // Store symptom severity
        setSuggestedQuestions(response.data.suggested_questions || []); // Store suggested questions (old)
        
        // NEW: Store refined follow-up data
        setSuggestedFollowups(response.data.suggested_followups || []); // NEW follow-up format
        setFollowupMode(response.data.followup_mode !== false); // Default true
        
        setIsCollapsed(true);
        
        console.log('✓ Session ID:', response.data.session_id);
        console.log('✓ Symptom Severity:', response.data.symptom_severity);
        console.log('✓ Suggested Follow-ups:', response.data.suggested_followups);
      } else {
        throw new Error('No data received from server');
      }
    } catch (err) {
      const errorMessage = err.response?.data?.error || err.message || 'An error occurred during prediction';
      setError(errorMessage);
      console.error('Prediction error:', err);
      console.error('Error details:', err.response?.data);
      alert(`Error: ${errorMessage}`);
    } finally {
      setLoading(false);
    }
  };

  const handleFollowupClick = (followup) => {
    setClickedFollowup(followup);
    setChatQuery(`Answering: ${followup.question}\n\n`);
    // Optionally auto-focus the textarea
    setTimeout(() => {
      document.querySelector('.chat-input-area textarea')?.focus();
    }, 100);
  };

  const handleChatSubmit = async (e) => {
    e.preventDefault();
    if (!chatQuery.trim() || !sessionId) return;

    setChatLoading(true);
    
    // Add user message to history
    const userMessage = {
      role: 'user',
      message: chatQuery,
      timestamp: new Date().toLocaleTimeString()
    };
    setConversationHistory(prev => [...prev, userMessage]);
    
    const currentQuery = chatQuery;
    const currentClickedFollowup = clickedFollowup; // Capture current followup
    
    setChatQuery(''); // Clear input immediately
    setClickedFollowup(null); // Reset clicked followup
    
    try {
      const response = await axios.post('http://localhost:5000/chat', {
        query: currentQuery,
        session_id: sessionId, // Send session ID with chat request
        clicked_followup: currentClickedFollowup // NEW: Send which followup was clicked
      });
      
      // Add assistant response to history
      const assistantMessage = {
        role: 'assistant',
        message: response.data.response,
        timestamp: new Date().toLocaleTimeString(),
        isFollowUp: response.data.is_follow_up_question,
        symptomArea: response.data.symptom_area
      };
      setConversationHistory(prev => [...prev, assistantMessage]);
      
      // NEW: Update follow-up suggestions and progress
      setSuggestedFollowups(response.data.suggested_followups || []);
      setFollowupMode(response.data.followup_mode !== false);
      setProgress(response.data.progress || { answered: 0, total: 12, percentage: 0 });
      
      console.log('✓ Chat response received');
      console.log('✓ New follow-ups:', response.data.suggested_followups?.length || 0);
      console.log('✓ Progress:', response.data.progress);
    } catch (err) {
      const errorMessage = {
        role: 'assistant',
        message: 'Error: Unable to get response. Please try again.',
        timestamp: new Date().toLocaleTimeString(),
        isError: true
      };
      setConversationHistory(prev => [...prev, errorMessage]);
      console.error('Chat error:', err);
    } finally {
      setChatLoading(false);
    }
  };

  const resetForm = () => {
    setIsCollapsed(false);
    setResult(null);
    setChatResponse('');
    setChatQuery('');
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>🧩 Autism Spectrum Disorder Screening Tool</h1>
        <p className="subtitle">Early screening assessment for children</p>
      </header>

      <main className="main-content">
        {!isCollapsed ? (
          <form onSubmit={handleSubmit} className="screening-form">
            <section className="form-section">
              <h2>Demographic Information</h2>
              
              <div className="form-group">
                <label htmlFor="age">Age (years) *</label>
                <input
                  type="number"
                  id="age"
                  name="age"
                  value={formData.age}
                  onChange={handleChange}
                  required
                  min="4"
                  max="11"
                  placeholder="Enter age (4-11 years)"
                />
              </div>

              <div className="form-group">
                <label htmlFor="sex">Sex *</label>
                <select
                  id="sex"
                  name="sex"
                  value={formData.sex}
                  onChange={handleChange}
                  required
                >
                  <option value="">Select sex</option>
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="urban_rural">Residential Area *</label>
                <select
                  id="urban_rural"
                  name="urban_rural"
                  value={formData.urban_rural}
                  onChange={handleChange}
                  required
                >
                  <option value="">Select residential area</option>
                  <option value="urban">Urban</option>
                  <option value="rural">Rural</option>
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="siblings_asd">Siblings with ASD? *</label>
                <select
                  id="siblings_asd"
                  name="siblings_asd"
                  value={formData.siblings_asd}
                  onChange={handleChange}
                  required
                >
                  <option value="">Select</option>
                  <option value="1">Yes</option>
                  <option value="0">No</option>
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="speech_delay">Speech Delay? *</label>
                <select
                  id="speech_delay"
                  name="speech_delay"
                  value={formData.speech_delay}
                  onChange={handleChange}
                  required
                >
                  <option value="">Select</option>
                  <option value="1">Yes</option>
                  <option value="0">No</option>
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="parental_concern">Parental Concern Level *</label>
                <select
                  id="parental_concern"
                  name="parental_concern"
                  value={formData.parental_concern}
                  onChange={handleChange}
                  required
                >
                  <option value="">Select concern level</option>
                  <option value="low">Low</option>
                  <option value="medium">Medium</option>
                  <option value="high">High</option>
                </select>
              </div>
            </section>

            <section className="form-section">
              <h2>Screening Questions</h2>
              <p className="instructions">Please answer the following questions about the child's behavior. Select "Yes" or "No" for each question.</p>
              
              {Object.entries(QUESTION_CATEGORIES).map(([category, questions]) => (
                <div key={category} className="category-section">
                  <h3 className="category-title">{category}</h3>
                  {questions.map((question) => (
                    <div key={question.id} className="question-group">
                      <label className="question-label">
                        <span className="question-number">{question.id}.</span>
                        <span className="question-text">{question.text}</span>
                      </label>
                      <div className="radio-group">
                        <label className="radio-label">
                          <input
                            type="radio"
                            name={question.id}
                            value="1"
                            checked={formData[question.id] === '1'}
                            onChange={handleChange}
                            required
                          />
                          <span>Yes</span>
                        </label>
                        <label className="radio-label">
                          <input
                            type="radio"
                            name={question.id}
                            value="0"
                            checked={formData[question.id] === '0'}
                            onChange={handleChange}
                            required
                          />
                          <span>No</span>
                        </label>
                      </div>
                    </div>
                  ))}
                </div>
              ))}
            </section>

            {error && (
              <div className="error-message">
                {error}
              </div>
            )}

            <button type="submit" className="submit-button" disabled={loading}>
              {loading ? 'Analyzing...' : 'Submit Screening'}
            </button>
          </form>
        ) : (
          <div className="results-container">
            {loading ? (
              <div className="loading-container">
                <h2>Analyzing Results...</h2>
                <p>Please wait while we process your screening data.</p>
              </div>
            ) : (
              <>
            <div className="collapsed-form">
              <div className="collapsed-header">
                <h3>Screening Summary</h3>
              </div>
              <div className="collapsed-info">
                <div className="info-item">
                  <span className="info-label">Age:</span>
                  <span className="info-value">{formData.age} years</span>
                </div>
                <div className="info-item">
                  <span className="info-label">Sex:</span>
                  <span className="info-value">{formData.sex === 'male' ? 'Male' : 'Female'}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">Residential Area:</span>
                  <span className="info-value">{formData.urban_rural === 'urban' ? 'Urban' : 'Rural'}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">Speech Delay:</span>
                  <span className="info-value">{formData.speech_delay === '1' ? 'Yes' : 'No'}</span>
                </div>
              </div>
            </div>

            {result && (
              <div className="prediction-results">
                <h2>Screening Results</h2>
                <div className={`prediction-card ${result.prediction && result.prediction.includes('ASD Traits Detected') ? 'asd-positive' : 'asd-negative'}`}>
                  <div className="prediction-main">
                    <span className="prediction-label">Assessment:</span>
                    <span className="prediction-value">{result.prediction || 'Unknown'}</span>
                  </div>
                  <div className="confidence-bar">
                    <span className="confidence-label">Confidence:</span>
                    <div className="progress-bar">
                      <div 
                        className="progress-fill" 
                        style={{width: `${result.confidence || 0}%`}}
                      ></div>
                    </div>
                    <span className="confidence-value">{result.confidence || 0}%</span>
                  </div>
                  <div className="probabilities">
                    <div className="probability-item">
                      <span>ASD Probability:</span>
                      <strong>{result.probability_asd || 0}%</strong>
                    </div>
                    <div className="probability-item">
                      <span>No ASD Probability:</span>
                      <strong>{result.probability_no_asd || 0}%</strong>
                    </div>
                  </div>
                </div>

                {result.ai_insights && (
                  <div className="ai-insights">
                    <h3>🤖 AI Insights</h3>
                    {result.ai_insights.severity && (
                      <div className="insight-section">
                        <h4>Severity Assessment</h4>
                        <p className="severity-badge">{result.ai_insights.severity}</p>
                      </div>
                    )}
                    {result.ai_insights.key_findings && (
                      <div className="insight-section">
                        <h4>Key Findings</h4>
                        <div className="insight-content">
                          {Array.isArray(result.ai_insights.key_findings) ? (
                            <ul>
                              {result.ai_insights.key_findings.map((finding, idx) => (
                                <li key={idx}>{finding}</li>
                              ))}
                            </ul>
                          ) : (
                            <div dangerouslySetInnerHTML={{__html: result.ai_insights.key_findings.replace(/\n/g, '<br/>')}} />
                          )}
                        </div>
                      </div>
                    )}
                    {result.ai_insights.recommendations && (
                      <div className="insight-section">
                        <h4>Recommendations</h4>
                        <div className="insight-content">
                          {Array.isArray(result.ai_insights.recommendations) ? (
                            <ul>
                              {result.ai_insights.recommendations.map((rec, idx) => (
                                <li key={idx}>{rec}</li>
                              ))}
                            </ul>
                          ) : (
                            <div dangerouslySetInnerHTML={{__html: result.ai_insights.recommendations.replace(/\n/g, '<br/>')}} />
                          )}
                        </div>
                      </div>
                    )}
                    {result.ai_insights.follow_up && (
                      <div className="insight-section">
                        <h4>Follow-up Actions</h4>
                        <div className="insight-content">
                          {typeof result.ai_insights.follow_up === 'string' ? (
                            <div dangerouslySetInnerHTML={{__html: result.ai_insights.follow_up.replace(/\n/g, '<br/>')}} />
                          ) : (
                            <p>{result.ai_insights.follow_up}</p>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            <div className="chat-section">
              <h3>💬 Detailed Assessment Conversation</h3>
              <p className="chat-description">
                Our AI assistant will ask follow-up questions to better understand your child's specific behaviors and challenges.
              </p>
              
              {/* Symptom Severity Summary */}
              {symptomSeverity && (
                <div className="severity-summary">
                  <h4>📊 Symptom Areas</h4>
                  <div className="severity-grid">
                    {Object.entries(symptomSeverity).map(([area, level]) => (
                      <div key={area} className={`severity-badge ${level}`}>
                        <span className="severity-icon">
                          {level === 'high' ? '🔴' : level === 'moderate' ? '🟡' : '🟢'}
                        </span>
                        <span className="severity-text">
                          {area.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                        </span>
                        <span className="severity-level">{level.toUpperCase()}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              {/* NEW: Progress Indicator */}
              {followupMode && progress.total > 0 && (
                <div className="progress-indicator">
                  <div className="progress-header">
                    <span className="progress-text">Progress: {progress.answered}/{progress.total} questions</span>
                    <span className="progress-percentage">{progress.percentage}%</span>
                  </div>
                  <div className="progress-bar">
                    <div className="progress-fill" style={{ width: `${progress.percentage}%` }} />
                  </div>
                </div>
              )}
              
              {/* NEW: Suggested Follow-ups as Clickable Buttons */}
              {followupMode && suggestedFollowups.length > 0 && (
                <div className="suggested-followups">
                  <h4>📋 Suggested Follow-up Questions (Click to Answer):</h4>
                  <div className="followup-buttons">
                    {suggestedFollowups.map((followup, idx) => (
                      <button
                        key={idx}
                        className={`followup-btn severity-${followup.severity}`}
                        onClick={() => handleFollowupClick(followup)}
                      >
                        <span className="followup-icon">
                          {followup.severity === 'high' ? '🔴' : 
                           followup.severity === 'moderate' ? '🟡' : '🟢'}
                        </span>
                        <span className="followup-text">{followup.question}</span>
                        <span className="followup-area">
                          {followup.symptom_area.replace(/_/g, ' ')}
                        </span>
                      </button>
                    ))}
                  </div>
                  <p className="followup-hint">
                    💡 Click a question above to answer it, or type your own question below
                  </p>
                </div>
              )}
              
              {/* Conversation History */}
              {conversationHistory.length > 0 && (
                <div className="conversation-history">
                  {conversationHistory.map((msg, idx) => (
                    <div key={idx} className={`chat-message ${msg.role}`}>
                      <div className="message-header">
                        <span className="message-role">
                          {msg.role === 'user' ? '👤 You' : '🤖 AI Assistant'}
                        </span>
                        <span className="message-time">{msg.timestamp}</span>
                      </div>
                      <div className="message-content">
                        {msg.message}
                      </div>
                      {msg.isFollowUp && msg.symptomArea && (
                        <div className="message-tag">
                          🏷️ Focus: {msg.symptomArea.replace(/_/g, ' ')}
                        </div>
                      )}
                    </div>
                  ))}
                  {chatLoading && (
                    <div className="chat-message assistant">
                      <div className="message-header">
                        <span className="message-role">🤖 AI Assistant</span>
                      </div>
                      <div className="typing-indicator">
                        <span></span><span></span><span></span>
                      </div>
                    </div>
                  )}
                </div>
              )}
              
              {/* Suggested Questions */}
              {suggestedQuestions && suggestedQuestions.length > 0 && conversationHistory.length === 0 && (
                <div className="suggested-questions">
                  <h4>💡 We'll explore these areas:</h4>
                  <ul>
                    {suggestedQuestions.map((question, idx) => (
                      <li key={idx}>{question}</li>
                    ))}
                  </ul>
                  <p className="start-prompt">Type "start" or ask any question to begin the conversation.</p>
                </div>
              )}
              
              <form onSubmit={handleChatSubmit} className="chat-form">
                <textarea
                  value={chatQuery}
                  onChange={(e) => setChatQuery(e.target.value)}
                  placeholder="Type your response or question here..."
                  className="chat-input"
                  rows="3"
                  disabled={!sessionId}
                />
                <button type="submit" className="chat-button" disabled={chatLoading || !sessionId || !chatQuery.trim()}>
                  {chatLoading ? 'Thinking...' : 'Send'}
                </button>
              </form>
            </div>
              </>
            )}
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>⚠️ <strong>Important:</strong> This is a screening tool only. Professional evaluation is required for diagnosis.</p>
      </footer>
    </div>
  );
}

export default App;
