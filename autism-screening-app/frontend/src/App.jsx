import { useState } from 'react';
import axios from 'axios';
import './App.css';

const QUESTIONS = [
  { id: 'A1_Score', text: 'S/he often notices small sounds when others do not' },
  { id: 'A2_Score', text: 'S/he usually concentrates more on the whole picture, rather than the small details' },
  { id: 'A3_Score', text: 'In a social group, s/he can easily keep track of several different people\'s conversations' },
  { id: 'A4_Score', text: 'S/he finds it easy to go back and forth between different activities' },
  { id: 'A5_Score', text: 'S/he doesn\'t know how to keep a conversation going with his/her peers' },
  { id: 'A6_Score', text: 'S/he is good at social chit-chat' },
  { id: 'A7_Score', text: 'When s/he is read a story, s/he finds it difficult to work out the character\'s intentions or feelings' },
  { id: 'A8_Score', text: 'When s/he was in preschool, s/he used to enjoy playing games involving pretending with other children' },
  { id: 'A9_Score', text: 'S/he finds it easy to work out what someone is thinking or feeling just by looking at their face' },
  { id: 'A10_Score', text: 'S/he finds it hard to make new friends' }
];

function App() {
  const [formData, setFormData] = useState({
    age: '',
    gender: '',
    ethnicity: '',
    jundice: '',
    austim: '',
    contry_of_res: '',
    used_app_before: '',
    relation: '',
    A1_Score: '',
    A2_Score: '',
    A3_Score: '',
    A4_Score: '',
    A5_Score: '',
    A6_Score: '',
    A7_Score: '',
    A8_Score: '',
    A9_Score: '',
    A10_Score: ''
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
                <label htmlFor="gender">Gender *</label>
                <select
                  id="gender"
                  name="gender"
                  value={formData.gender}
                  onChange={handleChange}
                  required
                >
                  <option value="">Select gender</option>
                  <option value="1">Male</option>
                  <option value="0">Female</option>
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="ethnicity">Ethnicity *</label>
                <select
                  id="ethnicity"
                  name="ethnicity"
                  value={formData.ethnicity}
                  onChange={handleChange}
                  required
                >
                  <option value="">Select ethnicity</option>
                  <option value="White-European">White European</option>
                  <option value="Asian">Asian</option>
                  <option value="Middle Eastern">Middle Eastern</option>
                  <option value="Black">Black</option>
                  <option value="Hispanic">Hispanic</option>
                  <option value="South Asian">South Asian</option>
                  <option value="Others">Others</option>
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="jundice">Was the child born with jaundice? *</label>
                <select
                  id="jundice"
                  name="jundice"
                  value={formData.jundice}
                  onChange={handleChange}
                  required
                >
                  <option value="">Select</option>
                  <option value="1">Yes</option>
                  <option value="0">No</option>
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="austim">Family history of ASD? *</label>
                <select
                  id="austim"
                  name="austim"
                  value={formData.austim}
                  onChange={handleChange}
                  required
                >
                  <option value="">Select</option>
                  <option value="1">Yes</option>
                  <option value="0">No</option>
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="contry_of_res">Country of Residence *</label>
                <input
                  type="text"
                  id="contry_of_res"
                  name="contry_of_res"
                  value={formData.contry_of_res}
                  onChange={handleChange}
                  required
                  placeholder="e.g., United States"
                />
              </div>

              <div className="form-group">
                <label htmlFor="used_app_before">Used screening app before? *</label>
                <select
                  id="used_app_before"
                  name="used_app_before"
                  value={formData.used_app_before}
                  onChange={handleChange}
                  required
                >
                  <option value="">Select</option>
                  <option value="1">Yes</option>
                  <option value="0">No</option>
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="relation">Relation to child *</label>
                <select
                  id="relation"
                  name="relation"
                  value={formData.relation}
                  onChange={handleChange}
                  required
                >
                  <option value="">Select relation</option>
                  <option value="Parent">Parent</option>
                  <option value="Health care professional">Health care professional</option>
                  <option value="Relative">Relative</option>
                  <option value="Others">Others</option>
                </select>
              </div>
            </section>

            <section className="form-section">
              <h2>Screening Questions</h2>
              <p className="instructions">Please answer the following questions about the child's behavior. Select "Yes" or "No" for each question.</p>
              
              {QUESTIONS.map((question, index) => (
                <div key={question.id} className="question-group">
                  <label className="question-label">
                    <span className="question-number">Q{index + 1}.</span>
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
                  <span className="info-label">Gender:</span>
                  <span className="info-value">{formData.gender === '1' ? 'Male' : 'Female'}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">Ethnicity:</span>
                  <span className="info-value">{formData.ethnicity}</span>
                </div>
                <div className="info-item">
                  <span className="info-label">Country:</span>
                  <span className="info-value">{formData.contry_of_res}</span>
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
