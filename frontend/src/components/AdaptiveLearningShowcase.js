import React, { useState, useEffect } from 'react';
import './AdaptiveLearningShowcase.css';

const AdaptiveLearningShowcase = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [systemStatus, setSystemStatus] = useState(null);
  const [registrationResult, setRegistrationResult] = useState(null);
  const [trainingResults, setTrainingResults] = useState({});
  const [predictions, setPredictions] = useState({});
  const [performance, setPerformance] = useState({});
  const [modelVersions, setModelVersions] = useState([]);
  const [trainingEvents, setTrainingEvents] = useState([]);
  const [dbStats, setDbStats] = useState(null);
  const [loading, setLoading] = useState({});
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [selectedModelType, setSelectedModelType] = useState('sgd');
  const [isRunning, setIsRunning] = useState(false);

  const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:5000';
  const modelTypes = ['sgd', 'lstm', 'ensemble'];
  const symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'];

  // Utility function for API calls
  const apiCall = async (endpoint, method = 'GET', body = null) => {
    try {
      const options = {
        method,
        headers: { 'Content-Type': 'application/json' },
      };
      if (body) options.body = JSON.stringify(body);
      
      const response = await fetch(`${API_BASE}${endpoint}`, options);
      const data = await response.json();
      return { success: response.ok, data, status: response.status };
    } catch (error) {
      return { success: false, error: error.message };
    }
  };

  // Load initial data
  useEffect(() => {
    loadSystemStatus();
    loadDatabaseStats();
    loadModelVersions();
    loadTrainingEvents();
  }, []);

  const loadSystemStatus = async () => {
    const result = await apiCall('/api/adaptive/status');
    if (result.success) {
      setSystemStatus(result.data);
      setIsRunning(result.data.is_running);
    }
  };

  const loadDatabaseStats = async () => {
    const result = await apiCall('/api/adaptive/stats');
    if (result.success) {
      setDbStats(result.data);
    }
  };

  const loadModelVersions = async () => {
    const result = await apiCall('/api/adaptive/versions?limit=20');
    if (result.success) {
      setModelVersions(result.data);
    }
  };

  const loadTrainingEvents = async () => {
    const result = await apiCall('/api/adaptive/training-events?limit=20');
    if (result.success) {
      setTrainingEvents(result.data);
    }
  };

  // Feature demonstrations
  const handleRegisterSymbol = async () => {
    setLoading(prev => ({ ...prev, register: true }));
    const result = await apiCall('/api/adaptive/register', 'POST', {
      symbol: selectedSymbol,
      model_types: modelTypes
    });
    setRegistrationResult(result);
    setLoading(prev => ({ ...prev, register: false }));
    if (result.success) {
      loadSystemStatus();
    }
  };

  const handleInitialTraining = async (modelType) => {
    setLoading(prev => ({ ...prev, [`train_${modelType}`]: true }));
    const result = await apiCall('/api/adaptive/train', 'POST', {
      symbol: selectedSymbol,
      model_type: modelType
    });
    setTrainingResults(prev => ({ ...prev, [modelType]: result }));
    setLoading(prev => ({ ...prev, [`train_${modelType}`]: false }));
    if (result.success) {
      loadModelVersions();
      loadTrainingEvents();
    }
  };

  const handleManualUpdate = async (modelType) => {
    setLoading(prev => ({ ...prev, [`update_${modelType}`]: true }));
    const result = await apiCall('/api/adaptive/update', 'POST', {
      symbol: selectedSymbol,
      model_type: modelType
    });
    setTrainingResults(prev => ({ ...prev, [`update_${modelType}`]: result }));
    setLoading(prev => ({ ...prev, [`update_${modelType}`]: false }));
    if (result.success) {
      loadModelVersions();
      loadTrainingEvents();
    }
  };

  const handleMakePrediction = async (modelType, horizon = 5) => {
    setLoading(prev => ({ ...prev, [`predict_${modelType}`]: true }));
    const result = await apiCall('/api/adaptive/predict', 'POST', {
      symbol: selectedSymbol,
      model_type: modelType,
      horizon: horizon
    });
    setPredictions(prev => ({ ...prev, [modelType]: result }));
    setLoading(prev => ({ ...prev, [`predict_${modelType}`]: false }));
  };

  const handleGetPerformance = async (modelType) => {
    setLoading(prev => ({ ...prev, [`perf_${modelType}`]: true }));
    const result = await apiCall(`/api/adaptive/performance/${selectedSymbol}/${modelType}`);
    setPerformance(prev => ({ ...prev, [modelType]: result }));
    setLoading(prev => ({ ...prev, [`perf_${modelType}`]: false }));
  };

  const handleStartContinuousLearning = async () => {
    setLoading(prev => ({ ...prev, start: true }));
    const result = await apiCall('/api/adaptive/start', 'POST');
    setLoading(prev => ({ ...prev, start: false }));
    if (result.success) {
      setIsRunning(true);
      loadSystemStatus();
    }
  };

  const handleStopContinuousLearning = async () => {
    setLoading(prev => ({ ...prev, stop: true }));
    const result = await apiCall('/api/adaptive/stop', 'POST');
    setLoading(prev => ({ ...prev, stop: false }));
    if (result.success) {
      setIsRunning(false);
      loadSystemStatus();
    }
  };

  const handleRollback = async (modelType, version) => {
    setLoading(prev => ({ ...prev, [`rollback_${modelType}`]: true }));
    const result = await apiCall('/api/adaptive/rollback', 'POST', {
      symbol: selectedSymbol,
      model_type: modelType,
      version: version
    });
    setLoading(prev => ({ ...prev, [`rollback_${modelType}`]: false }));
    if (result.success) {
      loadModelVersions();
      loadTrainingEvents();
    }
  };

  const renderOverview = () => (
    <div className="showcase-section">
      <h2>🧠 Adaptive Learning System Overview</h2>
      <div className="feature-grid">
        <div className="feature-card">
          <h3>🤖 Online Learning Models</h3>
          <ul>
            <li><strong>OnlineSGDRegressor</strong>: Fast incremental learning (&lt;1s updates)</li>
            <li><strong>OnlineLSTM</strong>: Deep learning with fine-tuning (10-60s updates)</li>
            <li><strong>AdaptiveEnsemble</strong>: Dynamic model weighting</li>
          </ul>
        </div>
        <div className="feature-card">
          <h3>📊 Model Management</h3>
          <ul>
            <li><strong>Automatic Versioning</strong>: Creates versions on performance improvement</li>
            <li><strong>Performance Tracking</strong>: RMSE, MAE, MAPE monitoring</li>
            <li><strong>Rollback Capabilities</strong>: Revert to previous versions</li>
          </ul>
        </div>
        <div className="feature-card">
          <h3>🔄 Continuous Learning</h3>
          <ul>
            <li><strong>Scheduled Updates</strong>: SGD (6h), LSTM (24h), Ensemble (12h)</li>
            <li><strong>Data Preprocessing</strong>: Technical indicators, feature engineering</li>
            <li><strong>Performance Alerts</strong>: Automatic rollback on degradation</li>
          </ul>
        </div>
        <div className="feature-card">
          <h3>🗄️ Database Integration</h3>
          <ul>
            <li><strong>model_versions</strong>: Version history and metrics</li>
            <li><strong>training_events</strong>: Training activity logs</li>
            <li><strong>performance_history</strong>: Detailed performance tracking</li>
          </ul>
        </div>
      </div>
      
      <div className="system-status">
        <h3>📈 Current System Status</h3>
        {systemStatus ? (
          <div className="status-grid">
            <div className="status-item">
              <span className="label">Continuous Learning:</span>
              <span className={`status ${isRunning ? 'running' : 'stopped'}`}>
                {isRunning ? '🟢 Running' : '🔴 Stopped'}
              </span>
            </div>
            <div className="status-item">
              <span className="label">Registered Models:</span>
              <span className="value">{systemStatus.registered_models}</span>
            </div>
            <div className="status-item">
              <span className="label">Recent Alerts:</span>
              <span className="value">{systemStatus.recent_alerts?.length || 0}</span>
            </div>
          </div>
        ) : (
          <div className="loading">Loading system status...</div>
        )}
      </div>

      <div className="database-stats">
        <h3>🗄️ Database Statistics</h3>
        {dbStats ? (
          <div className="stats-grid">
            <div className="stat-item">
              <span className="stat-value">{dbStats.total_model_versions || 0}</span>
              <span className="stat-label">Model Versions</span>
            </div>
            <div className="stat-item">
              <span className="stat-value">{dbStats.total_training_events || 0}</span>
              <span className="stat-label">Training Events</span>
            </div>
            <div className="stat-item">
              <span className="stat-value">{dbStats.active_models || 0}</span>
              <span className="stat-label">Active Models</span>
            </div>
            <div className="stat-item">
              <span className="stat-value">{dbStats.recent_versions || 0}</span>
              <span className="stat-label">Recent Versions (7d)</span>
            </div>
          </div>
        ) : (
          <div className="loading">Loading database stats...</div>
        )}
      </div>
    </div>
  );

  const renderRegistration = () => (
    <div className="showcase-section">
      <h2>📝 Symbol Registration</h2>
      <p>Register a symbol for adaptive learning with multiple model types.</p>
      
      <div className="demo-controls">
        <div className="control-group">
          <label>Select Symbol:</label>
          <select value={selectedSymbol} onChange={(e) => setSelectedSymbol(e.target.value)}>
            {symbols.map(symbol => (
              <option key={symbol} value={symbol}>{symbol}</option>
            ))}
          </select>
        </div>
        
        <button 
          onClick={handleRegisterSymbol}
          disabled={loading.register}
          className="demo-button primary"
        >
          {loading.register ? 'Registering...' : `Register ${selectedSymbol}`}
        </button>
      </div>

      {registrationResult && (
        <div className={`result ${registrationResult.success ? 'success' : 'error'}`}>
          <h4>Registration Result:</h4>
          <pre>{JSON.stringify(registrationResult.data, null, 2)}</pre>
        </div>
      )}
    </div>
  );

  const renderTraining = () => (
    <div className="showcase-section">
      <h2>🎯 Model Training</h2>
      <p>Train different model types and see their performance metrics.</p>
      
      <div className="model-grid">
        {modelTypes.map(modelType => (
          <div key={modelType} className="model-card">
            <h3>{modelType.toUpperCase()} Model</h3>
            <div className="model-actions">
              <button
                onClick={() => handleInitialTraining(modelType)}
                disabled={loading[`train_${modelType}`]}
                className="demo-button primary"
              >
                {loading[`train_${modelType}`] ? 'Training...' : 'Initial Training'}
              </button>
              <button
                onClick={() => handleManualUpdate(modelType)}
                disabled={loading[`update_${modelType}`]}
                className="demo-button secondary"
              >
                {loading[`update_${modelType}`] ? 'Updating...' : 'Manual Update'}
              </button>
            </div>
            
            {trainingResults[modelType] && (
              <div className={`result ${trainingResults[modelType].success ? 'success' : 'error'}`}>
                <h5>Training Result:</h5>
                {trainingResults[modelType].success ? (
                  <div className="metrics">
                    <div>Version: {trainingResults[modelType].data.version}</div>
                    {trainingResults[modelType].data.metrics && (
                      <>
                        <div>RMSE: {trainingResults[modelType].data.metrics.rmse?.toFixed(4)}</div>
                        <div>MAE: {trainingResults[modelType].data.metrics.mae?.toFixed(4)}</div>
                        <div>MAPE: {trainingResults[modelType].data.metrics.mape?.toFixed(2)}%</div>
                      </>
                    )}
                  </div>
                ) : (
                  <div className="error-message">{trainingResults[modelType].data?.message || trainingResults[modelType].error}</div>
                )}
              </div>
            )}

            {trainingResults[`update_${modelType}`] && (
              <div className={`result ${trainingResults[`update_${modelType}`].success ? 'success' : 'error'}`}>
                <h5>Update Result:</h5>
                {trainingResults[`update_${modelType}`].success ? (
                  <div className="metrics">
                    <div>Version: {trainingResults[`update_${modelType}`].data.version}</div>
                    <div>New Version Created: {trainingResults[`update_${modelType}`].data.new_version_created ? 'Yes' : 'No'}</div>
                    {trainingResults[`update_${modelType}`].data.metrics && (
                      <>
                        <div>RMSE: {trainingResults[`update_${modelType}`].data.metrics.rmse?.toFixed(4)}</div>
                        <div>MAE: {trainingResults[`update_${modelType}`].data.metrics.mae?.toFixed(4)}</div>
                      </>
                    )}
                  </div>
                ) : (
                  <div className="error-message">{trainingResults[`update_${modelType}`].data?.message || trainingResults[`update_${modelType}`].error}</div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );

  const renderPredictions = () => (
    <div className="showcase-section">
      <h2>🔮 Predictions</h2>
      <p>Make predictions with trained models for different horizons.</p>
      
      <div className="model-grid">
        {modelTypes.map(modelType => (
          <div key={modelType} className="model-card">
            <h3>{modelType.toUpperCase()} Predictions</h3>
            <div className="model-actions">
              <button
                onClick={() => handleMakePrediction(modelType, 1)}
                disabled={loading[`predict_${modelType}`]}
                className="demo-button primary"
              >
                {loading[`predict_${modelType}`] ? 'Predicting...' : '1-Day Forecast'}
              </button>
              <button
                onClick={() => handleMakePrediction(modelType, 5)}
                disabled={loading[`predict_${modelType}`]}
                className="demo-button secondary"
              >
                5-Day Forecast
              </button>
              <button
                onClick={() => handleMakePrediction(modelType, 10)}
                disabled={loading[`predict_${modelType}`]}
                className="demo-button tertiary"
              >
                10-Day Forecast
              </button>
            </div>
            
            {predictions[modelType] && (
              <div className={`result ${predictions[modelType].success ? 'success' : 'error'}`}>
                <h5>Prediction Result:</h5>
                {predictions[modelType].success ? (
                  <div className="prediction-data">
                    <div>Horizon: {predictions[modelType].data.horizon} days</div>
                    <div>Predictions:</div>
                    <div className="predictions-list">
                      {predictions[modelType].data.predictions?.map((pred, idx) => (
                        <span key={idx} className="prediction-value">
                          Day {idx + 1}: ${pred.toFixed(2)}
                        </span>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="error-message">{predictions[modelType].data?.message || predictions[modelType].error}</div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );

  const renderPerformance = () => (
    <div className="showcase-section">
      <h2>📊 Performance Monitoring</h2>
      <p>Monitor model performance, version history, and metrics over time.</p>
      
      <div className="model-grid">
        {modelTypes.map(modelType => (
          <div key={modelType} className="model-card">
            <h3>{modelType.toUpperCase()} Performance</h3>
            <button
              onClick={() => handleGetPerformance(modelType)}
              disabled={loading[`perf_${modelType}`]}
              className="demo-button primary"
            >
              {loading[`perf_${modelType}`] ? 'Loading...' : 'Get Performance'}
            </button>
            
            {performance[modelType] && (
              <div className={`result ${performance[modelType].success ? 'success' : 'error'}`}>
                {performance[modelType].success ? (
                  <div className="performance-data">
                    <h5>Performance Summary:</h5>
                    <div className="summary-stats">
                      <div>Total Versions: {performance[modelType].data.performance_summary?.total_versions || 0}</div>
                      <div>Current Version: {performance[modelType].data.performance_summary?.current_version || 0}</div>
                      <div>Best RMSE: {performance[modelType].data.performance_summary?.best_rmse?.toFixed(4) || 'N/A'}</div>
                      <div>Latest RMSE: {performance[modelType].data.performance_summary?.latest_rmse?.toFixed(4) || 'N/A'}</div>
                    </div>
                    
                    {performance[modelType].data.version_history?.length > 0 && (
                      <div className="version-history">
                        <h6>Recent Versions:</h6>
                        {performance[modelType].data.version_history.slice(-3).map(version => (
                          <div key={version.version} className="version-item">
                            <span>v{version.version}</span>
                            <span>RMSE: {version.metrics?.rmse?.toFixed(4)}</span>
                            <span>{new Date(version.timestamp).toLocaleDateString()}</span>
                            {version.version > 1 && (
                              <button
                                onClick={() => handleRollback(modelType, version.version)}
                                disabled={loading[`rollback_${modelType}`]}
                                className="rollback-button"
                              >
                                Rollback
                              </button>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="error-message">{performance[modelType].data?.message || performance[modelType].error}</div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );

  const renderContinuousLearning = () => (
    <div className="showcase-section">
      <h2>🔄 Continuous Learning Control</h2>
      <p>Start and stop the continuous learning scheduler that automatically updates models.</p>
      
      <div className="continuous-controls">
        <div className="status-display">
          <h3>Current Status: {isRunning ? '🟢 Running' : '🔴 Stopped'}</h3>
          <p>
            {isRunning 
              ? 'Models are updating automatically based on their schedules'
              : 'Continuous learning is stopped - models will only update manually'
            }
          </p>
        </div>
        
        <div className="control-buttons">
          <button
            onClick={handleStartContinuousLearning}
            disabled={loading.start || isRunning}
            className="demo-button primary"
          >
            {loading.start ? 'Starting...' : 'Start Continuous Learning'}
          </button>
          <button
            onClick={handleStopContinuousLearning}
            disabled={loading.stop || !isRunning}
            className="demo-button secondary"
          >
            {loading.stop ? 'Stopping...' : 'Stop Continuous Learning'}
          </button>
        </div>
        
        <div className="schedule-info">
          <h4>Update Schedules:</h4>
          <ul>
            <li><strong>SGD Models</strong>: Every 6 hours (fast updates)</li>
            <li><strong>LSTM Models</strong>: Every 24 hours (stability)</li>
            <li><strong>Ensemble Models</strong>: Every 12 hours (balanced)</li>
          </ul>
        </div>
      </div>
    </div>
  );

  const renderDatabaseView = () => (
    <div className="showcase-section">
      <h2>🗄️ Database Integration</h2>
      <p>View stored model versions, training events, and database statistics.</p>
      
      <div className="database-sections">
        <div className="db-section">
          <h3>Model Versions</h3>
          <button onClick={loadModelVersions} className="demo-button secondary">
            Refresh Versions
          </button>
          <div className="data-table">
            {modelVersions.length > 0 ? (
              <table>
                <thead>
                  <tr>
                    <th>Symbol</th>
                    <th>Model Type</th>
                    <th>Version</th>
                    <th>RMSE</th>
                    <th>Created</th>
                    <th>Active</th>
                  </tr>
                </thead>
                <tbody>
                  {modelVersions.slice(0, 10).map((version, idx) => (
                    <tr key={idx}>
                      <td>{version.symbol}</td>
                      <td>{version.model_type}</td>
                      <td>{version.version}</td>
                      <td>{version.performance_metrics?.rmse?.toFixed(4) || 'N/A'}</td>
                      <td>{new Date(version.created_at).toLocaleDateString()}</td>
                      <td>{version.is_active ? '✅' : '❌'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="no-data">No model versions found</div>
            )}
          </div>
        </div>
        
        <div className="db-section">
          <h3>Training Events</h3>
          <button onClick={loadTrainingEvents} className="demo-button secondary">
            Refresh Events
          </button>
          <div className="data-table">
            {trainingEvents.length > 0 ? (
              <table>
                <thead>
                  <tr>
                    <th>Symbol</th>
                    <th>Model Type</th>
                    <th>Trigger</th>
                    <th>Status</th>
                    <th>Timestamp</th>
                  </tr>
                </thead>
                <tbody>
                  {trainingEvents.slice(0, 10).map((event, idx) => (
                    <tr key={idx}>
                      <td>{event.symbol}</td>
                      <td>{event.model_type}</td>
                      <td>{event.trigger_type}</td>
                      <td className={event.status === 'completed' ? 'success' : 'error'}>
                        {event.status}
                      </td>
                      <td>{new Date(event.timestamp).toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="no-data">No training events found</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );

  const tabs = [
    { id: 'overview', label: '🏠 Overview', component: renderOverview },
    { id: 'registration', label: '📝 Registration', component: renderRegistration },
    { id: 'training', label: '🎯 Training', component: renderTraining },
    { id: 'predictions', label: '🔮 Predictions', component: renderPredictions },
    { id: 'performance', label: '📊 Performance', component: renderPerformance },
    { id: 'continuous', label: '🔄 Continuous Learning', component: renderContinuousLearning },
    { id: 'database', label: '🗄️ Database', component: renderDatabaseView },
  ];

  return (
    <div className="adaptive-learning-showcase">
      <div className="showcase-header">
        <h1>🧠 Adaptive Learning & Continuous Evaluation Showcase</h1>
        <p>Interactive demonstration of all implemented adaptive learning features</p>
      </div>
      
      <div className="showcase-tabs">
        {tabs.map(tab => (
          <button
            key={tab.id}
            className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      
      <div className="showcase-content">
        {tabs.find(tab => tab.id === activeTab)?.component()}
      </div>
      
      <div className="showcase-footer">
        <div className="api-info">
          <h3>🌐 Available API Endpoints</h3>
          <div className="endpoint-list">
            <div className="endpoint-item">POST /api/adaptive/register - Register symbol for adaptive learning</div>
            <div className="endpoint-item">POST /api/adaptive/train - Perform initial model training</div>
            <div className="endpoint-item">POST /api/adaptive/update - Manually trigger model update</div>
            <div className="endpoint-item">POST /api/adaptive/predict - Make predictions with trained models</div>
            <div className="endpoint-item">GET /api/adaptive/status - Get system status</div>
            <div className="endpoint-item">GET /api/adaptive/performance/&lt;symbol&gt;/&lt;model_type&gt; - Get model performance</div>
            <div className="endpoint-item">POST /api/adaptive/rollback - Rollback model to specific version</div>
            <div className="endpoint-item">POST /api/adaptive/start - Start continuous learning</div>
            <div className="endpoint-item">POST /api/adaptive/stop - Stop continuous learning</div>
            <div className="endpoint-item">GET /api/adaptive/stats - Get database statistics</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdaptiveLearningShowcase;