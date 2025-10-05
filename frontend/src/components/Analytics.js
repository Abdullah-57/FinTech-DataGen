import React, { useState, useEffect } from 'react';
import axios from 'axios';

const Analytics = () => {
  const [analytics, setAnalytics] = useState({
    datasets: [],
    predictions: [],
    accuracy: null
  });
  const [metadata, setMetadata] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        const [analyticsResponse, metadataResponse] = await Promise.all([
          axios.get('/api/analytics'),
          axios.get('/api/metadata')
        ]);
        setAnalytics(analyticsResponse.data);
        setMetadata(metadataResponse.data);
      } catch (error) {
        console.error('Failed to fetch analytics:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchAnalytics();
  }, []);

  if (loading) {
    return (
      <div className="analytics">
        <h2>📈 Analytics</h2>
        <div className="card">
          <p>Loading analytics data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="analytics">
      <h2>📈 Analytics</h2>
      
      <div className="grid">
        <div className="card">
          <h3>Model Performance</h3>
          <p>Accuracy: {analytics.accuracy ? `${analytics.accuracy}%` : 'N/A'}</p>
          <p>Total Datasets: {analytics.datasets.length}</p>
          <p>Predictions Made: {analytics.predictions.length}</p>
        </div>

        <div className="card">
          <h3>Recent Datasets</h3>
          {analytics.datasets.length > 0 ? (
            <ul>
              {analytics.datasets.slice(0, 5).map((dataset, index) => (
                <li key={index}>
                  {dataset.symbol} - {dataset.date} ({dataset.records} records)
                </li>
              ))}
            </ul>
          ) : (
            <p>No datasets available</p>
          )}
        </div>

        <div className="card">
          <h3>ML Model Status</h3>
          <p>Status: Ready</p>
          <p>Last Training: Never</p>
          <p>Model Type: Financial Prediction</p>
        </div>
      </div>

      <div className="card">
        <h3>📋 Instrument Metadata</h3>
        {metadata.length > 0 ? (
          <div>
            {metadata.map((meta, index) => (
              <div key={index} style={{ marginBottom: '20px', padding: '10px', border: '1px solid #ddd', borderRadius: '5px' }}>
                <h4>📊 {meta.instrument_info?.symbol || 'Unknown Symbol'}</h4>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                  <div>
                    <strong>Instrument Info:</strong>
                    <ul>
                      <li>Symbol: {meta.instrument_info?.symbol || 'N/A'}</li>
                      <li>Exchange: {meta.instrument_info?.exchange || 'N/A'}</li>
                      <li>Last Updated: {meta.instrument_info?.last_updated || 'N/A'}</li>
                      <li>Data Points: {meta.instrument_info?.data_points || 'N/A'}</li>
                      <li>Date Range: {meta.instrument_info?.date_range?.start} to {meta.instrument_info?.date_range?.end}</li>
                    </ul>
                  </div>
                  <div>
                    <strong>Data Sources:</strong>
                    <ul>
                      <li>Market Data: {meta.data_sources?.market_data || 'N/A'}</li>
                      <li>News Data: {meta.data_sources?.news_data || 'N/A'}</li>
                      <li>Technical Indicators: {meta.data_sources?.technical_indicators || 'N/A'}</li>
                      <li>Sentiment Analysis: {meta.data_sources?.sentiment_analysis || 'N/A'}</li>
                      {meta.data_sources?.forecast_models && (
                        <li>Forecast Models: {meta.data_sources.forecast_models}</li>
                      )}
                      {meta.data_sources?.ensemble_method && (
                        <li>Ensemble Method: {meta.data_sources.ensemble_method}</li>
                      )}
                    </ul>
                  </div>
                </div>
                <div>
                  <strong>Recent Activity:</strong>
                  <ul>
                    {meta.update_logs?.slice(-3).map((log, logIndex) => (
                      <li key={logIndex}>
                        {log.timestamp} - {log.action} ({log.status})
                        {log.records_added && ` - ${log.records_added} records`}
                        {log.models_used && ` - Models: ${log.models_used.join(', ')}`}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p>No metadata available. Generate some data first to see metadata information.</p>
        )}
      </div>

      <div className="card">
        <h3>Data Visualization</h3>
        <p>This is a placeholder for future data visualization components.</p>
        <p>Features to be implemented:</p>
        <ul>
          <li>📊 Price trend charts</li>
          <li>📈 Volume analysis</li>
          <li>🎯 Prediction accuracy metrics</li>
          <li>📉 Sentiment analysis graphs</li>
        </ul>
      </div>
    </div>
  );
};

export default Analytics;
