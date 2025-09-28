import React, { useState, useEffect } from 'react';
import axios from 'axios';

const Analytics = () => {
  const [analytics, setAnalytics] = useState({
    datasets: [],
    predictions: [],
    accuracy: null
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        const response = await axios.get('/api/analytics');
        setAnalytics(response.data);
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
