import React, { useState, useEffect } from 'react';
import axios from 'axios';

const Dashboard = () => {
  const [status, setStatus] = useState('offline');
  const [stats, setStats] = useState({
    totalDatasets: 0,
    totalRecords: 0,
    lastGenerated: null
  });

  useEffect(() => {
    // Test backend connection
    const testConnection = async () => {
      try {
        const response = await axios.get('/api/health');
        setStatus('online');
        setStats(response.data.stats || {});
      } catch (error) {
        console.error('Backend connection failed:', error);
        setStatus('offline');
      }
    };

    testConnection();
    const interval = setInterval(testConnection, 30000); // Check every 30 seconds

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="dashboard">
      <h2>📊 Dashboard</h2>
      
      <div className="grid">
        <div className="card">
          <h3>System Status</h3>
          <p>
            <span className={`status-indicator status-${status}`}></span>
            Backend: {status === 'online' ? 'Connected' : 'Disconnected'}
          </p>
          <p>Frontend: Online</p>
        </div>

        <div className="card">
          <h3>Database Stats</h3>
          <p>Total Datasets: {stats.totalDatasets}</p>
          <p>Total Records: {stats.totalRecords}</p>
          <p>Last Generated: {stats.lastGenerated || 'Never'}</p>
        </div>

        <div className="card">
          <h3>Quick Actions</h3>
          <button className="btn" onClick={() => window.location.href = '/generator'}>
            Generate New Dataset
          </button>
          <button className="btn" onClick={() => window.location.href = '/analytics'}>
            View Analytics
          </button>
        </div>
      </div>

      <div className="card">
        <h3>Recent Activity</h3>
        <p>Welcome to FinTech DataGen! This is a boilerplate setup to test the tech stack.</p>
        <ul>
          <li>✅ React.js Frontend initialized</li>
          <li>✅ Flask Backend ready</li>
          <li>✅ MongoDB connection configured</li>
          <li>✅ Machine Learning models folder created</li>
        </ul>
      </div>
    </div>
  );
};

export default Dashboard;
