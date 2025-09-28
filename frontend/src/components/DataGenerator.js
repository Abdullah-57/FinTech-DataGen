import React, { useState } from 'react';
import axios from 'axios';

const DataGenerator = () => {
  const [formData, setFormData] = useState({
    symbol: 'AAPL',
    exchange: 'NASDAQ',
    days: 7,
    dataType: 'financial'
  });
  const [isGenerating, setIsGenerating] = useState(false);
  const [result, setResult] = useState(null);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsGenerating(true);
    setResult(null);

    try {
      const response = await axios.post('/api/generate', formData);
      setResult({
        success: true,
        data: response.data
      });
    } catch (error) {
      setResult({
        success: false,
        error: error.response?.data?.message || 'Failed to generate data'
      });
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="data-generator">
      <h2>🔧 Data Generator</h2>
      
      <div className="card">
        <h3>Generate Financial Dataset</h3>
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="symbol">Symbol/Ticker:</label>
            <input
              type="text"
              id="symbol"
              name="symbol"
              value={formData.symbol}
              onChange={handleInputChange}
              placeholder="e.g., AAPL, BTC-USD"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="exchange">Exchange:</label>
            <select
              id="exchange"
              name="exchange"
              value={formData.exchange}
              onChange={handleInputChange}
            >
              <option value="NASDAQ">NASDAQ</option>
              <option value="NYSE">NYSE</option>
              <option value="Crypto">Crypto</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="days">Days of History:</label>
            <input
              type="number"
              id="days"
              name="days"
              value={formData.days}
              onChange={handleInputChange}
              min="1"
              max="365"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="dataType">Data Type:</label>
            <select
              id="dataType"
              name="dataType"
              value={formData.dataType}
              onChange={handleInputChange}
            >
              <option value="financial">Financial Data</option>
              <option value="news">News Sentiment</option>
              <option value="combined">Combined Dataset</option>
            </select>
          </div>

          <button 
            type="submit" 
            className="btn"
            disabled={isGenerating}
          >
            {isGenerating ? 'Generating...' : 'Generate Dataset'}
          </button>
        </form>
      </div>

      {result && (
        <div className="card">
          <h3>Generation Result</h3>
          {result.success ? (
            <div>
              <p style={{ color: 'green' }}>✅ Dataset generated successfully!</p>
              <pre>{JSON.stringify(result.data, null, 2)}</pre>
            </div>
          ) : (
            <div>
              <p style={{ color: 'red' }}>❌ Error: {result.error}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default DataGenerator;
