import React, { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';

const horizons = [
  { label: '1h', hours: 1 },
  { label: '3h', hours: 3 },
  { label: '24h', hours: 24 },
  { label: '72h', hours: 72 }
];

const Forecasts = () => {
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [horizon, setHorizon] = useState(24);
  const [prices, setPrices] = useState([]);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [models, setModels] = useState(['ma', 'arima', 'lstm']);
  const [ensemble, setEnsemble] = useState(false);
  const [publicHistorical, setPublicHistorical] = useState(null);
  const [publicForecast, setPublicForecast] = useState(null);
  const [showPublic, setShowPublic] = useState(false);
  const [datasetLoaded, setDatasetLoaded] = useState(false);

  const loadDatasets = async () => {
    try {
      console.log('🔍 Fetching datasets from /api/datasets...');
      const resp = await axios.get('/api/datasets', {
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        },
        responseType: 'json'
      });
      console.log('📡 API Response:', resp);
      console.log('📊 Response data:', resp.data);
      console.log('📊 Data type:', typeof resp.data);
      console.log('📊 Is array:', Array.isArray(resp.data));

      let data = resp.data;

      // Handle case where response is a string (needs parsing)
      if (typeof data === 'string') {
        try {
          console.log('🔧 Parsing string response as JSON...');
          // Clean NaN values before parsing
          const cleanedData = data.replace(/:\s*NaN/g, ': null');
          data = JSON.parse(cleanedData);
          console.log('✅ Successfully parsed JSON:', data);
        } catch (parseError) {
          console.error('❌ Failed to parse JSON:', parseError);
          console.error('❌ Raw data sample:', data.substring(0, 500) + '...');
          return [];
        }
      }

      // Ensure we always return an array
      if (Array.isArray(data)) {
        console.log(`✅ Found ${data.length} datasets`);
        return data;
      } else {
        console.warn('⚠️ API returned non-array data:', data);
        console.warn('⚠️ Data type:', typeof data);
        return [];
      }
    } catch (e) {
      console.error('❌ Failed to load datasets:', e);
      console.error('❌ Error details:', e.response?.data);
      return [];
    }
  };

  const loadPricesFromDataset = async (dataset) => {
    if (!dataset || !dataset.symbol) return [];
    const resp = await axios.get(`/api/prices?symbol=${encodeURIComponent(dataset.symbol)}&limit=300`);
    return resp.data.rows || [];
  };

  const runForecast = async () => {
    if (!selectedDataset) {
      throw new Error('No dataset selected');
    }
    const body = {
      symbol: selectedDataset.symbol,
      models: models,
      preview_horizon_hours: horizon,
      ensemble
    };
    const resp = await axios.post('/api/forecast/run', body);
    return resp.data.preview || null;
  };

  const loadSelectedDataset = async () => {
    if (!selectedDataset) {
      setError('Please select a dataset first');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      const p = await loadPricesFromDataset(selectedDataset);
      setPrices(p);
      setPreview(null); // Clear any previous forecast
      setDatasetLoaded(true);
    } catch (e) {
      setError(e?.response?.data?.error || e.message || 'Failed to load dataset');
      setDatasetLoaded(false);
    } finally {
      setLoading(false);
    }
  };

  const runForecastOnly = async () => {
    if (!datasetLoaded || !selectedDataset) {
      setError('Please load a dataset first');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      const prev = await runForecast();
      setPreview(prev);
      if (showPublic) {
        await fetchPublicEndpoints();
      }
    } catch (e) {
      setError(e?.response?.data?.error || e.message || 'Failed to run forecast');
    } finally {
      setLoading(false);
    }
  };

  const refresh = async () => {
    if (!selectedDataset) {
      setError('Please select a dataset first');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      const [p, prev] = await Promise.all([loadPricesFromDataset(selectedDataset), runForecast()]);
      setPrices(p);
      setPreview(prev);
      setDatasetLoaded(true);
      if (showPublic) {
        await fetchPublicEndpoints();
      }
    } catch (e) {
      setError(e?.response?.data?.error || e.message || 'Failed to load');
    } finally {
      setLoading(false);
    }
  };

  // Replace the current fetchPublicEndpoints with:
  const fetchPublicEndpoints = async () => {
    if (!selectedDataset) return;

    try {
      // Get historical data (this can stay the same)
      const hist = await axios.get(`/get_historical?symbol=${selectedDataset.symbol}&limit=300`);
      setPublicHistorical(hist.data);

      // Get stored predictions instead of generating new ones
      const storedPredictions = await axios.get(`/api/predictions?symbol=${selectedDataset.symbol}&models=${models.join(',')}&limit=5`);

      // Format the stored predictions to match expected structure
      const formattedResults = {
        symbol: selectedDataset.symbol,
        results: storedPredictions.data,
        preview: {
          models: storedPredictions.data.map(pred => ({
            model: pred.model,
            predicted_values: pred.predicted_values,
            horizon_hours: pred.forecast_horizon * 24 // Convert if needed
          }))
        }
      };

      setPublicForecast(formattedResults);
    } catch (e) {
      setPublicForecast({ error: e?.response?.data?.error || e.message });
    }
  };

  useEffect(() => {
    const initializeComponent = async () => {
      try {
        console.log('🚀 Initializing Forecasts component...');
        setLoading(true);
        setError(null);
        const availableDatasets = await loadDatasets();

        console.log('📋 Available datasets:', availableDatasets);
        console.log('📋 Datasets type:', typeof availableDatasets);
        console.log('📋 Is array:', Array.isArray(availableDatasets));

        // Ensure datasets is always an array
        if (Array.isArray(availableDatasets)) {
          setDatasets(availableDatasets);
          console.log(`✅ Set ${availableDatasets.length} datasets in state`);

          if (availableDatasets.length === 0) {
            console.log('⚠️ No datasets found, showing message to user');
            setError('No datasets found. Please go to the Data Generator page to create a dataset first.');
          } else {
            console.log('✅ Datasets loaded successfully');
          }
        } else {
          console.error('❌ loadDatasets returned non-array:', availableDatasets);
          setDatasets([]);
          setError('Failed to load datasets - invalid data format');
        }
      } catch (e) {
        console.error('❌ Error in initializeComponent:', e);
        setDatasets([]);
        setError('Failed to load datasets');
      } finally {
        setLoading(false);
        console.log('🏁 Forecasts component initialization complete');
      }
    };

    initializeComponent();
  }, []);

  const candlestick = useMemo(() => {
    if (!prices.length || !selectedDataset) return {};

    const dates = prices.map(r => r.date);
    return {
      x: dates,
      open: prices.map(r => r.open),
      high: prices.map(r => r.high),
      low: prices.map(r => r.low),
      close: prices.map(r => r.close),
      type: 'candlestick',
      name: `${selectedDataset.symbol} OHLCV`
    };
  }, [prices, selectedDataset]);

  const forecastTraces = useMemo(() => {
    if (!preview) return [];
    const x = preview.dates;
    return (preview.models || []).map(m => ({
      x,
      y: m.predicted_values,
      type: 'scatter',
      mode: 'lines+markers',
      name: `Forecast: ${m.model} (${preview.horizon_hours || m.horizon_hours}h)`
    }));
  }, [preview]);

  return (
    <div className="forecasts">
      <h2>📈 Forecasts</h2>
      <div className="card">
        <div className="grid" style={{ gridTemplateColumns: '320px 1fr', alignItems: 'start', gap: '24px' }}>
          <div>
            <div className="form-group">
              <label>Select Dataset</label>
              {!Array.isArray(datasets) || datasets.length === 0 ? (
                <div style={{
                  padding: '12px',
                  border: '2px dashed #ffc107',
                  borderRadius: '6px',
                  backgroundColor: '#fff3cd',
                  color: '#856404',
                  textAlign: 'center'
                }}>
                  <p style={{ margin: '0 0 8px 0', fontWeight: 'bold' }}>No datasets available</p>
                  <p style={{ margin: 0, fontSize: '14px' }}>
                    Please go to the <strong>Data Generator</strong> page to create a dataset first.
                  </p>
                </div>
              ) : (
                <select
                  value={selectedDataset ? selectedDataset._id : ''}
                  onChange={e => {
                    const dataset = datasets.find(d => d._id === e.target.value);
                    setSelectedDataset(dataset);
                    setDatasetLoaded(false);
                    setPrices([]);
                    setPreview(null);
                  }}
                >
                  <option value="">Choose a dataset...</option>
                  {Array.isArray(datasets) && datasets.map(dataset => (
                    <option key={dataset._id} value={dataset._id}>
                      {dataset.symbol} ({dataset.exchange}) - {dataset.records} records - {new Date(dataset.generated_at).toLocaleDateString()}
                    </option>
                  ))}
                </select>
              )}
            </div>

            {selectedDataset && (
              <div style={{
                padding: '8px 12px',
                backgroundColor: '#e7f3ff',
                border: '1px solid #b3d9ff',
                borderRadius: '4px',
                marginBottom: '16px',
                fontSize: '14px'
              }}>
                <strong>Selected:</strong> {selectedDataset.symbol} ({selectedDataset.exchange})<br />
                <strong>Records:</strong> {selectedDataset.records}<br />
                <strong>Generated:</strong> {new Date(selectedDataset.generated_at).toLocaleDateString()}
              </div>
            )}
            <div className="form-group">
              <label>Forecast Horizon</label>
              <select value={horizon} onChange={e => setHorizon(parseInt(e.target.value, 10))}>
                {horizons.map(h => (
                  <option key={h.hours} value={h.hours}>{h.label}</option>
                ))}
              </select>
            </div>
            <div className="form-group">
              <label>Models</label>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                {[
                  { id: 'ma', label: 'Moving Average' },
                  { id: 'arima', label: 'ARIMA' },
                  { id: 'lstm', label: 'LSTM' },
                  { id: 'transformer', label: 'Transformer' }
                ].map(opt => {
                  const selected = models.includes(opt.id);
                  return (
                    <button
                      key={opt.id}
                      type="button"
                      onClick={() => setModels(prev => selected ? prev.filter(x => x !== opt.id) : Array.from(new Set([...prev, opt.id])))}
                      style={{
                        padding: '8px 12px',
                        borderRadius: 6,
                        border: selected ? '2px solid #007bff' : '1px solid #ccc',
                        background: selected ? '#e7f1ff' : '#f8f9fa',
                        color: '#333',
                        cursor: 'pointer',
                        fontWeight: selected ? 600 : 500
                      }}
                    >
                      {opt.label}
                    </button>
                  );
                })}
              </div>
            </div>
            <div className="form-group">
              <label>Ensemble</label>
              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                <button
                  type="button"
                  onClick={() => setEnsemble(prev => !prev)}
                  style={{
                    padding: '8px 12px',
                    borderRadius: 6,
                    border: ensemble ? '2px solid #28a745' : '1px solid #ccc',
                    background: ensemble ? '#e9f7ef' : '#f8f9fa',
                    color: '#333',
                    cursor: 'pointer',
                    fontWeight: ensemble ? 600 : 500
                  }}
                >
                  {ensemble ? 'Ensemble: On' : 'Ensemble: Off'}
                </button>
              </div>
            </div>
            <div style={{ marginBottom: 12 }}>
              <p style={{ margin: '0 0 8px 0', fontSize: '14px', color: '#666' }}>
                💡 <strong>Step 1:</strong> Load dataset, then <strong>Step 2:</strong> Run forecast with your selected models
              </p>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                <button
                  className="btn"
                  onClick={loadSelectedDataset}
                  disabled={loading || !selectedDataset}
                  style={{
                    backgroundColor: selectedDataset ? '#007bff' : '#6c757d'
                  }}
                >
                  {loading ? 'Loading…' : '📊 Load Dataset'}
                </button>
                <button
                  className="btn"
                  onClick={runForecastOnly}
                  disabled={loading || !datasetLoaded || models.length === 0}
                  style={{
                    backgroundColor: (datasetLoaded && models.length > 0) ? '#28a745' : '#6c757d',
                    fontWeight: 'bold'
                  }}
                >
                  {loading ? 'Running…' : '🚀 Run Forecast'}
                </button>
                <button
                  className="btn"
                  onClick={refresh}
                  disabled={loading || !selectedDataset}
                  style={{ backgroundColor: '#6c757d' }}
                >
                  {loading ? 'Loading…' : '🔄 Refresh All'}
                </button>
              </div>
            </div>
            {selectedDataset && (
              <div style={{ marginTop: 8, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                {!showPublic && (
                  <button
                    type="button"
                    className="btn"
                    onClick={async () => { setShowPublic(true); await fetchPublicEndpoints(); }}
                    disabled={!datasetLoaded}
                  >
                    Show Public API Results
                  </button>
                )}
                {showPublic && (
                  <button
                    type="button"
                    className="btn"
                    onClick={() => { setShowPublic(false); setPublicHistorical(null); setPublicForecast(null); }}
                    style={{ backgroundColor: '#6c757d' }}
                  >
                    Hide Public API Results
                  </button>
                )}
              </div>
            )}
            {error && <p className="error-message" style={{ marginTop: 8 }}>❌ {error}</p>}

            {/* Debug section */}
            <div style={{ marginTop: 16, padding: '12px', backgroundColor: '#f8f9fa', border: '1px solid #dee2e6', borderRadius: '4px' }}>
              <h4 style={{ margin: '0 0 8px 0', fontSize: '14px', color: '#6c757d' }}>Debug Information</h4>
              <button
                className="btn"
                onClick={async () => {
                  try {
                    console.log('🔧 Testing debug endpoint...');
                    const resp = await axios.get('/api/debug/datasets');
                    console.log('🔧 Debug response:', resp.data);
                    alert('Debug info logged to console. Check browser console for details.');
                  } catch (e) {
                    console.error('🔧 Debug endpoint failed:', e);
                    alert('Debug endpoint failed. Check console for details.');
                  }
                }}
                style={{ fontSize: '12px', padding: '4px 8px' }}
              >
                🔧 Test Debug Endpoint
              </button>
            </div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div style={{ minHeight: 700, height: '72vh' }}>
              {!selectedDataset && (
                <div style={{
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  transform: 'translate(-50%, -50%)',
                  background: 'rgba(255, 255, 255, 0.9)',
                  padding: '20px',
                  borderRadius: '8px',
                  border: '2px dashed #ffc107',
                  textAlign: 'center',
                  zIndex: 1000
                }}>
                  <p style={{ margin: 0, fontSize: '16px', color: '#856404' }}>
                    � Please select a dataset to begin forecasting
                  </p>
                </div>
              )}

              {selectedDataset && !datasetLoaded && (
                <div style={{
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  transform: 'translate(-50%, -50%)',
                  background: 'rgba(255, 255, 255, 0.9)',
                  padding: '20px',
                  borderRadius: '8px',
                  border: '2px dashed #007bff',
                  textAlign: 'center',
                  zIndex: 1000
                }}>
                  <p style={{ margin: 0, fontSize: '16px', color: '#0056b3' }}>
                    📊 Dataset selected. Click "Load Dataset" to view historical data.
                  </p>
                </div>
              )}

              {datasetLoaded && !preview && prices.length > 0 && (
                <div style={{
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  transform: 'translate(-50%, -50%)',
                  background: 'rgba(255, 255, 255, 0.9)',
                  padding: '20px',
                  borderRadius: '8px',
                  border: '2px dashed #28a745',
                  textAlign: 'center',
                  zIndex: 1000
                }}>
                  <p style={{ margin: 0, fontSize: '16px', color: '#155724' }}>
                    📊 Dataset loaded. Click "Run Forecast" to see predictions.
                  </p>
                  <p style={{ margin: '8px 0 0 0', fontSize: '14px', color: '#6c757d' }}>
                    Config: {selectedDataset.symbol} | {models.join(', ')} | {horizon}h | {ensemble ? 'Ensemble ON' : 'Ensemble OFF'}
                  </p>
                </div>
              )}

              <Plot
                data={selectedDataset && datasetLoaded ? [candlestick, ...forecastTraces] : []}
                layout={{
                  title: selectedDataset
                    ? `${selectedDataset.symbol} Candlesticks${preview ? ' + Forecast' : datasetLoaded ? ' (No forecast run yet)' : ' (Dataset not loaded)'}`
                    : 'Select a dataset to view chart',
                  dragmode: 'zoom',
                  xaxis: { title: 'Date' },
                  yaxis: { title: 'Price' },
                  showlegend: true,
                  autosize: true
                }}
                useResizeHandler
                style={{ width: '100%', height: '100%' }}
              />
            </div>
            {showPublic && selectedDataset && (
              <div style={{
                display: 'block',
                width: 'calc(100% + 320px)',
                marginLeft: '-320px'
              }}>
                <div className="card" style={{ margin: '12px auto 0', maxWidth: 1100, width: '100%' }}>
                  <h4 style={{ margin: '0 0 8px 0', textAlign: 'center' }}>Public Endpoint Outputs for {selectedDataset.symbol}</h4>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                    <div style={{ border: '1px solid #e1e4e8', borderRadius: 6, overflow: 'hidden' }}>
                      <div style={{ padding: '8px 12px', background: '#f1f3f5', borderBottom: '1px solid #e1e4e8', fontWeight: 600 }}>GET /get_historical</div>
                      <pre style={{ height: 320, margin: 0, overflow: 'auto', background: '#fff', padding: 12, fontSize: 12, lineHeight: 1.4 }}>
                        {publicHistorical ? JSON.stringify(publicHistorical, null, 2) : '—'}
                      </pre>
                    </div>
                    <div style={{ border: '1px solid #e1e4e8', borderRadius: 6, overflow: 'hidden' }}>
                      <div style={{ padding: '8px 12px', background: '#f1f3f5', borderBottom: '1px solid #e1e4e8', fontWeight: 600 }}>GET /get_forecast</div>
                      <pre style={{ height: 320, margin: 0, overflow: 'auto', background: '#fff', padding: 12, fontSize: 12, lineHeight: 1.4 }}>
                        {publicForecast ? JSON.stringify(publicForecast, null, 2) : '—'}
                      </pre>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Forecasts;


