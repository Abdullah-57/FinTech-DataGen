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
  const [symbol, setSymbol] = useState('AAPL');
  const [exchange, setExchange] = useState('NASDAQ');
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

  const loadPrices = async () => {
    const resp = await axios.get(`/api/prices?symbol=${encodeURIComponent(symbol)}&limit=300`);
    return resp.data.rows || [];
  };

  const runForecast = async () => {
    const body = {
      symbol,
      models: models,
      preview_horizon_hours: horizon,
      ensemble
    };
    const resp = await axios.post('/api/forecast/run', body);
    return resp.data.preview || null;
  };

  const refresh = async () => {
    try {
      setLoading(true);
      setError(null);
      const [p, prev] = await Promise.all([loadPrices(), runForecast()]);
      setPrices(p);
      setPreview(prev);
      if (showPublic) {
        await fetchPublicEndpoints();
      }
    } catch (e) {
      setError(e?.response?.data?.error || e.message || 'Failed to load');
    } finally {
      setLoading(false);
    }
  };

  const fetchPublicEndpoints = async () => {
    try {
      const hist = await axios.get(`/get_historical?symbol=${encodeURIComponent(symbol)}&limit=300`);
      setPublicHistorical(hist.data);
      const q = new URLSearchParams({
        symbol,
        horizon: `${horizon}h`,
        models: models.join(','),
        ensemble: String(ensemble)
      }).toString();
      const fc = await axios.get(`/get_forecast?${q}`);
      setPublicForecast(fc.data);
    } catch (e) {
      // Do not override main error; just annotate public blocks
      setPublicForecast({ error: e?.response?.data?.error || e.message });
    }
  };

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const candlestick = useMemo(() => {
    const dates = prices.map(r => r.date);
    return {
      x: dates,
      open: prices.map(r => r.open),
      high: prices.map(r => r.high),
      low: prices.map(r => r.low),
      close: prices.map(r => r.close),
      type: 'candlestick',
      name: `${symbol} OHLCV`
    };
  }, [prices, symbol]);

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
        <div className="grid" style={{gridTemplateColumns: '320px 1fr', alignItems: 'start', gap: '24px'}}>
          <div>
            <div className="form-group">
              <label>Instrument (Symbol)</label>
              <input value={symbol} onChange={e => setSymbol(e.target.value)} placeholder="e.g., AAPL, BTC-USD" />
            </div>
            <div className="form-group">
              <label>Exchange</label>
              <input value={exchange} onChange={e => setExchange(e.target.value)} placeholder="e.g., NASDAQ, Crypto" />
            </div>
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
              <div style={{display:'flex', flexWrap:'wrap', gap:'8px'}}>
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
              <div style={{display:'flex', gap:'8px', flexWrap:'wrap'}}>
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
            <button className="btn" onClick={refresh} disabled={loading}>
              {loading ? 'Loading…' : 'Refresh'}
            </button>
            <div style={{marginTop: 8, display:'flex', gap: 8, flexWrap:'wrap'}}>
              {!showPublic && (
                <button
                  type="button"
                  className="btn"
                  onClick={async () => { setShowPublic(true); await fetchPublicEndpoints(); }}
                >
                  Show Public API Results
                </button>
              )}
              {showPublic && (
                <button
                  type="button"
                  className="btn"
                  onClick={() => { setShowPublic(false); setPublicHistorical(null); setPublicForecast(null); }}
                  style={{backgroundColor:'#6c757d'}}
                >
                  Hide Public API Results
                </button>
              )}
            </div>
            {error && <p className="error-message" style={{marginTop: 8}}>❌ {error}</p>}
          </div>
          <div style={{display:'flex', flexDirection:'column', gap: 12}}>
            <div style={{minHeight: 700, height: '72vh'}}>
              <Plot
              data={[candlestick, ...forecastTraces]}
              layout={{
                title: `${symbol} Candlesticks + Forecast`,
                dragmode: 'zoom',
                xaxis: { title: 'Date' },
                yaxis: { title: 'Price' },
                showlegend: true,
                autosize: true
              }}
              useResizeHandler
              style={{width: '100%', height: '100%'}}
              />
            </div>
            {showPublic && (
              <div style={{
                display:'block',
                width: 'calc(100% + 320px)',
                marginLeft: '-320px'
              }}>
                <div className="card" style={{margin: '12px auto 0', maxWidth: 1100, width: '100%'}}>
                  <h4 style={{margin:'0 0 8px 0', textAlign:'center'}}>Public Endpoint Outputs</h4>
                  <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap: 12}}>
                    <div style={{border:'1px solid #e1e4e8', borderRadius: 6, overflow:'hidden'}}>
                      <div style={{padding:'8px 12px', background:'#f1f3f5', borderBottom:'1px solid #e1e4e8', fontWeight:600}}>GET /get_historical</div>
                      <pre style={{height: 320, margin:0, overflow:'auto', background:'#fff', padding: 12, fontSize:12, lineHeight:1.4}}>
{publicHistorical ? JSON.stringify(publicHistorical, null, 2) : '—'}
                      </pre>
                    </div>
                    <div style={{border:'1px solid #e1e4e8', borderRadius: 6, overflow:'hidden'}}>
                      <div style={{padding:'8px 12px', background:'#f1f3f5', borderBottom:'1px solid #e1e4e8', fontWeight:600}}>GET /get_forecast</div>
                      <pre style={{height: 320, margin:0, overflow:'auto', background:'#fff', padding: 12, fontSize:12, lineHeight:1.4}}>
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


