import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import DataGenerator from './components/DataGenerator';
import Forecasts from './components/Forecasts';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <nav className="navbar">
          <div className="container">
            <h1>💰 FinTech DataGen</h1>
            <ul className="nav-links">
              <li><Link to="/">Dashboard</Link></li>
              <li><Link to="/generator">Data Generator</Link></li>
              <li><Link to="/forecasts">Forecasts</Link></li>
            </ul>
          </div>
        </nav>

        <main className="main-content">
          <div className="container">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/generator" element={<DataGenerator />} />
              <Route path="/forecasts" element={<Forecasts />} />
            </Routes>
          </div>
        </main>
      </div>
    </Router>
  );
}

export default App;
