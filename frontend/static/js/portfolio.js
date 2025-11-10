// Portfolio Management JavaScript

class PortfolioManager {
    constructor() {
        this.portfolioId = localStorage.getItem('portfolio_id') || null;
        this.apiBase = 'http://localhost:5000/api';
        this.refreshInterval = null;
    }

    async listPortfolios() {
        try {
            const response = await fetch(`${this.apiBase}/portfolio/list`);
            const data = await response.json();
            
            if (data.success) {
                return data.portfolios;
            }
            throw new Error(data.error || 'Failed to list portfolios');
        } catch (error) {
            console.error('Error listing portfolios:', error);
            throw error;
        }
    }

    async createPortfolio(initialCash = 100000, name = 'Portfolio') {
        try {
            const response = await fetch(`${this.apiBase}/portfolio/create`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    initial_cash: initialCash,
                    name: name
                })
            });
            const data = await response.json();
            
            if (data.success) {
                this.portfolioId = data.portfolio_id;
                return data;
            }
            throw new Error(data.error || 'Failed to create portfolio');
        } catch (error) {
            console.error('Error creating portfolio:', error);
            throw error;
        }
    }

    async getPortfolio() {
        if (!this.portfolioId) {
            throw new Error('No portfolio ID');
        }
        
        try {
            const response = await fetch(`${this.apiBase}/portfolio/${this.portfolioId}`);
            const data = await response.json();
            
            if (data.success) {
                return data.portfolio;
            }
            throw new Error(data.error || 'Failed to get portfolio');
        } catch (error) {
            console.error('Error getting portfolio:', error);
            throw error;
        }
    }

    async getPositions() {
        if (!this.portfolioId) {
            throw new Error('No portfolio ID');
        }
        
        try {
            const response = await fetch(`${this.apiBase}/portfolio/${this.portfolioId}/positions`);
            const data = await response.json();
            
            if (data.success) {
                return data.summary;
            }
            throw new Error(data.error || 'Failed to get positions');
        } catch (error) {
            console.error('Error getting positions:', error);
            throw error;
        }
    }

    async executeTrade(symbol, action, shares, trigger = 'manual') {
        if (!this.portfolioId) {
            throw new Error('No portfolio ID');
        }
        
        try {
            const response = await fetch(`${this.apiBase}/portfolio/${this.portfolioId}/trade`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symbol, action, shares, trigger })
            });
            
            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }
            
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error executing trade:', error);
            throw error;
        }
    }

    async generateSignal(symbol, currentPrice, predictedPrice, confidence, modelName) {
        if (!this.portfolioId) {
            throw new Error('No portfolio ID');
        }
        
        try {
            const response = await fetch(`${this.apiBase}/portfolio/${this.portfolioId}/signal`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    symbol,
                    current_price: currentPrice,
                    predicted_price: predictedPrice,
                    confidence,
                    model_name: modelName
                })
            });
            const data = await response.json();
            
            if (data.success) {
                return data.signal;
            }
            throw new Error(data.error || 'Failed to generate signal');
        } catch (error) {
            console.error('Error generating signal:', error);
            throw error;
        }
    }

    async autoTrade(symbol, modelName = 'ensemble') {
        if (!this.portfolioId) {
            throw new Error('No portfolio ID');
        }
        
        try {
            const response = await fetch(`${this.apiBase}/portfolio/${this.portfolioId}/auto-trade`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symbol, model_name: modelName })
            });
            
            if (!response.ok) {
                const text = await response.text();
                console.error('Auto trade error response:', text);
                throw new Error(`Server error: ${response.status}`);
            }
            
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error auto trading:', error);
            throw error;
        }
    }

    async getTradeHistory(limit = 50) {
        if (!this.portfolioId) {
            throw new Error('No portfolio ID');
        }
        
        try {
            const response = await fetch(`${this.apiBase}/portfolio/${this.portfolioId}/trades?limit=${limit}`);
            const data = await response.json();
            
            if (data.success) {
                return data.trades;
            }
            throw new Error(data.error || 'Failed to get trade history');
        } catch (error) {
            console.error('Error getting trade history:', error);
            throw error;
        }
    }

    async getPerformance(days = 30) {
        if (!this.portfolioId) {
            throw new Error('No portfolio ID');
        }
        
        try {
            const response = await fetch(`${this.apiBase}/portfolio/${this.portfolioId}/performance?days=${days}`);
            const data = await response.json();
            
            if (data.success) {
                return data.performance;
            }
            throw new Error(data.error || 'Failed to get performance');
        } catch (error) {
            console.error('Error getting performance:', error);
            throw error;
        }
    }

    async getPerformanceHistory(days = 90) {
        if (!this.portfolioId) {
            throw new Error('No portfolio ID');
        }
        
        try {
            const response = await fetch(`${this.apiBase}/portfolio/${this.portfolioId}/performance/history?days=${days}`);
            const data = await response.json();
            
            if (data.success) {
                return data.history;
            }
            throw new Error(data.error || 'Failed to get performance history');
        } catch (error) {
            console.error('Error getting performance history:', error);
            throw error;
        }
    }

    async getRiskDashboard() {
        if (!this.portfolioId) {
            throw new Error('No portfolio ID');
        }
        
        try {
            const response = await fetch(`${this.apiBase}/portfolio/${this.portfolioId}/risk`);
            const data = await response.json();
            
            if (data.success) {
                return data.risk_dashboard;
            }
            throw new Error(data.error || 'Failed to get risk dashboard');
        } catch (error) {
            console.error('Error getting risk dashboard:', error);
            throw error;
        }
    }

    async checkStopLosses() {
        if (!this.portfolioId) {
            throw new Error('No portfolio ID');
        }
        
        try {
            const response = await fetch(`${this.apiBase}/portfolio/${this.portfolioId}/stop-losses`);
            const data = await response.json();
            
            if (data.success) {
                return data.stop_losses;
            }
            throw new Error(data.error || 'Failed to check stop losses');
        } catch (error) {
            console.error('Error checking stop losses:', error);
            throw error;
        }
    }

    startAutoRefresh(interval = 30000) {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
        
        this.refreshInterval = setInterval(async () => {
            try {
                await this.refreshDashboard();
            } catch (error) {
                console.error('Auto refresh error:', error);
            }
        }, interval);
    }

    stopAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
        }
    }

    async refreshDashboard() {
        // Override this method in your UI implementation
        console.log('Refreshing portfolio dashboard...');
    }

    formatCurrency(value) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD'
        }).format(value);
    }

    formatPercentage(value) {
        return `${(value * 100).toFixed(2)}%`;
    }

    formatNumber(value, decimals = 2) {
        return value.toFixed(decimals);
    }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PortfolioManager;
}
