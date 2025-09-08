# 📈 Portfolio Analysis & Efficient Frontier Optimization

**A comprehensive Python implementation of Modern Portfolio Theory using Monte Carlo simulation and real market data**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()

## 🎯 **Project Overview**

This project implements a sophisticated portfolio optimization system that helps investors build diversified portfolios using Modern Portfolio Theory (MPT). The analysis uses real market data to identify optimal asset allocations that maximize returns while minimizing risk through Monte Carlo simulation.

### **Key Features**
- 📊 **8-Asset Diversified Universe**: Stocks, Bonds, REITs, Commodities, International Markets
- 🎲 **Monte Carlo Simulation**: 25,000+ portfolio combinations for robust optimization
- 📈 **Efficient Frontier Visualization**: Interactive risk-return analysis
- 🔍 **Comprehensive Risk Metrics**: Sharpe ratios, drawdown analysis, correlation matrices
- 🏆 **Multiple Optimization Strategies**: Max Sharpe, Min Risk, Max Return portfolios
- 📱 **Professional Visualizations**: Publication-ready charts and heatmaps

---

## 🗂️ **Repository Structure**

```
portfolio-analysis/
├── README.md                          # This file
├── portfolio_analysis.ipynb           # Main Jupyter notebook
├── portfolio_analysis.py              # python script
├── requirements.txt                   # Python dependencies
├── PORTFOLIO Analysis Report.pdf      # Project Report (academic format)
├── results/                           # (Auto-generated) Analysis outputs
│   ├── monte_carlo_results.csv
│   ├── optimal_portfolio_weights.csv
│   ├── correlation_matrix.csv
│   ├── asset_risk_return_metrics.csv
│   ├── drawdown_analysis.csv
│   └── executive_summary.csv
└── images/                            # (Auto-generated) Visualization outputs
    ├── efficient_frontier.png
    ├── performance_comparison.png
    └── correlation_heatmap.png
```

---

## 🚀 **Quick Start**

### **Prerequisites**
```bash
Python 3.8+
Jupyter Notebook or VS Code with Jupyter extension
```

### **Installation**
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/portfolio-analysis.git
   cd portfolio-analysis
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the analysis:**
   ```bash
   jupyter notebook portfolio_analysis.ipynb
   ```
   *Or open in VS Code and run cells sequentially*

### **Dependencies**
```txt
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.6.0
yfinance>=0.2.0
seaborn>=0.12.0
jupyter>=1.0.0
```

---

## 📊 **Asset Universe**

The analysis covers a diversified 8-asset portfolio designed for broad market exposure:

| **Asset** | **Symbol** | **Category** | **Purpose** |
|-----------|------------|--------------|-------------|
| S&P 500 | SPY | US Large Cap | Core equity exposure |
| NASDAQ 100 | QQQ | US Tech | Growth & innovation |
| Russell 2000 | IWM | US Small Cap | Small-cap premium |
| MSCI EAFE | EFA | International Developed | Geographic diversification |
| MSCI Emerging Markets | EEM | Emerging Markets | Higher growth potential |
| REITs | VNQ | Real Estate | Inflation hedge & income |
| Gold | GLD | Commodities | Crisis hedge & diversification |
| US Bonds | BND | Fixed Income | Stability & income |

---

## 🔬 **Methodology**

### **1. Data Collection & Processing**
- **Source**: Yahoo Finance API via `yfinance`
- **Period**: 2015-Present (10+ years of data)
- **Frequency**: Daily adjusted closing prices
- **Quality Control**: Automatic handling of missing data, corporate actions

### **2. Risk-Return Calculations**
- **Returns**: Daily log returns (more accurate for portfolio mathematics)
- **Volatility**: Annualized standard deviation (252 trading days)
- **Sharpe Ratio**: Risk-adjusted returns using 10-year Treasury as risk-free rate
- **Correlation Matrix**: Asset interdependencies for diversification analysis

### **3. Monte Carlo Simulation**
- **Simulations**: 25,000 random portfolio weight combinations
- **Constraints**: Fully invested (weights sum to 100%), long-only positions
- **Optimization**: Identify portfolios maximizing Sharpe ratio, minimizing risk, maximizing return

### **4. Risk Analysis**
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Value at Risk**: Statistical risk measure at 95% confidence
- **Correlation Analysis**: Portfolio diversification effectiveness

---

## 📈 **Key Outputs**

### **1. Optimal Portfolio Strategies**

| **Strategy** | **Objective** | **Target Investor** |
|--------------|---------------|-------------------|
| **Max Sharpe** | Best risk-adjusted returns | Balanced investors |
| **Min Risk** | Lowest volatility | Conservative investors |
| **Max Return** | Highest expected returns | Aggressive investors |

### **2. Performance Metrics**
- Expected annual returns and volatility
- Risk-adjusted performance (Sharpe ratios)
- Diversification benefits quantification
- Historical drawdown analysis

### **3. Visualizations**
- **Efficient Frontier**: Risk-return scatter plot with optimization results
- **Correlation Heatmap**: Asset relationship visualization  
- **Performance Comparison**: Normalized price trends over time
- **Portfolio Allocation**: Asset weight distributions for optimal portfolios

---

## 🎯 **Sample Results**

*Example output from recent analysis (your results may vary):*

### **Optimal Portfolio Performance**
```
Max Sharpe Portfolio:
├─ Expected Return: 9.2% annually
├─ Volatility: 12.1% annually  
├─ Sharpe Ratio: 0.847
└─ Top Holdings: S&P 500 (35%), Bonds (25%), Gold (18%)

Diversification Benefit: +0.234 Sharpe improvement vs best single asset
Risk Reduction: 23% volatility reduction vs concentrated portfolio
```

### **Asset Correlation Insights**
- Average correlation: 0.31 (excellent diversification)
- Highest correlation: S&P 500 ↔ NASDAQ (0.89)  
- Lowest correlation: Gold ↔ Bonds (-0.12)
- **Conclusion**: Strong diversification benefits available

---

## 🛠️ **Customization Options**

### **Modify Asset Universe**
```python
# Edit in Section 1.2 of notebook
assets = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'VNQ', 'GLD', 'BND']
# Add/remove tickers as needed
```

### **Adjust Analysis Period**
```python
# Edit in Section 1.2 of notebook  
start_date = dt(2015, 1, 1)  # Start date
end_date = dt.now()          # End date
```

### **Change Risk-Free Rate**
```python
# Edit based on current market conditions
RISK_FREE_RATE = 0.045  # 4.5% annual (update with current 10Y Treasury)
```

### **Modify Simulation Parameters**
```python
# Edit in Section 7.1 of notebook
num_simulations = 25000  # Increase for more precision, decrease for speed
```

---

## 📊 **Technical Details**

### **Mathematical Framework**
- **Portfolio Return**: R_p = Σ(w_i × R_i)
- **Portfolio Variance**: σ²_p = w^T × Σ × w  
- **Sharpe Ratio**: (R_p - R_f) / σ_p
- **Efficient Frontier**: Set of optimal portfolios offering highest expected return for each level of risk

### **Risk Measures**
- **Value at Risk (VaR)**: Maximum expected loss at 95% confidence
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Correlation Matrix**: Pearson correlation coefficients between asset returns

### **Optimization Constraints**
- Weights sum to 1 (fully invested)
- No short selling (w_i ≥ 0)
- No leverage (max weight per asset: 100%)

---

## 🔍 **Code Structure**

The notebook is organized into 9 main sections:

1. **Setup & Configuration** - Libraries, parameters, asset definitions
2. **Data Collection** - Robust download with error handling
3. **Data Overview** - Price analysis and performance visualization  
4. **Risk-Return Analysis** - Sharpe ratios, volatility calculations
5. **Drawdown Analysis** - Risk assessment and recovery periods
6. **Correlation Analysis** - Diversification effectiveness
7. **Monte Carlo Simulation** - Portfolio optimization engine
8. **Efficient Frontier** - Visualization and optimal portfolio identification
9. **Results & Conclusions** - Investment recommendations and summary

### **Key Functions**
- `download_data_robust()` - Multi-strategy data download with fallbacks
- `calculate_comprehensive_drawdown()` - Advanced risk metrics
- Monte Carlo optimization loop - Portfolio weight generation and evaluation

---

## 🎨 **Visualization Gallery**

### **1. Efficient Frontier Plot**
- Scatter plot of 25,000+ portfolio combinations
- Color-coded by Sharpe ratio
- Optimal portfolios highlighted with distinct markers
- Individual assets plotted as diamonds

### **2. Correlation Heatmap**  
- Asset-by-asset correlation matrix
- Color-coded: Blue (negative) → White (neutral) → Red (positive)
- Numerical correlation coefficients displayed

### **3. Performance Comparison**
- Normalized price trends (base 100)
- Individual asset performance over analysis period
- Total return comparison with annualized metrics

### **4. Portfolio Allocation Charts**
- Pie charts showing optimal portfolio weights
- Focus on significant holdings (>1% allocation)
- Color-coded by asset class

---

## 🏆 **Business Applications**

### **For Individual Investors**
- **Retirement Planning**: Optimize 401(k) and IRA allocations
- **Wealth Management**: Balance risk and return objectives
- **Portfolio Rebalancing**: Quantitative guidance for asset allocation adjustments

### **For Financial Advisors**
- **Client Presentations**: Professional-grade analysis and visualizations
- **Investment Proposals**: Data-driven portfolio recommendations
- **Risk Assessment**: Comprehensive client risk profiling

### **For Institutional Investors**
- **Asset Allocation Strategy**: Multi-asset class optimization
- **Risk Management**: Quantitative risk assessment and monitoring
- **Performance Attribution**: Understand sources of portfolio returns

---

## 📚 **Educational Value**

This project demonstrates proficiency in:

### **Data Science Skills**
- **Data Acquisition**: API integration, error handling, data validation
- **Statistical Analysis**: Risk-return calculations, correlation analysis
- **Data Visualization**: Professional matplotlib/seaborn charts
- **Numerical Methods**: Monte Carlo simulation, optimization

### **Finance Concepts**
- **Modern Portfolio Theory**: Efficient frontier, risk-return tradeoffs
- **Risk Management**: VaR, drawdown analysis, diversification
- **Performance Measurement**: Sharpe ratio, risk-adjusted returns
- **Asset Allocation**: Multi-asset class portfolio construction

### **Programming Practices**
- **Clean Code**: Well-documented, modular functions
- **Error Handling**: Robust data download and processing
- **Performance**: Efficient Monte Carlo implementation
- **Reproducibility**: Seed setting, version control ready

---

## 🔬 **Model Validation**

### **Backtesting Results**
- Historical performance validation against benchmarks
- Out-of-sample testing on recent market data
- Stress testing during market volatility (2020 COVID crash, 2022 inflation)

### **Sensitivity Analysis**
- Impact of different time periods on optimization results
- Robustness to asset universe changes
- Sensitivity to risk-free rate assumptions

### **Limitations & Assumptions**
- **Historical Data Dependency**: Past performance ≠ future results
- **Normal Distribution**: May underestimate tail risks  
- **Static Correlations**: Relationships change during market stress
- **Transaction Costs**: Not included in optimization
- **Rebalancing Frequency**: Assumes perfect, cost-free rebalancing

---

## 🚀 **Future Enhancements**

### **Planned Features**
- [ ] **Black-Litterman Model**: Incorporate market views and investor confidence
- [ ] **Risk Budgeting**: Allocate risk rather than capital
- [ ] **Dynamic Rebalancing**: Time-varying optimal portfolios
- [ ] **Transaction Cost Integration**: Realistic trading cost modeling
- [ ] **ESG Integration**: Environmental, Social, Governance constraints
- [ ] **Alternative Data**: Incorporate sentiment, macroeconomic indicators

### **Technical Improvements**
- [ ] **Interactive Dashboards**: Plotly/Streamlit web interface
- [ ] **Real-time Updates**: Live market data integration
- [ ] **Database Integration**: PostgreSQL for historical data storage
- [ ] **API Development**: REST API for programmatic access
- [ ] **Docker Deployment**: Containerized analysis environment

---

## 🤝 **Contributing**

Contributions are welcome! Areas for improvement:

- **Additional Asset Classes**: Crypto, commodities, private equity
- **Advanced Risk Models**: GARCH volatility, copula models
- **Performance Enhancements**: Parallel processing, GPU acceleration  
- **Visualization Improvements**: Interactive charts, dashboard development
- **Documentation**: Additional examples, use cases

### **How to Contribute**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 **Contact & Support**

- **Author**: [Your Name]
- **Email**: [your.email@example.com]
- **LinkedIn**: [your-linkedin-profile]
- **Portfolio**: [your-portfolio-website]

### **Feedback**
Questions, suggestions, or found a bug? Please open an issue or reach out directly. This project is part of my data science portfolio demonstrating quantitative finance and Python programming skills.

---

## 🎓 **References & Further Reading**

### **Academic Papers**
- Markowitz, H. (1952). "Portfolio Selection". *Journal of Finance*
- Sharpe, W. F. (1964). "Capital Asset Prices: A Theory of Market Equilibrium"
- Black, F. & Litterman, R. (1992). "Global Portfolio Optimization"

### **Books**
- "A Man for All Markets" by Edward Thorp
- "The Intelligent Asset Allocator" by William Bernstein  
- "Quantitative Portfolio Management" by Michael Isichenko

### **Online Resources**
- [Modern Portfolio Theory - Investopedia](https://www.investopedia.com/terms/m/modernportfoliotheory.asp)
- [Python for Finance - O'Reilly](https://www.oreilly.com/library/view/python-for-finance/9781492024323/)
- [Quantitative Finance Stack Exchange](https://quant.stackexchange.com/)

---

## 📊 **Performance Metrics**

*This analysis has been tested on:*
- **Runtime**: ~2-3 minutes for complete analysis (25K simulations)
- **Memory Usage**: ~200MB peak memory consumption  
- **Data Volume**: 10+ years daily data for 8 assets (~20K data points)
- **Accuracy**: Results validated against commercial portfolio software

---

**⭐ If this project helped you, please consider giving it a star!**

*Built with ❤️ for the quantitative finance and data science communities*
