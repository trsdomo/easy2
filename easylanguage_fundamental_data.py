# =============================================================================
# FUNDAMENTAL DATA SIMULATION SYSTEM per EasyLanguage Compiler
# =============================================================================

import numpy as np
import pandas as pd
from collections import defaultdict, deque
from datetime import datetime, timedelta
import math
import warnings
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import json

@dataclass
class FundamentalDataPoint:
    """Struttura dati per un punto di dati fondamentali"""
    symbol: str
    date: datetime
    metric: str
    value: float
    period: str  # 'Q' for quarterly, 'Y' for yearly, 'TTM' for trailing twelve months
    source: str = "simulated"

class EconomicIndicatorSimulator:
    """Simulatore per indicatori economici macro"""
    
    def __init__(self):
        # Economic cycle parameters
        self.gdp_growth_trend = 0.025  # 2.5% annual GDP growth
        self.inflation_target = 0.02   # 2% inflation target
        self.unemployment_natural = 0.045  # 4.5% natural unemployment
        
        # Current state
        self.current_gdp_growth = 0.025
        self.current_inflation = 0.021
        self.current_unemployment = 0.045
        self.current_interest_rate = 0.025
        
        # Volatility parameters
        self.gdp_volatility = 0.015
        self.inflation_volatility = 0.008
        self.unemployment_volatility = 0.005
        
        # Economic cycle tracking
        self.cycle_position = 0.0  # 0-1, position in economic cycle
        self.cycle_length = 240    # ~20 years cycle in months
        
    def update_indicators(self):
        """Aggiorna gli indicatori economici con un modello realistico"""
        # Update cycle position
        self.cycle_position += 1.0 / self.cycle_length
        if self.cycle_position > 1.0:
            self.cycle_position -= 1.0
        
        # Cycle influences (sine wave with noise)
        cycle_factor = math.sin(2 * math.pi * self.cycle_position)
        
        # GDP Growth with cycle
        gdp_cycle_impact = cycle_factor * 0.02  # ±2% cycle impact
        gdp_shock = np.random.normal(0, self.gdp_volatility)
        self.current_gdp_growth = self.gdp_growth_trend + gdp_cycle_impact + gdp_shock
        self.current_gdp_growth = max(-0.05, min(0.08, self.current_gdp_growth))
        
        # Inflation with cycle and GDP correlation
        inflation_cycle = cycle_factor * 0.015  # ±1.5% cycle impact
        inflation_gdp_correlation = (self.current_gdp_growth - self.gdp_growth_trend) * 0.5
        inflation_shock = np.random.normal(0, self.inflation_volatility)
        self.current_inflation = self.inflation_target + inflation_cycle + inflation_gdp_correlation + inflation_shock
        self.current_inflation = max(-0.01, min(0.10, self.current_inflation))
        
        # Unemployment (inverse correlation with GDP)
        unemployment_cycle = -cycle_factor * 0.02  # Inverse cycle
        unemployment_gdp = -(self.current_gdp_growth - self.gdp_growth_trend) * 0.8
        unemployment_shock = np.random.normal(0, self.unemployment_volatility)
        self.current_unemployment = self.unemployment_natural + unemployment_cycle + unemployment_gdp + unemployment_shock
        self.current_unemployment = max(0.02, min(0.15, self.current_unemployment))
        
        # Interest rates (Fed response function)
        taylor_rule_rate = 0.02 + 1.5 * (self.current_inflation - 0.02) + 0.5 * (self.current_gdp_growth - 0.025)
        rate_shock = np.random.normal(0, 0.002)
        target_rate = max(0.0, min(0.10, taylor_rule_rate + rate_shock))
        
        # Smooth rate changes (Fed doesn't change rates drastically)
        rate_change = (target_rate - self.current_interest_rate) * 0.1  # 10% adjustment
        self.current_interest_rate += rate_change
        self.current_interest_rate = max(0.0, min(0.10, self.current_interest_rate))

class FundamentalDataSimulator:
    """Simulatore completo per dati fondamentali aziendali"""
    
    def __init__(self, symbol: str = "SPY", sector: str = "Technology"):
        self.symbol = symbol
        self.sector = sector
        
        # Company base metrics (realistic defaults)
        self.base_revenue = 10_000_000_000  # $10B revenue
        self.base_market_cap = 50_000_000_000  # $50B market cap
        self.base_employees = 25_000
        
        # Financial ratios (sector-specific defaults)
        self.sector_multiples = self._get_sector_multiples(sector)
        
        # Growth parameters
        self.revenue_growth_trend = 0.08  # 8% annual growth
        self.earnings_growth_trend = 0.12  # 12% annual growth
        self.margin_trend = 0.15  # 15% net margin target
        
        # Volatility parameters  
        self.revenue_volatility = 0.15
        self.earnings_volatility = 0.25
        self.margin_volatility = 0.05
        
        # Current state
        self.current_revenue = self.base_revenue
        self.current_earnings = self.base_revenue * 0.15
        self.current_book_value = self.base_market_cap * 0.3
        self.current_debt = self.base_market_cap * 0.2
        self.current_cash = self.base_market_cap * 0.1
        
        # Quarterly tracking
        self.quarterly_data = deque(maxlen=20)  # 5 years of quarters
        self.annual_data = deque(maxlen=10)     # 10 years of annual
        
        # Economic indicator simulator
        self.economic_indicators = EconomicIndicatorSimulator()
        
        # Earnings dates simulation
        self.next_earnings_date = self._generate_next_earnings_date()
        self.earnings_history = deque(maxlen=40)  # 10 years of quarterly earnings
        
    def _get_sector_multiples(self, sector: str) -> Dict[str, float]:
        """Returns typical financial ratios for different sectors"""
        sector_data = {
            "Technology": {
                "pe_ratio": 25.0, "pb_ratio": 5.0, "ps_ratio": 8.0,
                "profit_margin": 0.18, "roe": 0.20, "debt_equity": 0.15
            },
            "Healthcare": {
                "pe_ratio": 18.0, "pb_ratio": 3.5, "ps_ratio": 4.0,
                "profit_margin": 0.12, "roe": 0.15, "debt_equity": 0.25
            },
            "Financial": {
                "pe_ratio": 12.0, "pb_ratio": 1.2, "ps_ratio": 2.5,
                "profit_margin": 0.20, "roe": 0.12, "debt_equity": 0.80
            },
            "Energy": {
                "pe_ratio": 15.0, "pb_ratio": 1.5, "ps_ratio": 1.8,
                "profit_margin": 0.08, "roe": 0.10, "debt_equity": 0.40
            },
            "Consumer": {
                "pe_ratio": 20.0, "pb_ratio": 3.0, "ps_ratio": 2.0,
                "profit_margin": 0.10, "roe": 0.18, "debt_equity": 0.30
            },
            "Industrial": {
                "pe_ratio": 16.0, "pb_ratio": 2.5, "ps_ratio": 1.5,
                "profit_margin": 0.08, "roe": 0.14, "debt_equity": 0.35
            }
        }
        
        return sector_data.get(sector, sector_data["Technology"])
    
    def _generate_next_earnings_date(self) -> datetime:
        """Genera la prossima data di rilascio earnings"""
        today = datetime.now()
        
        # Find next quarter end
        current_quarter = (today.month - 1) // 3 + 1
        if current_quarter == 4:
            next_quarter_end = datetime(today.year + 1, 3, 31)
        else:
            next_month = current_quarter * 3 + 3
            next_quarter_end = datetime(today.year, next_month, 30)
        
        # Earnings typically released 2-6 weeks after quarter end
        earnings_delay = timedelta(days=random.randint(14, 42))
        return next_quarter_end + earnings_delay
    
    def update_fundamental_data(self, current_price: float):
        """Aggiorna i dati fondamentali basati su prezzo corrente e ciclo economico"""
        
        # Update economic indicators first
        self.economic_indicators.update_indicators()
        
        # Economic impact on company fundamentals
        gdp_impact = self.economic_indicators.current_gdp_growth - 0.025
        interest_rate_impact = self.economic_indicators.current_interest_rate - 0.025
        
        # Revenue growth influenced by economic conditions
        economic_revenue_impact = gdp_impact * 2.0  # Revenue sensitive to GDP
        revenue_shock = np.random.normal(0, self.revenue_volatility)
        revenue_growth = self.revenue_growth_trend + economic_revenue_impact + revenue_shock
        
        # Update revenue (quarterly)
        quarterly_revenue_growth = revenue_growth / 4.0
        self.current_revenue *= (1 + quarterly_revenue_growth)
        
        # Earnings leverage (more volatile than revenue)
        earnings_leverage = 1.5  # Operating leverage
        economic_earnings_impact = gdp_impact * 3.0 - interest_rate_impact * 1.5
        earnings_shock = np.random.normal(0, self.earnings_volatility)
        earnings_growth = self.earnings_growth_trend + economic_earnings_impact + earnings_shock
        
        quarterly_earnings_growth = earnings_growth / 4.0 * earnings_leverage
        self.current_earnings *= (1 + quarterly_earnings_growth)
        
        # Ensure earnings don't go negative (minimum 1% margin)
        min_earnings = self.current_revenue * 0.01
        self.current_earnings = max(min_earnings, self.current_earnings)
        
        # Update balance sheet items
        # Book value grows with retained earnings
        retained_earnings = self.current_earnings * 0.7  # 70% retained
        self.current_book_value += retained_earnings
        
        # Debt management based on interest rates
        if self.economic_indicators.current_interest_rate < 0.03:
            # Low rates = increase debt slightly
            self.current_debt *= 1.01
        else:
            # High rates = pay down debt
            debt_paydown = min(self.current_cash * 0.1, self.current_debt * 0.05)
            self.current_debt -= debt_paydown
            self.current_cash -= debt_paydown
        
        # Cash flow from operations
        operating_cash_flow = self.current_earnings * 1.2  # Typical OCF/earnings ratio
        self.current_cash += operating_cash_flow * 0.25  # 25% cash accumulation
        
        # Market cap based on current price and shares outstanding
        shares_outstanding = self.base_market_cap / 100.0  # Assume $100 base price
        current_market_cap = current_price * shares_outstanding
        
        # Store quarterly data point
        quarterly_data = {
            'date': datetime.now(),
            'revenue': self.current_revenue,
            'earnings': self.current_earnings,
            'book_value': self.current_book_value,
            'debt': self.current_debt,
            'cash': self.current_cash,
            'market_cap': current_market_cap,
            'shares_outstanding': shares_outstanding
        }
        
        self.quarterly_data.append(quarterly_data)
        
    def get_ttm_data(self) -> Dict[str, float]:
        """Calcola dati TTM (Trailing Twelve Months)"""
        if len(self.quarterly_data) < 4:
            return {
                'ttm_revenue': self.current_revenue * 4,
                'ttm_earnings': self.current_earnings * 4,
                'ttm_ocf': self.current_earnings * 1.2 * 4
            }
        
        # Sum last 4 quarters
        last_4_quarters = list(self.quarterly_data)[-4:]
        
        ttm_revenue = sum(q['revenue'] for q in last_4_quarters)
        ttm_earnings = sum(q['earnings'] for q in last_4_quarters)
        ttm_ocf = ttm_earnings * 1.2
        
        return {
            'ttm_revenue': ttm_revenue,
            'ttm_earnings': ttm_earnings, 
            'ttm_ocf': ttm_ocf
        }

class OptimizedFundamentalTracker:
    """Tracker ottimizzato per dati fondamentali con caching intelligente"""
    
    def __init__(self, symbol: str = "SPY", sector: str = "Technology"):
        self.symbol = symbol
        self.simulator = FundamentalDataSimulator(symbol, sector)
        
        # Cache per performance
        self._cache = {}
        self._cache_timestamp = 0
        self._cache_ttl = 100  # Cache TTL in bars
        
        # Fundamental data storage (ottimizzato)
        self.fundamental_metrics = defaultdict(float)
        
        # Update counter
        self._update_count = 0
        
        # Economic indicators cache
        self._economic_cache = {}
        
        print(f"✓ Fundamental Data Tracker initialized for {symbol} ({sector})")
    
    def update_fundamentals(self, current_price: float):
        """Update ottimizzato dei dati fondamentali"""
        self._update_count += 1
        
        # Update simulator ogni 20 bar (daily if intraday data)
        if self._update_count % 20 == 0:
            self.simulator.update_fundamental_data(current_price)
            self._invalidate_cache()
        
        # Update metrics cache
        self._update_fundamental_metrics(current_price)
    
    def _update_fundamental_metrics(self, current_price: float):
        """Aggiorna le metriche fondamentali correnti"""
        try:
            ttm_data = self.simulator.get_ttm_data()
            
            # Current shares outstanding
            shares = self.simulator.base_market_cap / 100.0
            
            # Market cap
            market_cap = current_price * shares
            
            # Earnings per share
            eps = ttm_data['ttm_earnings'] / shares if shares > 0 else 0
            
            # Book value per share
            book_value_per_share = self.simulator.current_book_value / shares if shares > 0 else 0
            
            # Cash per share
            cash_per_share = self.simulator.current_cash / shares if shares > 0 else 0
            
            # Debt per share
            debt_per_share = self.simulator.current_debt / shares if shares > 0 else 0
            
            # Financial ratios
            pe_ratio = current_price / eps if eps > 0 else 0
            pb_ratio = current_price / book_value_per_share if book_value_per_share > 0 else 0
            ps_ratio = market_cap / ttm_data['ttm_revenue'] if ttm_data['ttm_revenue'] > 0 else 0
            
            # Profitability ratios
            profit_margin = ttm_data['ttm_earnings'] / ttm_data['ttm_revenue'] if ttm_data['ttm_revenue'] > 0 else 0
            roe = ttm_data['ttm_earnings'] / self.simulator.current_book_value if self.simulator.current_book_value > 0 else 0
            debt_to_equity = self.simulator.current_debt / self.simulator.current_book_value if self.simulator.current_book_value > 0 else 0
            
            # Store metrics
            self.fundamental_metrics.update({
                'market_cap': market_cap,
                'eps': eps,
                'pe_ratio': pe_ratio,
                'pb_ratio': pb_ratio,
                'ps_ratio': ps_ratio,
                'book_value_per_share': book_value_per_share,
                'cash_per_share': cash_per_share,
                'debt_per_share': debt_per_share,
                'profit_margin': profit_margin,
                'roe': roe,
                'debt_to_equity': debt_to_equity,
                'revenue': ttm_data['ttm_revenue'],
                'earnings': ttm_data['ttm_earnings'],
                'operating_cash_flow': ttm_data['ttm_ocf'],
                'current_price': current_price
            })
            
        except Exception as e:
            warnings.warn(f"Error updating fundamental metrics: {e}")
    
    def _invalidate_cache(self):
        """Invalida la cache quando i dati cambiano"""
        self._cache.clear()
        self._cache_timestamp = self._update_count
    
    def get_cached_metric(self, metric_name: str) -> Optional[float]:
        """Recupero ottimizzato con cache"""
        cache_key = f"{metric_name}_{self._update_count // self._cache_ttl}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Calculate and cache
        value = self.fundamental_metrics.get(metric_name, 0.0)
        self._cache[cache_key] = value
        return value

# =============================================================================
# METODI DA AGGIUNGERE ALLA CLASSE CompiledStrategy
# =============================================================================

def __init_fundamental_data__(self, symbol: str = "SPY", sector: str = "Technology"):
    """
    Inizializzazione sistema dati fondamentali
    Chiamare in __init__() della strategia
    """
    self.fundamental_tracker = OptimizedFundamentalTracker(symbol, sector)
    
    # Configuration
    self.fundamental_update_frequency = 20  # Update every N bars
    self.enable_economic_indicators = True
    self.enable_earnings_simulation = True
    
    # Performance tracking
    self._fundamental_update_count = 0
    
    print("✓ Fundamental Data System initialized")

def update_fundamental_data(self):
    """
    Aggiorna dati fondamentali - chiamare in next()
    """
    try:
        current_price = float(self.data.close[0])
        self.fundamental_tracker.update_fundamentals(current_price)
        self._fundamental_update_count += 1
        
    except Exception as e:
        warnings.warn(f"Error updating fundamental data: {e}")

# =============================================================================
# FUNZIONI EASYLANGUAGE - FUNDAMENTAL DATA
# =============================================================================

def GetFundData(self, metric: str, period: str = "TTM") -> float:
    """
    Funzione principale per ottenere dati fondamentali
    EasyLanguage: GetFundData("EPS", "TTM")
    
    Supported metrics:
    - EPS, PE, PB, PS (ratios)
    - Revenue, Earnings, Cash, Debt
    - ROE, ProfitMargin, DebtEquity
    - MarketCap, BookValue
    """
    try:
        metric_lower = metric.lower()
        
        # Map EasyLanguage names to internal names
        metric_map = {
            'eps': 'eps',
            'pe': 'pe_ratio',
            'pb': 'pb_ratio', 
            'ps': 'ps_ratio',
            'revenue': 'revenue',
            'earnings': 'earnings',
            'cash': 'cash_per_share',
            'debt': 'debt_per_share',
            'roe': 'roe',
            'profitmargin': 'profit_margin',
            'debtequity': 'debt_to_equity',
            'marketcap': 'market_cap',
            'bookvalue': 'book_value_per_share',
            'operatingcashflow': 'operating_cash_flow',
            'ocf': 'operating_cash_flow'
        }
        
        internal_metric = metric_map.get(metric_lower)
        if internal_metric:
            return float(self.fundamental_tracker.get_cached_metric(internal_metric))
        else:
            warnings.warn(f"Unknown fundamental metric: {metric}")
            return 0.0
            
    except Exception as e:
        warnings.warn(f"Error in GetFundData({metric}): {e}")
        return 0.0

def EPS(self, period: str = "TTM") -> float:
    """
    Earnings Per Share
    EasyLanguage: EPS or EPS("TTM")
    """
    return self.GetFundData("EPS", period)

def PERatio(self) -> float:
    """
    Price to Earnings Ratio
    EasyLanguage: PERatio
    """
    return self.GetFundData("PE")

def PBRatio(self) -> float:
    """
    Price to Book Ratio
    EasyLanguage: PBRatio
    """
    return self.GetFundData("PB")

def PSRatio(self) -> float:
    """
    Price to Sales Ratio
    EasyLanguage: PSRatio
    """
    return self.GetFundData("PS")

def ROE(self) -> float:
    """
    Return on Equity
    EasyLanguage: ROE
    """
    return self.GetFundData("ROE") * 100  # Return as percentage

def ProfitMargin(self) -> float:
    """
    Net Profit Margin
    EasyLanguage: ProfitMargin
    """
    return self.GetFundData("ProfitMargin") * 100  # Return as percentage

def DebtToEquity(self) -> float:
    """
    Debt to Equity Ratio
    EasyLanguage: DebtToEquity
    """
    return self.GetFundData("DebtEquity")

def MarketCap(self) -> float:
    """
    Market Capitalization
    EasyLanguage: MarketCap
    """
    return self.GetFundData("MarketCap")

def BookValuePerShare(self) -> float:
    """
    Book Value Per Share
    EasyLanguage: BookValuePerShare
    """
    return self.GetFundData("BookValue")

def CashPerShare(self) -> float:
    """
    Cash Per Share
    EasyLanguage: CashPerShare
    """
    return self.GetFundData("Cash")

def DebtPerShare(self) -> float:
    """
    Debt Per Share
    EasyLanguage: DebtPerShare
    """
    return self.GetFundData("Debt")

def Revenue(self, period: str = "TTM") -> float:
    """
    Total Revenue (TTM)
    EasyLanguage: Revenue or Revenue("TTM")
    """
    return self.GetFundData("Revenue", period)

def NetIncome(self, period: str = "TTM") -> float:
    """
    Net Income (TTM)
    EasyLanguage: NetIncome or NetIncome("TTM")
    """
    return self.GetFundData("Earnings", period)

def OperatingCashFlow(self, period: str = "TTM") -> float:
    """
    Operating Cash Flow (TTM)
    EasyLanguage: OperatingCashFlow
    """
    return self.GetFundData("OCF", period)

# =============================================================================
# ECONOMIC INDICATORS FUNCTIONS
# =============================================================================

def GDPGrowth(self) -> float:
    """
    Current GDP Growth Rate (annualized)
    EasyLanguage: GDPGrowth
    """
    try:
        return self.fundamental_tracker.simulator.economic_indicators.current_gdp_growth * 100
    except:
        return 2.5  # Default 2.5%

def InflationRate(self) -> float:
    """
    Current Inflation Rate (annualized)
    EasyLanguage: InflationRate
    """
    try:
        return self.fundamental_tracker.simulator.economic_indicators.current_inflation * 100
    except:
        return 2.0  # Default 2%

def UnemploymentRate(self) -> float:
    """
    Current Unemployment Rate
    EasyLanguage: UnemploymentRate
    """
    try:
        return self.fundamental_tracker.simulator.economic_indicators.current_unemployment * 100
    except:
        return 4.5  # Default 4.5%

def InterestRate(self) -> float:
    """
    Current Risk-free Interest Rate
    EasyLanguage: InterestRate
    """
    try:
        return self.fundamental_tracker.simulator.economic_indicators.current_interest_rate * 100
    except:
        return 2.5  # Default 2.5%

# =============================================================================
# VALUATION & SCREENING FUNCTIONS
# =============================================================================

def PEGRatio(self, growth_rate: float = None) -> float:
    """
    PEG Ratio (PE / Growth Rate)
    EasyLanguage: PEGRatio or PEGRatio(15.0)
    """
    try:
        pe = self.PERatio()
        
        if growth_rate is None:
            # Estimate growth rate from historical data
            growth_rate = 12.0  # Default 12% earnings growth
        
        if growth_rate > 0 and pe > 0:
            return pe / growth_rate
        else:
            return 0.0
    except:
        return 0.0

def DividendYield(self) -> float:
    """
    Dividend Yield (simulated)
    EasyLanguage: DividendYield
    """
    try:
        # Simulate dividend yield based on sector and PE ratio
        pe = self.PERatio()
        sector_yield = self.fundamental_tracker.simulator.sector_multiples.get('dividend_yield', 0.02)
        
        # Higher PE typically means lower dividend yield
        if pe > 0:
            estimated_yield = max(0.0, sector_yield * (20.0 / max(pe, 10.0)))
            return estimated_yield * 100  # Return as percentage
        else:
            return sector_yield * 100
    except:
        return 2.0  # Default 2%

def EarningsYield(self) -> float:
    """
    Earnings Yield (1 / PE Ratio)
    EasyLanguage: EarningsYield  
    """
    try:
        pe = self.PERatio()
        if pe > 0:
            return (1.0 / pe) * 100  # Return as percentage
        else:
            return 0.0
    except:
        return 0.0

def PriceToFairValue(self, discount_rate: float = 10.0) -> float:
    """
    Price to Fair Value using DCF model
    EasyLanguage: PriceToFairValue or PriceToFairValue(10.0)
    """
    try:
        # Simplified DCF calculation
        ocf = self.OperatingCashFlow()
        growth_rate = 8.0  # Assumed growth rate
        terminal_growth = 3.0  # Terminal growth rate
        
        if ocf > 0:
            # 5-year projection + terminal value
            discount_factor = discount_rate / 100.0
            growth_factor = growth_rate / 100.0
            terminal_factor = terminal_growth / 100.0
            
            # NPV of 5 years of cash flows
            npv_5yr = 0
            for year in range(1, 6):
                cf = ocf * ((1 + growth_factor) ** year)
                pv = cf / ((1 + discount_factor) ** year)
                npv_5yr += pv
            
            # Terminal value
            terminal_cf = ocf * ((1 + growth_factor) ** 5) * (1 + terminal_factor)
            terminal_value = terminal_cf / (discount_factor - terminal_factor)
            pv_terminal = terminal_value / ((1 + discount_factor) ** 5)
            
            # Total fair value per share
            total_fair_value = (npv_5yr + pv_terminal)
            shares = self.fundamental_tracker.simulator.base_market_cap / 100.0
            fair_value_per_share = total_fair_value / shares
            
            current_price = self.fundamental_tracker.fundamental_metrics.get('current_price', 100.0)
            
            if fair_value_per_share > 0:
                return current_price / fair_value_per_share
            else:
                return 1.0
        else:
            return 1.0
    except:
        return 1.0

def IsValueStock(self, pe_threshold: float = 15.0, pb_threshold: float = 2.0) -> bool:
    """
    Determines if stock meets value criteria
    EasyLanguage: IsValueStock or IsValueStock(15.0, 2.0)
    """
    try:
        pe = self.PERatio()
        pb = self.PBRatio()
        debt_equity = self.DebtToEquity()
        
        # Value criteria
        low_pe = pe > 0 and pe < pe_threshold
        low_pb = pb > 0 and pb < pb_threshold
        reasonable_debt = debt_equity < 0.6  # Less than 60% debt/equity
        
        return low_pe and low_pb and reasonable_debt
    except:
        return False

