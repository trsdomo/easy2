# ========================================================================
# QUOTE FIELDS & MARKET DATA FUNCTIONS - EasyLanguage for Backtrader
# Simulazione avanzata di dati di mercato e microstructure
# ========================================================================

import math
import numpy as np
from collections import deque, defaultdict
from typing import Optional, List, Tuple, Union, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import random

@dataclass
class QuoteData:
    """Struttura per dati di quotazione"""
    bid: float = 0.0
    ask: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0
    last: float = 0.0
    last_size: float = 0.0
    timestamp: Optional[datetime] = None
    
@dataclass
class MarketDepth:
    """Struttura per profondità di mercato (Level II)"""
    bids: List[Tuple[float, float]] = field(default_factory=list)  # [(price, size), ...]
    asks: List[Tuple[float, float]] = field(default_factory=list)  # [(price, size), ...]
    
@dataclass 
class VWAPData:
    """Struttura per calcoli VWAP"""
    cumulative_volume: float = 0.0
    cumulative_pv: float = 0.0  # price * volume
    vwap: float = 0.0
    
@dataclass
class TickData:
    """Struttura per dati tick"""
    price: float
    size: float
    timestamp: datetime
    side: str  # 'buy', 'sell', 'unknown'
    
class MarketDataSimulator:
    """Simulatore di dati di mercato realistici"""
    
    def __init__(self, base_spread_pct: float = 0.001, volatility: float = 0.02):
        self.base_spread_pct = base_spread_pct  # Spread base come % del prezzo
        self.volatility = volatility
        self.quote_history = deque(maxlen=1000)
        self.tick_history = deque(maxlen=5000)
        self.last_quote = QuoteData()
        
        # Parametri per simulazione realistica
        self.spread_volatility = 0.3  # Variabilità dello spread
        self.size_mean = 100.0
        self.size_std = 50.0
        
    def generate_quote(self, mid_price: float, volume: float = 0.0) -> QuoteData:
        """Genera dati di quotazione realistici"""
        try:
            # Calcola spread dinamico basato su volatilità e volume
            base_spread = mid_price * self.base_spread_pct
            
            # Spread più ampio durante bassa liquidità
            volume_factor = max(0.5, 1.0 - (volume / 10000.0))
            dynamic_spread = base_spread * (1.0 + random.gauss(0, self.spread_volatility)) * volume_factor
            dynamic_spread = max(0.01, dynamic_spread)  # Minimo 1 cent
            
            # Calcola bid/ask
            half_spread = dynamic_spread / 2.0
            bid = mid_price - half_spread
            ask = mid_price + half_spread
            
            # Simula size realistici
            bid_size = max(1, random.gauss(self.size_mean, self.size_std))
            ask_size = max(1, random.gauss(self.size_mean, self.size_std))
            
            quote = QuoteData(
                bid=round(bid, 4),
                ask=round(ask, 4),
                bid_size=round(bid_size),
                ask_size=round(ask_size),
                last=mid_price,
                last_size=volume if volume > 0 else random.gauss(self.size_mean, self.size_std),
                timestamp=datetime.now()
            )
            
            self.last_quote = quote
            self.quote_history.append(quote)
            
            return quote
            
        except Exception:
            # Fallback to simple quote
            return QuoteData(
                bid=mid_price - 0.01,
                ask=mid_price + 0.01,
                bid_size=100,
                ask_size=100,
                last=mid_price,
                last_size=volume if volume > 0 else 100
            )

class MarketDataTracker:
    """Helper class per tracking di dati di mercato avanzati"""
    
    def __init__(self, max_history: int = 1000):
        self.simulator = MarketDataSimulator()
        self.vwap_data = VWAPData()
        self.twap_data = {'prices': deque(maxlen=max_history), 'count': 0}
        
        # High/Low tracking 
        self.high_52wk = 0.0
        self.low_52wk = float('inf')
        self.high_52wk_date = None
        self.low_52wk_date = None
        
        # Volume analysis
        self.volume_history = deque(maxlen=max_history)
        self.volume_profile = defaultdict(float)  # price -> volume
        
        # Market depth simulation
        self.market_depth = MarketDepth()
        self.depth_levels = 5  # Number of levels to simulate
        
        # Tick data
        self.tick_data = deque(maxlen=5000)
        self.last_trade_price = 0.0
        self.last_trade_size = 0.0
        
        # Market microstructure
        self.order_flow = {'buy_volume': 0.0, 'sell_volume': 0.0}
        self.trade_count = 0
        self.tick_direction = 0  # +1 uptick, -1 downtick, 0 neutral
        
        # Intraday reset tracking
        self.session_start_time = None
        self.daily_vwap_data = VWAPData()
        
    def update_market_data(self, price: float, volume: float, timestamp: Optional[datetime] = None):
        """Aggiorna tutti i dati di mercato"""
        if timestamp is None:
            timestamp = datetime.now()
            
        # Reset daily data if new session
        self._check_session_reset(timestamp)
        
        # Update quote data
        quote = self.simulator.generate_quote(price, volume)
        
        # Update VWAP
        self._update_vwap(price, volume)
        
        # Update TWAP
        self._update_twap(price)
        
        # Update 52-week high/low
        self._update_52week_extremes(price, timestamp)
        
        # Update volume analysis
        self._update_volume_analysis(price, volume)
        
        # Update market depth
        self._update_market_depth(quote)
        
        # Update tick data
        self._update_tick_data(price, volume, timestamp)
        
        # Update order flow
        self._update_order_flow(price, volume)
        
    def _check_session_reset(self, timestamp: datetime):
        """Controlla se inizia una nuova sessione e resetta dati intraday"""
        if self.session_start_time is None:
            self.session_start_time = timestamp.replace(hour=9, minute=30, second=0, microsecond=0)
        
        # Se è una nuova sessione (dopo le 4 AM del giorno successivo)
        if timestamp.date() > self.session_start_time.date():
            self.daily_vwap_data = VWAPData()
            self.session_start_time = timestamp.replace(hour=9, minute=30, second=0, microsecond=0)
            self.order_flow = {'buy_volume': 0.0, 'sell_volume': 0.0}
            self.trade_count = 0
    
    def _update_vwap(self, price: float, volume: float):
        """Aggiorna calcoli VWAP"""
        if volume > 0:
            # VWAP cumulativo
            self.vwap_data.cumulative_volume += volume
            self.vwap_data.cumulative_pv += price * volume
            
            if self.vwap_data.cumulative_volume > 0:
                self.vwap_data.vwap = self.vwap_data.cumulative_pv / self.vwap_data.cumulative_volume
            
            # VWAP giornaliero
            self.daily_vwap_data.cumulative_volume += volume
            self.daily_vwap_data.cumulative_pv += price * volume
            
            if self.daily_vwap_data.cumulative_volume > 0:
                self.daily_vwap_data.vwap = self.daily_vwap_data.cumulative_pv / self.daily_vwap_data.cumulative_volume
    
    def _update_twap(self, price: float):
        """Aggiorna calcoli TWAP"""
        self.twap_data['prices'].append(price)
        self.twap_data['count'] += 1
    
    def _update_52week_extremes(self, price: float, timestamp: datetime):
        """Aggiorna massimi e minimi 52 settimane"""
        if price > self.high_52wk:
            self.high_52wk = price
            self.high_52wk_date = timestamp
        
        if price < self.low_52wk:
            self.low_52wk = price
            self.low_52wk_date = timestamp
    
    def _update_volume_analysis(self, price: float, volume: float):
        """Aggiorna analisi volume"""
        if volume > 0:
            self.volume_history.append(volume)
            # Volume profile (prezzo arrotondato)
            price_level = round(price, 2)
            self.volume_profile[price_level] += volume
    
    def _update_market_depth(self, quote: QuoteData):
        """Simula market depth realistico"""
        try:
            mid_price = (quote.bid + quote.ask) / 2.0
            tick_size = 0.01
            
            # Genera bid levels
            self.market_depth.bids = []
            for i in range(self.depth_levels):
                level_price = quote.bid - (i * tick_size)
                level_size = quote.bid_size * (0.8 ** i) * random.uniform(0.5, 1.5)
                self.market_depth.bids.append((round(level_price, 4), round(level_size)))
            
            # Genera ask levels
            self.market_depth.asks = []
            for i in range(self.depth_levels):
                level_price = quote.ask + (i * tick_size)
                level_size = quote.ask_size * (0.8 ** i) * random.uniform(0.5, 1.5)
                self.market_depth.asks.append((round(level_price, 4), round(level_size)))
                
        except Exception:
            # Fallback depth
            self.market_depth.bids = [(quote.bid, quote.bid_size)]
            self.market_depth.asks = [(quote.ask, quote.ask_size)]
    
    def _update_tick_data(self, price: float, volume: float, timestamp: datetime):
        """Aggiorna dati tick e determina direzione"""
        # Determina tick direction
        if price > self.last_trade_price:
            side = 'buy'
            self.tick_direction = 1
        elif price < self.last_trade_price:
            side = 'sell'
            self.tick_direction = -1
        else:
            side = 'unknown'
            self.tick_direction = 0
        
        tick = TickData(price=price, size=volume, timestamp=timestamp, side=side)
        self.tick_data.append(tick)
        
        self.last_trade_price = price
        self.last_trade_size = volume
        self.trade_count += 1
    
    def _update_order_flow(self, price: float, volume: float):
        """Aggiorna order flow analysis"""
        if self.tick_direction > 0:  # Uptick = buy
            self.order_flow['buy_volume'] += volume
        elif self.tick_direction < 0:  # Downtick = sell
            self.order_flow['sell_volume'] += volume

# ========================================================================
# METODI DA AGGIUNGERE ALLA CLASSE CompiledStrategy
# ========================================================================

def __init_market_data_tracking__(self, max_history: int = 1000):
    """Inizializza il sistema di tracking dati di mercato"""
    self._market_data_tracker = MarketDataTracker(max_history)
    self._market_data_initialized = False

def _update_market_data(self):
    """Aggiorna i dati di mercato (chiamare in next())"""
    try:
        current_price = float(self.data.close[0])
        current_volume = float(self.data.volume[0]) if hasattr(self.data, 'volume') else 0.0
        current_datetime = self.data.datetime.datetime(0)
        
        self._market_data_tracker.update_market_data(current_price, current_volume, current_datetime)
        self._market_data_initialized = True
        
    except Exception as e:
        self._handle_error(f"Market data update error: {e}")

# ========================================================================
# INSIDE BID/ASK FUNCTIONS
# ========================================================================

def InsideBid(self) -> float:
    """Restituisce il miglior prezzo bid corrente"""
    try:
        if not self._market_data_initialized:
            return 0.0
        
        quote = self._market_data_tracker.simulator.last_quote
        return quote.bid
        
    except Exception as e:
        self._handle_error(f"InsideBid error: {e}")
        return 0.0

def InsideAsk(self) -> float:
    """Restituisce il miglior prezzo ask corrente"""
    try:
        if not self._market_data_initialized:
            return 0.0
        
        quote = self._market_data_tracker.simulator.last_quote
        return quote.ask
        
    except Exception as e:
        self._handle_error(f"InsideAsk error: {e}")
        return 0.0

def BidSize(self) -> float:
    """Restituisce la size del miglior bid"""
    try:
        if not self._market_data_initialized:
            return 0.0
        
        quote = self._market_data_tracker.simulator.last_quote
        return quote.bid_size
        
    except Exception as e:
        self._handle_error(f"BidSize error: {e}")
        return 0.0

def AskSize(self) -> float:
    """Restituisce la size del miglior ask"""
    try:
        if not self._market_data_initialized:
            return 0.0
        
        quote = self._market_data_tracker.simulator.last_quote
        return quote.ask_size
        
    except Exception as e:
        self._handle_error(f"AskSize error: {e}")
        return 0.0

def BidAskSpread(self) -> float:
    """Restituisce lo spread bid-ask"""
    try:
        bid = self.InsideBid()
        ask = self.InsideAsk()
        
        return ask - bid if ask > bid else 0.0
        
    except Exception as e:
        self._handle_error(f"BidAskSpread error: {e}")
        return 0.0

def BidAskSpreadPct(self) -> float:
    """Restituisce lo spread bid-ask come percentuale"""
    try:
        spread = self.BidAskSpread()
        mid_price = (self.InsideBid() + self.InsideAsk()) / 2.0
        
        if mid_price > 0:
            return (spread / mid_price) * 100.0
        return 0.0
        
    except Exception as e:
        self._handle_error(f"BidAskSpreadPct error: {e}")
        return 0.0

def MidPoint(self) -> float:
    """Restituisce il mid-point tra bid e ask"""
    try:
        bid = self.InsideBid()
        ask = self.InsideAsk()
        
        return (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0
        
    except Exception as e:
        self._handle_error(f"MidPoint error: {e}")
        return 0.0

# ========================================================================
# VWAP & TWAP FUNCTIONS
# ========================================================================

def VWAP(self) -> float:
    """Restituisce il Volume Weighted Average Price cumulativo"""
    try:
        if not self._market_data_initialized:
            return 0.0
        
        return self._market_data_tracker.vwap_data.vwap
        
    except Exception as e:
        self._handle_error(f"VWAP error: {e}")
        return 0.0

def DailyVWAP(self) -> float:
    """Restituisce il VWAP della sessione corrente"""
    try:
        if not self._market_data_initialized:
            return 0.0
        
        return self._market_data_tracker.daily_vwap_data.vwap
        
    except Exception as e:
        self._handle_error(f"DailyVWAP error: {e}")
        return 0.0

def TWAP(self, length: int) -> float:
    """Restituisce il Time Weighted Average Price"""
    try:
        if not self._market_data_initialized or length <= 0:
            return 0.0
        
        prices = list(self._market_data_tracker.twap_data['prices'])
        if len(prices) == 0:
            return 0.0
        
        # Prendi ultimi N prezzi
        relevant_prices = prices[-length:] if len(prices) >= length else prices
        
        return sum(relevant_prices) / len(relevant_prices)
        
    except Exception as e:
        self._handle_error(f"TWAP error: {e}")
        return 0.0

def VWAPDeviation(self) -> float:
    """Restituisce la deviazione dal VWAP come percentuale"""
    try:
        current_price = float(self.data.close[0])
        vwap = self.VWAP()
        
        if vwap > 0:
            return ((current_price - vwap) / vwap) * 100.0
        return 0.0
        
    except Exception as e:
        self._handle_error(f"VWAPDeviation error: {e}")
        return 0.0

def VWAPCrossover(self) -> int:
    """Restituisce direzione crossover con VWAP (1=above, -1=below, 0=on)"""
    try:
        current_price = float(self.data.close[0])
        vwap = self.VWAP()
        
        if vwap == 0:
            return 0
        
        if current_price > vwap * 1.001:  # 0.1% buffer per evitare noise
            return 1
        elif current_price < vwap * 0.999:
            return -1
        else:
            return 0
        
    except Exception as e:
        self._handle_error(f"VWAPCrossover error: {e}")
        return 0

# ========================================================================
# 52-WEEK HIGH/LOW FUNCTIONS
# ========================================================================

def High52Wk(self) -> float:
    """Restituisce il massimo delle ultime 52 settimane"""
    try:
        if not self._market_data_initialized:
            return 0.0
        
        return self._market_data_tracker.high_52wk
        
    except Exception as e:
        self._handle_error(f"High52Wk error: {e}")
        return 0.0

def Low52Wk(self) -> float:
    """Restituisce il minimo delle ultime 52 settimane"""
    try:
        if not self._market_data_initialized:
            return float('inf')
        
        low_52wk = self._market_data_tracker.low_52wk
        return low_52wk if low_52wk != float('inf') else 0.0
        
    except Exception as e:
        self._handle_error(f"Low52Wk error: {e}")
        return 0.0

def High52WkDate(self) -> float:
    """Restituisce la data del massimo 52 settimane in formato EasyLanguage"""
    try:
        if not self._market_data_initialized or not self._market_data_tracker.high_52wk_date:
            return 0.0
        
        date = self._market_data_tracker.high_52wk_date
        return float(f"{date.year:04d}{date.month:02d}{date.day:02d}")
        
    except Exception as e:
        self._handle_error(f"High52WkDate error: {e}")
        return 0.0

def Low52WkDate(self) -> float:
    """Restituisce la data del minimo 52 settimane in formato EasyLanguage"""
    try:
        if not self._market_data_initialized or not self._market_data_tracker.low_52wk_date:
            return 0.0
        
        date = self._market_data_tracker.low_52wk_date
        return float(f"{date.year:04d}{date.month:02d}{date.day:02d}")
        
    except Exception as e:
        self._handle_error(f"Low52WkDate error: {e}")
        return 0.0

def DistanceFromHigh52Wk(self) -> float:
    """Restituisce la distanza percentuale dal massimo 52 settimane"""
    try:
        current_price = float(self.data.close[0])
        high_52wk = self.High52Wk()
        
        if high_52wk > 0:
            return ((current_price - high_52wk) / high_52wk) * 100.0
        return 0.0
        
    except Exception as e:
        self._handle_error(f"DistanceFromHigh52Wk error: {e}")
        return 0.0

def DistanceFromLow52Wk(self) -> float:
    """Restituisce la distanza percentuale dal minimo 52 settimane"""
    try:
        current_price = float(self.data.close[0])
        low_52wk = self.Low52Wk()
        
        if low_52wk > 0:
            return ((current_price - low_52wk) / low_52wk) * 100.0
        return 0.0
        
    except Exception as e:
        self._handle_error(f"DistanceFromLow52Wk error: {e}")
        return 0.0

def IsNew52WkHigh(self) -> bool:
    """Verifica se il prezzo corrente è un nuovo massimo 52 settimane"""
    try:
        current_price = float(self.data.close[0])
        high_52wk = self.High52Wk()
        
        # Considera nuovo high se supera il precedente di almeno 0.01%
        return current_price > high_52wk * 1.0001
        
    except Exception as e:
        self._handle_error(f"IsNew52WkHigh error: {e}")
        return False

def IsNew52WkLow(self) -> bool:
    """Verifica se il prezzo corrente è un nuovo minimo 52 settimane"""
    try:
        current_price = float(self.data.close[0])
        low_52wk = self.Low52Wk()
        
        # Considera nuovo low se scende sotto il precedente di almeno 0.01%
        return current_price < low_52wk * 0.9999 and low_52wk != float('inf')
        
    except Exception as e:
        self._handle_error(f"IsNew52WkLow error: {e}")
        return False

# ========================================================================
# VOLUME ANALYSIS FUNCTIONS
# ========================================================================

def AverageVolume(self, length: int) -> float:
    """Restituisce il volume medio degli ultimi N periodi"""
    try:
        if length <= 0 or not self._market_data_initialized:
            return 0.0
        
        volumes = list(self._market_data_tracker.volume_history)
        if len(volumes) == 0:
            return 0.0
        
        relevant_volumes = volumes[-length:] if len(volumes) >= length else volumes
        return sum(relevant_volumes) / len(relevant_volumes)
        
    except Exception as e:
        self._handle_error(f"AverageVolume error: {e}")
        return 0.0

def RelativeVolume(self, length: int = 20) -> float:
    """Restituisce il volume relativo (corrente vs media)"""
    try:
        current_volume = float(self.data.volume[0]) if hasattr(self.data, 'volume') else 0.0
        avg_volume = self.AverageVolume(length)
        
        if avg_volume > 0:
            return current_volume / avg_volume
        return 0.0
        
    except Exception as e:
        self._handle_error(f"RelativeVolume error: {e}")
        return 0.0

def VolumeRatio(self, length: int = 10) -> float:
    """Restituisce il ratio volume corrente vs precedente periodo"""
    try:
        if not self._market_data_initialized or length <= 0:
            return 0.0
        
        volumes = list(self._market_data_tracker.volume_history)
        if len(volumes) < length * 2:
            return 0.0
        
        current_period = volumes[-length:]
        previous_period = volumes[-length*2:-length]
        
        current_avg = sum(current_period) / len(current_period)
        previous_avg = sum(previous_period) / len(previous_period)
        
        if previous_avg > 0:
            return current_avg / previous_avg
        return 0.0
        
    except Exception as e:
        self._handle_error(f"VolumeRatio error: {e}")
        return 0.0

def VolumeBuzz(self, threshold: float = 2.0, length: int = 20) -> bool:
    """Verifica se c'è un volume spike significativo"""
    try:
        relative_vol = self.RelativeVolume(length)
        return relative_vol > threshold
        
    except Exception as e:
        self._handle_error(f"VolumeBuzz error: {e}")
        return False

def VolumeWeightedPrice(self, price_level: float, tolerance: float = 0.01) -> float:
    """Restituisce il volume a un livello di prezzo specifico"""
    try:
        if not self._market_data_initialized:
            return 0.0
        
        target_level = round(price_level, 2)
        total_volume = 0.0
        
        # Cerca in un range di tolleranza
        for level, volume in self._market_data_tracker.volume_profile.items():
            if abs(level - target_level) <= tolerance:
                total_volume += volume
        
        return total_volume
        
    except Exception as e:
        self._handle_error(f"VolumeWeightedPrice error: {e}")
        return 0.0

def MaxVolumePrice(self) -> float:
    """Restituisce il livello di prezzo con il volume massimo (POC - Point of Control)"""
    try:
        if not self._market_data_initialized:
            return 0.0
        
        if not self._market_data_tracker.volume_profile:
            return 0.0
        
        max_volume = 0.0
        max_price = 0.0
        
        for price, volume in self._market_data_tracker.volume_profile.items():
            if volume > max_volume:
                max_volume = volume
                max_price = price
        
        return max_price
        
    except Exception as e:
        self._handle_error(f"MaxVolumePrice error: {e}")
        return 0.0

# ========================================================================
# MARKET DEPTH FUNCTIONS (LEVEL II)
# ========================================================================

def BidDepth(self, level: int = 1) -> Tuple[float, float]:
    """Restituisce prezzo e size per il livello bid specificato"""
    try:
        if not self._market_data_initialized or level <= 0:
            return 0.0, 0.0
        
        bids = self._market_data_tracker.market_depth.bids
        if level <= len(bids):
            return bids[level - 1]  # level 1 = index 0
        return 0.0, 0.0
        
    except Exception as e:
        self._handle_error(f"BidDepth error: {e}")
        return 0.0, 0.0

def AskDepth(self, level: int = 1) -> Tuple[float, float]:
    """Restituisce prezzo e size per il livello ask specificato"""
    try:
        if not self._market_data_initialized or level <= 0:
            return 0.0, 0.0
        
        asks = self._market_data_tracker.market_depth.asks
        if level <= len(asks):
            return asks[level - 1]  # level 1 = index 0
        return 0.0, 0.0
        
    except Exception as e:
        self._handle_error(f"AskDepth error: {e}")
        return 0.0, 0.0

def TotalBidSize(self, levels: int = 5) -> float:
    """Restituisce la size totale dei primi N livelli bid"""
    try:
        if not self._market_data_initialized or levels <= 0:
            return 0.0
        
        total_size = 0.0
        bids = self._market_data_tracker.market_depth.bids
        
        for i in range(min(levels, len(bids))):
            _, size = bids[i]
            total_size += size
        
        return total_size
        
    except Exception as e:
        self._handle_error(f"TotalBidSize error: {e}")
        return 0.0
