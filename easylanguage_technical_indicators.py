# =============================================================================
# TECHNICAL ANALYSIS INDICATORS - EASYLANGUAGE INTEGRATION
# Compatible with Backtrader when available
# =============================================================================

import numpy as np
import math
from collections import deque
import warnings
from functools import lru_cache

class TechnicalIndicatorTracker:
    """
    Ottimizzato per calcoli di indicatori tecnici con caching intelligente
    """
    def __init__(self, max_history=500):
        self.max_history = max_history
        
        # Circular buffers per performance
        self._price_buffer = deque(maxlen=max_history)
        self._high_buffer = deque(maxlen=max_history)
        self._low_buffer = deque(max_history)
        self._close_buffer = deque(maxlen=max_history)
        self._volume_buffer = deque(maxlen=max_history)
        
        # Cache per indicatori computazionalmente pesanti
        self._indicator_cache = {}
        self._cache_timestamp = 0
        
        # RSI tracking
        self._rsi_gains = deque(maxlen=max_history)
        self._rsi_losses = deque(maxlen=max_history)
        self._rsi_avg_gain = 0.0
        self._rsi_avg_loss = 0.0
        
        # MACD tracking
        self._ema_fast = None
        self._ema_slow = None
        self._macd_signal = None
        
        # Stochastic tracking
        self._stoch_k_buffer = deque(maxlen=100)
        self._stoch_d_buffer = deque(maxlen=100)
        
        # Moving Average cache
        self._ma_cache = {}
        
    def add_bar_data(self, open_price, high, low, close, volume=1000):
        """Aggiunge i dati di una barra e aggiorna i tracker"""
        self._price_buffer.append(close)
        self._high_buffer.append(high)
        self._low_buffer.append(low)
        self._close_buffer.append(close)
        self._volume_buffer.append(volume)
        
        # Update RSI components
        if len(self._close_buffer) > 1:
            change = close - self._close_buffer[-2]
            gain = max(0, change)
            loss = max(0, -change)
            self._rsi_gains.append(gain)
            self._rsi_losses.append(loss)
        
        # Invalidate cache
        self._cache_timestamp += 1
        self._indicator_cache.clear()
        self._ma_cache.clear()

def __init_technical_indicators__(self):
    """
    Inizializzazione sistema indicatori tecnici ottimizzato
    """
    self.tech_tracker = TechnicalIndicatorTracker()
    
    # Integrazione con Backtrader
    self._bt_indicators = {}
    self._use_backtrader = True
    
    # Performance cache
    self._last_bar_processed = -1
    
    print("✓ Technical Analysis Indicators initialized")

def _update_technical_indicators(self):
    """
    Aggiorna i dati degli indicatori con la barra corrente
    """
    try:
        if hasattr(self, 'data') and len(self.data.close) > 0:
            current_bar = len(self.data.close) - 1
            
            # Skip se già processato
            if current_bar == self._last_bar_processed:
                return
                
            open_price = float(self.data.open[0])
            high = float(self.data.high[0])
            low = float(self.data.low[0])
            close = float(self.data.close[0])
            volume = float(self.data.volume[0]) if hasattr(self.data, 'volume') else 1000.0
            
            self.tech_tracker.add_bar_data(open_price, high, low, close, volume)
            self._last_bar_processed = current_bar
            
    except Exception as e:
        print(f"Error updating technical indicators: {e}")

# =============================================================================
# MOMENTUM INDICATORS
# =============================================================================

def RSI(self, price_series="close", length=14):
    """
    Relative Strength Index - Integrato con Backtrader
    """
    try:
        self._update_technical_indicators()
        
        # Prova integrazione Backtrader
        if self._use_backtrader and hasattr(self, 'data'):
            try:
                import backtrader.indicators as btind
                cache_key = f"rsi_{price_series}_{length}"
                
                if cache_key not in self._bt_indicators:
                    price_data = getattr(self.data, price_series, self.data.close)
                    self._bt_indicators[cache_key] = btind.RSI(price_data, period=length)
                
                rsi_val = self._bt_indicators[cache_key][0]
                return float(rsi_val) if not math.isnan(rsi_val) else 50.0
                
            except (ImportError, AttributeError, Exception):
                pass
        
        # Calcolo manuale ottimizzato
        return self._calculate_rsi_manual(length)
        
    except Exception as e:
        print(f"RSI calculation error: {e}")
        return 50.0

def _calculate_rsi_manual(self, length=14):
    """RSI con algoritmo di Wilder ottimizzato"""
    if len(self.tech_tracker._close_buffer) < length + 1:
        return 50.0
    
    try:
        # Usa i buffer pre-calcolati
        gains = list(self.tech_tracker._rsi_gains)[-length:]
        losses = list(self.tech_tracker._rsi_losses)[-length:]
        
        if len(gains) < length:
            return 50.0
        
        # Media iniziale
        if self.tech_tracker._rsi_avg_gain == 0.0:
            self.tech_tracker._rsi_avg_gain = sum(gains) / length
            self.tech_tracker._rsi_avg_loss = sum(losses) / length
        else:
            # Smoothed moving average (Wilder's method)
            alpha = 1.0 / length
            self.tech_tracker._rsi_avg_gain = ((length - 1) * self.tech_tracker._rsi_avg_gain + gains[-1]) / length
            self.tech_tracker._rsi_avg_loss = ((length - 1) * self.tech_tracker._rsi_avg_loss + losses[-1]) / length
        
        if self.tech_tracker._rsi_avg_loss == 0:
            return 100.0
        
        rs = self.tech_tracker._rsi_avg_gain / self.tech_tracker._rsi_avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return max(0.0, min(100.0, rsi))
        
    except Exception:
        return 50.0

def MACD(self, price_series="close", fast_length=12, slow_length=26, signal_length=9):
    """
    MACD Indicator - Moving Average Convergence Divergence
    Returns: (MACD Line, Signal Line, Histogram)
    """
    try:
        self._update_technical_indicators()
        
        # Backtrader integration
        if self._use_backtrader and hasattr(self, 'data'):
            try:
                import backtrader.indicators as btind
                cache_key = f"macd_{price_series}_{fast_length}_{slow_length}_{signal_length}"
                
                if cache_key not in self._bt_indicators:
                    price_data = getattr(self.data, price_series, self.data.close)
                    self._bt_indicators[cache_key] = btind.MACD(
                        price_data, 
                        period_me1=fast_length, 
                        period_me2=slow_length, 
                        period_signal=signal_length
                    )
                
                macd_ind = self._bt_indicators[cache_key]
                macd_line = float(macd_ind.macd[0]) if not math.isnan(macd_ind.macd[0]) else 0.0
                signal_line = float(macd_ind.signal[0]) if not math.isnan(macd_ind.signal[0]) else 0.0
                histogram = macd_line - signal_line
                
                return (macd_line, signal_line, histogram)
                
            except (ImportError, AttributeError, Exception):
                pass
        
        # Manual calculation
        return self._calculate_macd_manual(fast_length, slow_length, signal_length)
        
    except Exception as e:
        print(f"MACD calculation error: {e}")
        return (0.0, 0.0, 0.0)

def _calculate_macd_manual(self, fast_length, slow_length, signal_length):
    """MACD manual calculation with EMA optimization"""
    if len(self.tech_tracker._close_buffer) < slow_length:
        return (0.0, 0.0, 0.0)
    
    try:
        prices = list(self.tech_tracker._close_buffer)
        
        # Calculate EMAs
        ema_fast = self._calculate_ema(prices, fast_length)
        ema_slow = self._calculate_ema(prices, slow_length)
        
        macd_line = ema_fast - ema_slow
        
        # Signal line (EMA of MACD line)
        if not hasattr(self, '_macd_history'):
            self._macd_history = deque(maxlen=100)
        
        self._macd_history.append(macd_line)
        
        if len(self._macd_history) >= signal_length:
            signal_line = self._calculate_ema(list(self._macd_history), signal_length)
        else:
            signal_line = macd_line
        
        histogram = macd_line - signal_line
        
        return (macd_line, signal_line, histogram)
        
    except Exception:
        return (0.0, 0.0, 0.0)

def Stochastic(self, k_length=14, k_slowing=3, d_length=3):
    """
    Stochastic Oscillator - %K and %D
    Returns: (%K, %D)
    """
    try:
        self._update_technical_indicators()
        
        # Backtrader integration
        if self._use_backtrader and hasattr(self, 'data'):
            try:
                import backtrader.indicators as btind
                cache_key = f"stoch_{k_length}_{k_slowing}_{d_length}"
                
                if cache_key not in self._bt_indicators:
                    self._bt_indicators[cache_key] = btind.Stochastic(
                        self.data,
                        period=k_length,
                        period_dfast=k_slowing,
                        period_dslow=d_length
                    )
                
                stoch_ind = self._bt_indicators[cache_key]
                k_val = float(stoch_ind.percK[0]) if not math.isnan(stoch_ind.percK[0]) else 50.0
                d_val = float(stoch_ind.percD[0]) if not math.isnan(stoch_ind.percD[0]) else 50.0
                
                return (k_val, d_val)
                
            except (ImportError, AttributeError, Exception):
                pass
        
        # Manual calculation
        return self._calculate_stochastic_manual(k_length, k_slowing, d_length)
        
    except Exception as e:
        print(f"Stochastic calculation error: {e}")
        return (50.0, 50.0)

def _calculate_stochastic_manual(self, k_length, k_slowing, d_length):
    """Stochastic manual calculation"""
    if len(self.tech_tracker._high_buffer) < k_length:
        return (50.0, 50.0)
    
    try:
        highs = list(self.tech_tracker._high_buffer)[-k_length:]
        lows = list(self.tech_tracker._low_buffer)[-k_length:]
        current_close = self.tech_tracker._close_buffer[-1]
        
        highest_high = max(highs)
        lowest_low = min(lows)
        
        if highest_high == lowest_low:
            raw_k = 50.0
        else:
            raw_k = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100.0
        
        # Smooth %K
        self.tech_tracker._stoch_k_buffer.append(raw_k)
        
        if len(self.tech_tracker._stoch_k_buffer) >= k_slowing:
            k_values = list(self.tech_tracker._stoch_k_buffer)[-k_slowing:]
            smoothed_k = sum(k_values) / len(k_values)
        else:
            smoothed_k = raw_k
        
        # %D is SMA of smoothed %K
        self.tech_tracker._stoch_d_buffer.append(smoothed_k)
        
        if len(self.tech_tracker._stoch_d_buffer) >= d_length:
            d_values = list(self.tech_tracker._stoch_d_buffer)[-d_length:]
            d_val = sum(d_values) / len(d_values)
        else:
            d_val = smoothed_k
        
        return (max(0.0, min(100.0, smoothed_k)), max(0.0, min(100.0, d_val)))
        
    except Exception:
        return (50.0, 50.0)

# =============================================================================
# MOVING AVERAGES
# =============================================================================

def MovingAverage(self, price_series="close", length=20, ma_type="Simple"):
    """
    Universal Moving Average function
    Types: Simple, Exponential, Weighted, Hull
    """
    try:
        self._update_technical_indicators()
        
        # Get price data
        if hasattr(self, 'data'):
            price_data = getattr(self.data, price_series, self.data.close)
            current_price = float(price_data[0])
        else:
            current_price = self.tech_tracker._close_buffer[-1] if self.tech_tracker._close_buffer else 100.0
        
        cache_key = f"ma_{price_series}_{length}_{ma_type.lower()}"
        
        # Backtrader integration per SMA/EMA
        if self._use_backtrader and hasattr(self, 'data') and ma_type.lower() in ['simple', 'exponential']:
            try:
                import backtrader.indicators as btind
                
                if cache_key not in self._bt_indicators:
                    if ma_type.lower() == 'simple':
                        self._bt_indicators[cache_key] = btind.SMA(price_data, period=length)
                    else:  # exponential
                        self._bt_indicators[cache_key] = btind.EMA(price_data, period=length)
                
                ma_val = self._bt_indicators[cache_key][0]
                return float(ma_val) if not math.isnan(ma_val) else current_price
                
            except (ImportError, AttributeError, Exception):
                pass
        
        # Manual calculation
        return self._calculate_moving_average_manual(price_series, length, ma_type)
        
    except Exception as e:
        print(f"Moving Average calculation error: {e}")
        return self.tech_tracker._close_buffer[-1] if self.tech_tracker._close_buffer else 100.0

def _calculate_moving_average_manual(self, price_series, length, ma_type):
    """Manual moving average calculation"""
    if len(self.tech_tracker._close_buffer) < length:
        return self.tech_tracker._close_buffer[-1] if self.tech_tracker._close_buffer else 100.0
    
    prices = list(self.tech_tracker._close_buffer)[-length:]
    
    try:
        if ma_type.lower() == 'simple':
            return sum(prices) / len(prices)
        
        elif ma_type.lower() == 'exponential':
            return self._calculate_ema(list(self.tech_tracker._close_buffer), length)
        
        elif ma_type.lower() == 'weighted':
            weights = range(1, length + 1)
            weighted_sum = sum(price * weight for price, weight in zip(prices, weights))
            weight_sum = sum(weights)
            return weighted_sum / weight_sum
        
        elif ma_type.lower() == 'hull':
            # Hull Moving Average
            half_length = length // 2
            sqrt_length = int(math.sqrt(length))
            
            wma_half = self._calculate_wma(prices, half_length)
            wma_full = self._calculate_wma(prices, length)
            hull_values = [2 * wma_half - wma_full]
            
            return self._calculate_wma(hull_values, min(sqrt_length, len(hull_values)))
        
        else:
            # Default to simple
            return sum(prices) / len(prices)
            
    except Exception:
        return prices[-1] if prices else 100.0

@lru_cache(maxsize=32)
def _calculate_ema(self, prices_tuple, length):
    """Optimized EMA calculation with caching"""
    prices = list(prices_tuple) if isinstance(prices_tuple, tuple) else prices_tuple
    
    if len(prices) < length:
        return prices[-1] if prices else 0.0
    
    alpha = 2.0 / (length + 1)
    ema = prices[0]
    
    for price in prices[1:]:
        ema = alpha * price + (1 - alpha) * ema
    
    return ema

def _calculate_wma(self, prices, length):
    """Weighted Moving Average calculation"""
    if len(prices) < length:
        length = len(prices)
    
    recent_prices = prices[-length:]
    weights = range(1, len(recent_prices) + 1)
    
    weighted_sum = sum(price * weight for price, weight in zip(recent_prices, weights))
    weight_sum = sum(weights)
    
    return weighted_sum / weight_sum if weight_sum > 0 else recent_prices[-1]

def BollingerBands(self, price_series="close", length=20, num_dev=2.0):
    """
    Bollinger Bands - Returns (Upper Band, Middle Band, Lower Band)
    """
    try:
        self._update_technical_indicators()
        
        # Backtrader integration
        if self._use_backtrader and hasattr(self, 'data'):
            try:
                import backtrader.indicators as btind
                cache_key = f"bb_{price_series}_{length}_{num_dev}"
                
                if cache_key not in self._bt_indicators:
                    price_data = getattr(self.data, price_series, self.data.close)
                    self._bt_indicators[cache_key] = btind.BollingerBands(
                        price_data, 
                        period=length, 
                        devfactor=num_dev
                    )
                
                bb_ind = self._bt_indicators[cache_key]
                upper = float(bb_ind.top[0]) if not math.isnan(bb_ind.top[0]) else 0.0
                middle = float(bb_ind.mid[0]) if not math.isnan(bb_ind.mid[0]) else 0.0
                lower = float(bb_ind.bot[0]) if not math.isnan(bb_ind.bot[0]) else 0.0
                
                return (upper, middle, lower)
                
            except (ImportError, AttributeError, Exception):
                pass
        
        # Manual calculation
        return self._calculate_bollinger_bands_manual(length, num_dev)
        
    except Exception as e:
        print(f"Bollinger Bands calculation error: {e}")
        current_price = self.tech_tracker._close_buffer[-1] if self.tech_tracker._close_buffer else 100.0
        return (current_price * 1.02, current_price, current_price * 0.98)

def _calculate_bollinger_bands_manual(self, length, num_dev):
    """Manual Bollinger Bands calculation"""
    if len(self.tech_tracker._close_buffer) < length:
        current_price = self.tech_tracker._close_buffer[-1] if self.tech_tracker._close_buffer else 100.0
        return (current_price * 1.02, current_price, current_price * 0.98)
    
    try:
        prices = list(self.tech_tracker._close_buffer)[-length:]
        
        # Middle band (SMA)
        middle_band = sum(prices) / len(prices)
        
        # Standard deviation
        variance = sum((price - middle_band) ** 2 for price in prices) / len(prices)
        std_dev = math.sqrt(variance)
        
        # Upper and lower bands
        upper_band = middle_band + (num_dev * std_dev)
        lower_band = middle_band - (num_dev * std_dev)
        
        return (upper_band, middle_band, lower_band)
        
    except Exception:
        current_price = self.tech_tracker._close_buffer[-1] if self.tech_tracker._close_buffer else 100.0
        return (current_price * 1.02, current_price, current_price * 0.98)

# =============================================================================
# ADDITIONAL INDICATORS
# =============================================================================

def ADX(self, length=14):
    """Average Directional Index"""
    try:
        self._update_technical_indicators()
        
        if self._use_backtrader and hasattr(self, 'data'):
            try:
                import backtrader.indicators as btind
                cache_key = f"adx_{length}"
                
                if cache_key not in self._bt_indicators:
                    self._bt_indicators[cache_key] = btind.ADX(self.data, period=length)
                
                adx_val = self._bt_indicators[cache_key][0]
                return float(adx_val) if not math.isnan(adx_val) else 25.0
                
            except (ImportError, AttributeError, Exception):
                pass
        
        # Simplified ADX calculation
        return self._calculate_adx_simplified(length)
        
    except Exception as e:
        print(f"ADX calculation error: {e}")
        return 25.0

def _calculate_adx_simplified(self, length):
    """Simplified ADX calculation"""
    if len(self.tech_tracker._high_buffer) < length + 1:
        return 25.0
    
    try:
        # Calculate True Range and Directional Movement
        highs = list(self.tech_tracker._high_buffer)[-length-1:]
        lows = list(self.tech_tracker._low_buffer)[-length-1:]
        closes = list(self.tech_tracker._close_buffer)[-length-1:]
        
        dm_plus = []
        dm_minus = []
        tr_values = []
        
        for i in range(1, len(highs)):
            high_diff = highs[i] - highs[i-1]
            low_diff = lows[i-1] - lows[i]
            
            dm_p = high_diff if high_diff > low_diff and high_diff > 0 else 0
            dm_m = low_diff if low_diff > high_diff and low_diff > 0 else 0
            
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            
            dm_plus.append(dm_p)
            dm_minus.append(dm_m)
            tr_values.append(tr)
        
        if not tr_values:
            return 25.0
        
        # Smooth the values
        avg_dm_plus = sum(dm_plus) / len(dm_plus)
        avg_dm_minus = sum(dm_minus) / len(dm_minus)
        avg_tr = sum(tr_values) / len(tr_values)
        
        if avg_tr == 0:
            return 25.0
        
        di_plus = (avg_dm_plus / avg_tr) * 100
        di_minus = (avg_dm_minus / avg_tr) * 100
        
        if di_plus + di_minus == 0:
            return 25.0
        
        dx = abs(di_plus - di_minus) / (di_plus + di_minus) * 100
        
        # ADX is smoothed DX (simplified)
        return max(0.0, min(100.0, dx))
        
    except Exception:
        return 25.0

def CCI(self, length=20):
    """Commodity Channel Index"""
    try:
        self._update_technical_indicators()
        
        if len(self.tech_tracker._high_buffer) < length:
            return 0.0
        
        # Calculate Typical Price
        typical_prices = []
        for i in range(-length, 0):
            high = self.tech_tracker._high_buffer[i]
            low = self.tech_tracker._low_buffer[i]
            close = self.tech_tracker._close_buffer[i]
            typical_price = (high + low + close) / 3.0
            typical_prices.append(typical_price)
        
        # Simple Moving Average of Typical Price
        sma_tp = sum(typical_prices) / len(typical_prices)
        
        # Mean Deviation
        mean_deviation = sum(abs(tp - sma_tp) for tp in typical_prices) / len(typical_prices)
        
        if mean_deviation == 0:
            return 0.0
        
        # CCI calculation
        current_tp = typical_prices[-1]
        cci = (current_tp - sma_tp) / (0.015 * mean_deviation)
        
        return max(-500.0, min(500.0, cci))
        
    except Exception as e:
        print(f"CCI calculation error: {e}")
        return 0.0

def Williams_R(self, length=14):
    """Williams %R"""
    try:
        self._update_technical_indicators()
        
        if len(self.tech_tracker._high_buffer) < length:
            return -50.0
        
        highs = list(self.tech_tracker._high_buffer)[-length:]
        lows = list(self.tech_tracker._low_buffer)[-length:]
        current_close = self.tech_tracker._close_buffer[-1]
        
        highest_high = max(highs)
        lowest_low = min(lows)
        
        if highest_high == lowest_low:
            return -50.0
        
        williams_r = ((highest_high - current_close) / (highest_high - lowest_low)) * -100.0
        
        return max(-100.0, min(0.0, williams_r))
        
    except Exception as e:
        print(f"Williams %R calculation error: {e}")
        return -50.0

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def run_technical_indicators_test():
    """Test suite per gli indicatori tecnici"""
    print("=== TECHNICAL INDICATORS TEST ===")
    
    class MockStrategy:
        def __init__(self):
            class MockData:
                def __init__(self):
                    # Generate test data
                    self.close = [100 + i * 0.5 + np.random.normal(0, 1) for i in range(50)]
                    self.high = [c + np.random.uniform(0, 2) for c in self.close]
                    self.low = [c - np.random.uniform(0, 2) for c in self.close]
                    self.open = [c + np.random.uniform(-1, 1) for c in self.close]
                    self.volume = [1000 + np.random.randint(-200, 200) for _ in range(50)]
                    self.current_idx = len(self.close) - 1
                
                def __getitem__(self, idx):
                    actual_idx = max(0, min(len(self.close) - 1, self.current_idx + idx))
                    return self.close[actual_idx]
            
            self.data = MockData()
            self.__init_technical_indicators__()
    
    # Test all indicators
    strategy = MockStrategy()
    
    # Add all technical indicator methods
    import types
    for name, obj in globals().items():
        if callable(obj) and (name.startswith('__init_technical') or 
                             name.startswith('_update_technical') or
                             name.startswith('_calculate_') or
                             name in ['RSI', 'MACD', 'Stochastic', 'MovingAverage', 
                                     'BollingerBands', 'ADX', 'CCI', 'Williams_R']):
            setattr(strategy, name, types.MethodType(obj, strategy))
    
    # Run tests
    print(f"RSI(14): {strategy.RSI():.2f}")
    
    macd_data = strategy.MACD()
    print(f"MACD: Line={macd_data[0]:.4f}, Signal={macd_data[1]:.4f}, Hist={macd_data[2]:.4f}")
    
    stoch_data = strategy.Stochastic()
    print(f"Stochastic: %K={stoch_data[0]:.2f}, %D={stoch_data[1]:.2f}")
    
    sma = strategy.MovingAverage("close", 20, "Simple")
    ema = strategy.MovingAverage("close", 20, "Exponential")
    print(f"SMA(20): {sma:.2f}, EMA(20): {ema:.2f}")
    
    bb_data = strategy.BollingerBands()
    print(f"Bollinger Bands: Upper={bb_data[0]:.2f}, Middle={bb_data[1]:.2f}, Lower={bb_data[2]:.2f}")
    
    print(f"ADX(14): {strategy.ADX():.2f}")
    print(f"CCI(20): {strategy.CCI():.2f}")
    print(f"Williams %R(14): {strategy.Williams_R():.2f}")
    
    print("✓ Technical Indicators test completed")

if __name__ == "__main__":
    run_technical_indicators_test()
