# =============================================================================
# PRICE ACTION FUNCTIONS - EASYLANGUAGE INTEGRATION
# High-Performance Price Analysis with Backtrader Compatibility
# =============================================================================

import numpy as np
import math
from collections import deque
from functools import lru_cache
import warnings

class PriceActionTracker:
    """
    Tracker ottimizzato per funzioni price action con caching avanzato
    """
    def __init__(self, max_history=1000):
        self.max_history = max_history
        
        # Circular buffers ottimizzati
        self._open_buffer = deque(maxlen=max_history)
        self._high_buffer = deque(maxlen=max_history)
        self._low_buffer = deque(maxlen=max_history)
        self._close_buffer = deque(maxlen=max_history)
        self._volume_buffer = deque(maxlen=max_history)
        
        # Pre-computed values cache
        self._range_cache = deque(maxlen=max_history)
        self._true_range_cache = deque(maxlen=max_history)
        self._typical_price_cache = deque(maxlen=max_history)
        self._weighted_close_cache = deque(maxlen=max_history)
        self._median_price_cache = deque(maxlen=max_history)
        self._avg_price_cache = deque(maxlen=max_history)
        
        # Highest/Lowest tracking ottimizzato
        self._highest_cache = {}
        self._lowest_cache = {}
        self._cache_timestamp = 0
        
        # Performance metrics
        self._calculations_count = 0
        self._cache_hits = 0
        
    def add_bar_data(self, open_price, high, low, close, volume=1000):
        """Aggiunge dati barra e pre-calcola valori comuni"""
        self._open_buffer.append(open_price)
        self._high_buffer.append(high)
        self._low_buffer.append(low)
        self._close_buffer.append(close)
        self._volume_buffer.append(volume)
        
        # Pre-calcola valori comuni
        self._calculate_bar_values(open_price, high, low, close, volume)
        
        # Invalida cache per Highest/Lowest
        self._cache_timestamp += 1
        if len(self._highest_cache) > 100:  # Limit cache size
            self._highest_cache.clear()
        if len(self._lowest_cache) > 100:
            self._lowest_cache.clear()
    
    def _calculate_bar_values(self, open_price, high, low, close, volume):
        """Pre-calcola valori price action per la barra corrente"""
        # Range
        range_val = high - low
        self._range_cache.append(range_val)
        
        # True Range
        if len(self._close_buffer) > 1:
            prev_close = self._close_buffer[-2]
            true_range = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
        else:
            true_range = range_val
        self._true_range_cache.append(true_range)
        
        # Typical Price (H+L+C)/3
        typical_price = (high + low + close) / 3.0
        self._typical_price_cache.append(typical_price)
        
        # Weighted Close (H+L+2*C)/4
        weighted_close = (high + low + 2 * close) / 4.0
        self._weighted_close_cache.append(weighted_close)
        
        # Median Price (H+L)/2
        median_price = (high + low) / 2.0
        self._median_price_cache.append(median_price)
        
        # Average Price (O+H+L+C)/4
        avg_price = (open_price + high + low + close) / 4.0
        self._avg_price_cache.append(avg_price)

def __init_price_action__(self):
    """
    Inizializzazione sistema price action ottimizzato
    """
    self.price_tracker = PriceActionTracker()
    
    # Performance optimization flags
    self._use_numpy_acceleration = True
    self._last_bar_processed = -1
    
    # Cache per calcoli pesanti
    self._calculation_cache = {}
    
    print("✓ Price Action Functions initialized")

def _update_price_action_data(self):
    """
    Aggiorna i dati price action con la barra corrente
    """
    try:
        if hasattr(self, 'data') and len(self.data.close) > 0:
            current_bar = len(self.data.close) - 1
            
            # Skip se già processato
            if current_bar == self._last_bar_processed:
                return {
            'calculations_count': self.price_tracker._calculations_count,
            'cache_hits': self.price_tracker._cache_hits,
            'cache_hit_rate_percent': cache_hit_rate,
            'buffer_sizes': {
                'open': len(self.price_tracker._open_buffer),
                'high': len(self.price_tracker._high_buffer),
                'low': len(self.price_tracker._low_buffer),
                'close': len(self.price_tracker._close_buffer),
                'volume': len(self.price_tracker._volume_buffer)
            },
            'cache_sizes': {
                'highest_cache': len(self.price_tracker._highest_cache),
                'lowest_cache': len(self.price_tracker._lowest_cache)
            }
        }
        
    except Exception as e:
        print(f"Error getting price action stats: {e}")
        return {}

def reset_price_action_cache(self):
    """Reset all caches for memory optimization"""
    try:
        self.price_tracker._highest_cache.clear()
        self.price_tracker._lowest_cache.clear()
        self._calculation_cache.clear()
        print("✓ Price action cache reset completed")
        
    except Exception as e:
        print(f"Error resetting cache: {e}")

# =============================================================================
# TEST SUITE
# =============================================================================

def run_price_action_test():
    """Comprehensive test suite per le funzioni price action"""
    print("=== PRICE ACTION FUNCTIONS TEST ===")
    
    class MockStrategy:
        def __init__(self):
            class MockData:
                def __init__(self):
                    # Generate realistic OHLCV data
                    import random
                    random.seed(42)  # Reproducible results
                    
                    base_price = 100.0
                    self.close = []
                    self.high = []
                    self.low = []
                    self.open = []
                    self.volume = []
                    
                    for i in range(100):
                        # Random walk with some trend
                        change = random.uniform(-2, 2) + 0.05  # Slight upward bias
                        close_price = max(10.0, base_price + change)
                        
                        # Generate OHLC from close
                        range_size = random.uniform(0.5, 3.0)
                        high_price = close_price + random.uniform(0, range_size)
                        low_price = close_price - random.uniform(0, range_size)
                        open_price = low_price + random.uniform(0, high_price - low_price)
                        
                        self.close.append(close_price)
                        self.high.append(high_price)
                        self.low.append(low_price)
                        self.open.append(open_price)
                        self.volume.append(random.randint(800, 1500))
                        
                        base_price = close_price
                    
                    self.current_idx = 0
                
                def __getitem__(self, idx):
                    actual_idx = max(0, min(len(self.close) - 1, self.current_idx + idx))
                    return self.close[actual_idx]
            
            self.data = MockData()
            self.__init_price_action__()
    
    # Create strategy and add methods
    strategy = MockStrategy()
    
    import types
    for name, obj in globals().items():
        if callable(obj) and (name.startswith('__init_price') or 
                             name.startswith('_update_price') or
                             name.startswith('_get_numpy') or
                             name in ['Open', 'High', 'Low', 'Close', 'Volume',
                                     'Range', 'TrueRange', 'TypicalPrice', 'WeightedClose',
                                     'MedianPrice', 'AvgPrice', 'Highest', 'Lowest',
                                     'HighestHigh', 'LowestLow', 'PriceChange', 'PercentChange',
                                     'IsNewHigh', 'IsNewLow', 'IsInsideBar', 'IsOutsideBar',
                                     'IsUpBar', 'IsDownBar', 'IsDoji', 'IsHammer', 'IsShootingStar',
                                     'get_price_action_stats', 'reset_price_action_cache']):
            setattr(strategy, name, types.MethodType(obj, strategy))
    
    # Run comprehensive tests
    print("Testing Basic OHLCV Functions:")
    print(f"  Current OHLCV: O={strategy.Open():.2f}, H={strategy.High():.2f}, L={strategy.Low():.2f}, C={strategy.Close():.2f}, V={strategy.Volume():.0f}")
    print(f"  Previous OHLCV: O={strategy.Open(1):.2f}, H={strategy.High(1):.2f}, L={strategy.Low(1):.2f}, C={strategy.Close(1):.2f}")
    
    print("\nTesting Derived Price Functions:")
    print(f"  Range: {strategy.Range():.2f}")
    print(f"  True Range: {strategy.TrueRange():.2f}")
    print(f"  Typical Price: {strategy.TypicalPrice():.2f}")
    print(f"  Weighted Close: {strategy.WeightedClose():.2f}")
    print(f"  Median Price: {strategy.MedianPrice():.2f}")
    print(f"  Average Price: {strategy.AvgPrice():.2f}")
    
    print("\nTesting Highest/Lowest Functions:")
    print(f"  Highest High(10): {strategy.HighestHigh(10):.2f}")
    print(f"  Lowest Low(10): {strategy.LowestLow(10):.2f}")
    print(f"  Highest Close(20): {strategy.Highest('close', 20):.2f}")
    print(f"  Lowest Close(20): {strategy.Lowest('close', 20):.2f}")
    
    print("\nTesting Change Functions:")
    print(f"  Price Change(1): {strategy.PriceChange(1):.2f}")
    print(f"  Percent Change(1): {strategy.PercentChange(1):.2f}%")
    print(f"  Price Change(5): {strategy.PriceChange(5):.2f}")
    print(f"  Percent Change(5): {strategy.PercentChange(5):.2f}%")
    
    print("\nTesting Pattern Recognition:")
    print(f"  Is New High(20): {strategy.IsNewHigh(20)}")
    print(f"  Is New Low(20): {strategy.IsNewLow(20)}")
    print(f"  Is Inside Bar: {strategy.IsInsideBar()}")
    print(f"  Is Outside Bar: {strategy.IsOutsideBar()}")
    print(f"  Is Up Bar: {strategy.IsUpBar()}")
    print(f"  Is Down Bar: {strategy.IsDownBar()}")
    print(f"  Is Doji: {strategy.IsDoji()}")
    print(f"  Is Hammer: {strategy.IsHammer()}")
    print(f"  Is Shooting Star: {strategy.IsShootingStar()}")
    
    print("\nTesting Performance on Multiple Bars:")
    import time
    start_time = time.time()
    
    for i in range(50, 99):  # Test on different bars
        strategy.data.current_idx = i
        highest = strategy.HighestHigh(20)
        lowest = strategy.LowestLow(20)
        tr = strategy.TrueRange()
        tp = strategy.TypicalPrice()
    
    end_time = time.time()
    print(f"  Processed 49 bars in {(end_time - start_time)*1000:.2f} ms")
    
    # Test cache performance
    stats = strategy.get_price_action_stats()
    print(f"\nPerformance Statistics:")
    print(f"  Calculations: {stats['calculations_count']}")
    print(f"  Cache Hits: {stats['cache_hits']}")
    print(f"  Cache Hit Rate: {stats['cache_hit_rate_percent']:.1f}%")
    print(f"  Buffer Sizes: {stats['buffer_sizes']}")
    
    # Test cache reset
    strategy.reset_price_action_cache()
    
    print("✓ Price Action Functions test completed")

if __name__ == "__main__":
    run_price_action_test()
            
    open_price = float(self.data.open[0])
    high = float(self.data.high[0])
    low = float(self.data.low[0])
    close = float(self.data.close[0])
    volume = float(self.data.volume[0]) if hasattr(self.data, 'volume') else 1000.0
    
    self.price_tracker.add_bar_data(open_price, high, low, close, volume)
    self._last_bar_processed = current_bar
    
except Exception as e:
    print(f"Error updating price action data: {e}")

# =============================================================================
# BASIC PRICE DATA FUNCTIONS
# =============================================================================

def Open(self, bars_back=0):
    """Returns Open price of specified bar (0=current bar)"""
    try:
        if hasattr(self, 'data') and hasattr(self.data, 'open'):
            if bars_back == 0:
                return float(self.data.open[0])
            else:
                # Historical reference
                total_bars = len(self.data.open)
                if bars_back < total_bars:
                    return float(self.data.open[bars_back])
        
        # Fallback to tracker
        if len(self.price_tracker._open_buffer) > bars_back:
            return self.price_tracker._open_buffer[-(bars_back + 1)]
        
        return 0.0
        
    except Exception as e:
        print(f"Open calculation error: {e}")
        return 0.0

def High(self, bars_back=0):
    """Returns High price of specified bar (0=current bar)"""
    try:
        if hasattr(self, 'data') and hasattr(self.data, 'high'):
            if bars_back == 0:
                return float(self.data.high[0])
            else:
                total_bars = len(self.data.high)
                if bars_back < total_bars:
                    return float(self.data.high[bars_back])
        
        # Fallback to tracker
        if len(self.price_tracker._high_buffer) > bars_back:
            return self.price_tracker._high_buffer[-(bars_back + 1)]
        
        return 0.0
        
    except Exception as e:
        print(f"High calculation error: {e}")
        return 0.0

def Low(self, bars_back=0):
    """Returns Low price of specified bar (0=current bar)"""
    try:
        if hasattr(self, 'data') and hasattr(self.data, 'low'):
            if bars_back == 0:
                return float(self.data.low[0])
            else:
                total_bars = len(self.data.low)
                if bars_back < total_bars:
                    return float(self.data.low[bars_back])
        
        # Fallback to tracker
        if len(self.price_tracker._low_buffer) > bars_back:
            return self.price_tracker._low_buffer[-(bars_back + 1)]
        
        return 0.0
        
    except Exception as e:
        print(f"Low calculation error: {e}")
        return 0.0

def Close(self, bars_back=0):
    """Returns Close price of specified bar (0=current bar)"""
    try:
        if hasattr(self, 'data') and hasattr(self.data, 'close'):
            if bars_back == 0:
                return float(self.data.close[0])
            else:
                total_bars = len(self.data.close)
                if bars_back < total_bars:
                    return float(self.data.close[bars_back])
        
        # Fallback to tracker
        if len(self.price_tracker._close_buffer) > bars_back:
            return self.price_tracker._close_buffer[-(bars_back + 1)]
        
        return 0.0
        
    except Exception as e:
        print(f"Close calculation error: {e}")
        return 0.0

def Volume(self, bars_back=0):
    """Returns Volume of specified bar (0=current bar)"""
    try:
        if hasattr(self, 'data') and hasattr(self.data, 'volume'):
            if bars_back == 0:
                return float(self.data.volume[0])
            else:
                total_bars = len(self.data.volume)
                if bars_back < total_bars:
                    return float(self.data.volume[bars_back])
        
        # Fallback to tracker
        if len(self.price_tracker._volume_buffer) > bars_back:
            return self.price_tracker._volume_buffer[-(bars_back + 1)]
        
        return 1000.0  # Default volume
        
    except Exception as e:
        print(f"Volume calculation error: {e}")
        return 1000.0

# =============================================================================
# DERIVED PRICE FUNCTIONS
# =============================================================================

def Range(self, bars_back=0):
    """Returns High - Low for specified bar"""
    try:
        self._update_price_action_data()
        
        if bars_back == 0 and len(self.price_tracker._range_cache) > 0:
            return self.price_tracker._range_cache[-1]
        
        high_val = self.High(bars_back)
        low_val = self.Low(bars_back)
        return high_val - low_val
        
    except Exception as e:
        print(f"Range calculation error: {e}")
        return 0.0

def TrueRange(self, bars_back=0):
    """
    Returns True Range: max(H-L, |H-PC|, |L-PC|) where PC = Previous Close
    """
    try:
        self._update_price_action_data()
        
        if bars_back == 0 and len(self.price_tracker._true_range_cache) > 0:
            return self.price_tracker._true_range_cache[-1]
        
        high_val = self.High(bars_back)
        low_val = self.Low(bars_back)
        prev_close = self.Close(bars_back + 1)
        
        if prev_close == 0.0:
            return high_val - low_val
        
        return max(
            high_val - low_val,
            abs(high_val - prev_close),
            abs(low_val - prev_close)
        )
        
    except Exception as e:
        print(f"TrueRange calculation error: {e}")
        return 0.0

def TypicalPrice(self, bars_back=0):
    """Returns (High + Low + Close) / 3"""
    try:
        self._update_price_action_data()
        
        if bars_back == 0 and len(self.price_tracker._typical_price_cache) > 0:
            return self.price_tracker._typical_price_cache[-1]
        
        high_val = self.High(bars_back)
        low_val = self.Low(bars_back)
        close_val = self.Close(bars_back)
        
        return (high_val + low_val + close_val) / 3.0
        
    except Exception as e:
        print(f"TypicalPrice calculation error: {e}")
        return 0.0

def WeightedClose(self, bars_back=0):
    """Returns (High + Low + 2*Close) / 4"""
    try:
        self._update_price_action_data()
        
        if bars_back == 0 and len(self.price_tracker._weighted_close_cache) > 0:
            return self.price_tracker._weighted_close_cache[-1]
        
        high_val = self.High(bars_back)
        low_val = self.Low(bars_back)
        close_val = self.Close(bars_back)
        
        return (high_val + low_val + 2 * close_val) / 4.0
        
    except Exception as e:
        print(f"WeightedClose calculation error: {e}")
        return 0.0

def MedianPrice(self, bars_back=0):
    """Returns (High + Low) / 2"""
    try:
        self._update_price_action_data()
        
        if bars_back == 0 and len(self.price_tracker._median_price_cache) > 0:
            return self.price_tracker._median_price_cache[-1]
        
        high_val = self.High(bars_back)
        low_val = self.Low(bars_back)
        
        return (high_val + low_val) / 2.0
        
    except Exception as e:
        print(f"MedianPrice calculation error: {e}")
        return 0.0

def AvgPrice(self, bars_back=0):
    """Returns (Open + High + Low + Close) / 4"""
    try:
        self._update_price_action_data()
        
        if bars_back == 0 and len(self.price_tracker._avg_price_cache) > 0:
            return self.price_tracker._avg_price_cache[-1]
        
        open_val = self.Open(bars_back)
        high_val = self.High(bars_back)
        low_val = self.Low(bars_back)
        close_val = self.Close(bars_back)
        
        return (open_val + high_val + low_val + close_val) / 4.0
        
    except Exception as e:
        print(f"AvgPrice calculation error: {e}")
        return 0.0

# =============================================================================
# DISPLACEMENT FUNCTIONS (EasyLanguage Format)
# =============================================================================

def OpenD(self, displacement=0):
    """Returns Open with displacement (EasyLanguage compatible)"""
    return self.Open(abs(displacement))

def HighD(self, displacement=0):
    """Returns High with displacement (EasyLanguage compatible)"""
    return self.High(abs(displacement))

def LowD(self, displacement=0):
    """Returns Low with displacement (EasyLanguage compatible)"""
    return self.Low(abs(displacement))

def CloseD(self, displacement=0):
    """Returns Close with displacement (EasyLanguage compatible)"""
    return self.Close(abs(displacement))

# =============================================================================
# HIGHEST/LOWEST FUNCTIONS - OPTIMIZED
# =============================================================================

def Highest(self, series_name="high", length=10):
    """
    Returns highest value over specified length
    Optimized with intelligent caching
    """
    try:
        self._update_price_action_data()
        
        # Cache key
        cache_key = f"highest_{series_name}_{length}_{self.price_tracker._cache_timestamp}"
        
        if cache_key in self.price_tracker._highest_cache:
            self.price_tracker._cache_hits += 1
            return self.price_tracker._highest_cache[cache_key]
        
        # Get data series
        if series_name.lower() == "high":
            data_buffer = self.price_tracker._high_buffer
        elif series_name.lower() == "low":
            data_buffer = self.price_tracker._low_buffer
        elif series_name.lower() == "close":
            data_buffer = self.price_tracker._close_buffer
        elif series_name.lower() == "open":
            data_buffer = self.price_tracker._open_buffer
        elif series_name.lower() == "volume":
            data_buffer = self.price_tracker._volume_buffer
        else:
            # Default to high
            data_buffer = self.price_tracker._high_buffer
        
        if len(data_buffer) < length:
            length = len(data_buffer)
        
        if length == 0:
            return 0.0
        
        # Calculate highest - optimized for performance
        recent_values = list(data_buffer)[-length:]
        highest_val = max(recent_values) if recent_values else 0.0
        
        # Cache result
        self.price_tracker._highest_cache[cache_key] = highest_val
        self.price_tracker._calculations_count += 1
        
        return highest_val
        
    except Exception as e:
        print(f"Highest calculation error: {e}")
        return 0.0

def Lowest(self, series_name="low", length=10):
    """
    Returns lowest value over specified length
    Optimized with intelligent caching
    """
    try:
        self._update_price_action_data()
        
        # Cache key
        cache_key = f"lowest_{series_name}_{length}_{self.price_tracker._cache_timestamp}"
        
        if cache_key in self.price_tracker._lowest_cache:
            self.price_tracker._cache_hits += 1
            return self.price_tracker._lowest_cache[cache_key]
        
        # Get data series
        if series_name.lower() == "high":
            data_buffer = self.price_tracker._high_buffer
        elif series_name.lower() == "low":
            data_buffer = self.price_tracker._low_buffer
        elif series_name.lower() == "close":
            data_buffer = self.price_tracker._close_buffer
        elif series_name.lower() == "open":
            data_buffer = self.price_tracker._open_buffer
        elif series_name.lower() == "volume":
            data_buffer = self.price_tracker._volume_buffer
        else:
            # Default to low
            data_buffer = self.price_tracker._low_buffer
        
        if len(data_buffer) < length:
            length = len(data_buffer)
        
        if length == 0:
            return 0.0
        
        # Calculate lowest - optimized for performance
        recent_values = list(data_buffer)[-length:]
        lowest_val = min(recent_values) if recent_values else 0.0
        
        # Cache result
        self.price_tracker._lowest_cache[cache_key] = lowest_val
        self.price_tracker._calculations_count += 1
        
        return lowest_val
        
    except Exception as e:
        print(f"Lowest calculation error: {e}")
        return 0.0

def HighestHigh(self, length=10):
    """Convenience function for Highest High"""
    return self.Highest("high", length)

def LowestLow(self, length=10):
    """Convenience function for Lowest Low"""
    return self.Lowest("low", length)

def HighestClose(self, length=10):
    """Convenience function for Highest Close"""
    return self.Highest("close", length)

def LowestClose(self, length=10):
    """Convenience function for Lowest Close"""
    return self.Lowest("close", length)

def HighestVolume(self, length=10):
    """Convenience function for Highest Volume"""
    return self.Highest("volume", length)

def LowestVolume(self, length=10):
    """Convenience function for Lowest Volume"""
    return self.Lowest("volume", length)

# =============================================================================
# ADVANCED PRICE ACTION FUNCTIONS
# =============================================================================

def PriceChange(self, bars_back=1):
    """Returns price change from bars_back periods ago"""
    try:
        current_price = self.Close(0)
        previous_price = self.Close(bars_back)
        return current_price - previous_price
        
    except Exception as e:
        print(f"PriceChange calculation error: {e}")
        return 0.0

def PercentChange(self, bars_back=1):
    """Returns percentage change from bars_back periods ago"""
    try:
        current_price = self.Close(0)
        previous_price = self.Close(bars_back)
        
        if previous_price == 0:
            return 0.0
        
        return ((current_price - previous_price) / previous_price) * 100.0
        
    except Exception as e:
        print(f"PercentChange calculation error: {e}")
        return 0.0

def IsNewHigh(self, length=10):
    """Returns True if current high is highest in specified length"""
    try:
        current_high = self.High(0)
        highest_high = self.HighestHigh(length)
        return current_high >= highest_high
        
    except Exception as e:
        return False

def IsNewLow(self, length=10):
    """Returns True if current low is lowest in specified length"""
    try:
        current_low = self.Low(0)
        lowest_low = self.LowestLow(length)
        return current_low <= lowest_low
        
    except Exception as e:
        return False

def BarsSinceNewHigh(self, length=20):
    """Returns number of bars since new high was made"""
    try:
        current_high = self.High(0)
        bars_count = 0
        
        for i in range(1, min(length, len(self.price_tracker._high_buffer))):
            historical_high = self.High(i)
            if historical_high >= current_high:
                return bars_count
            bars_count += 1
        
        return bars_count
        
    except Exception as e:
        return 0

def BarsSinceNewLow(self, length=20):
    """Returns number of bars since new low was made"""
    try:
        current_low = self.Low(0)
        bars_count = 0
        
        for i in range(1, min(length, len(self.price_tracker._low_buffer))):
            historical_low = self.Low(i)
            if historical_low <= current_low:
                return bars_count
            bars_count += 1
        
        return bars_count
        
    except Exception as e:
        return 0

# =============================================================================
# PATTERN RECOGNITION HELPERS
# =============================================================================

def IsInsideBar(self, bars_back=0):
    """Returns True if bar is inside previous bar (H<PH and L>PL)"""
    try:
        current_high = self.High(bars_back)
        current_low = self.Low(bars_back)
        prev_high = self.High(bars_back + 1)
        prev_low = self.Low(bars_back + 1)
        
        return current_high < prev_high and current_low > prev_low
        
    except Exception as e:
        return False

def IsOutsideBar(self, bars_back=0):
    """Returns True if bar is outside previous bar (H>PH and L<PL)"""
    try:
        current_high = self.High(bars_back)
        current_low = self.Low(bars_back)
        prev_high = self.High(bars_back + 1)
        prev_low = self.Low(bars_back + 1)
        
        return current_high > prev_high and current_low < prev_low
        
    except Exception as e:
        return False

def IsUpBar(self, bars_back=0):
    """Returns True if close > open"""
    try:
        close_val = self.Close(bars_back)
        open_val = self.Open(bars_back)
        return close_val > open_val
        
    except Exception as e:
        return False

def IsDownBar(self, bars_back=0):
    """Returns True if close < open"""
    try:
        close_val = self.Close(bars_back)
        open_val = self.Open(bars_back)
        return close_val < open_val
        
    except Exception as e:
        return False

def IsDoji(self, bars_back=0, threshold=0.1):
    """Returns True if bar is doji (close ≈ open within threshold %)"""
    try:
        close_val = self.Close(bars_back)
        open_val = self.Open(bars_back)
        range_val = self.Range(bars_back)
        
        if range_val == 0:
            return True
        
        body_size = abs(close_val - open_val)
        body_percent = (body_size / range_val) * 100.0
        
        return body_percent <= threshold
        
    except Exception as e:
        return False

def IsHammer(self, bars_back=0):
    """Basic hammer pattern recognition"""
    try:
        open_val = self.Open(bars_back)
        high_val = self.High(bars_back)
        low_val = self.Low(bars_back)
        close_val = self.Close(bars_back)
        
        body_size = abs(close_val - open_val)
        upper_shadow = high_val - max(open_val, close_val)
        lower_shadow = min(open_val, close_val) - low_val
        
        # Hammer criteria: small body, long lower shadow, short upper shadow
        return (lower_shadow > 2 * body_size and 
                upper_shadow < body_size and 
                body_size > 0)
        
    except Exception as e:
        return False

def IsShootingStar(self, bars_back=0):
    """Basic shooting star pattern recognition"""
    try:
        open_val = self.Open(bars_back)
        high_val = self.High(bars_back)
        low_val = self.Low(bars_back)
        close_val = self.Close(bars_back)
        
        body_size = abs(close_val - open_val)
        upper_shadow = high_val - max(open_val, close_val)
        lower_shadow = min(open_val, close_val) - low_val
        
        # Shooting star criteria: small body, long upper shadow, short lower shadow
        return (upper_shadow > 2 * body_size and 
                lower_shadow < body_size and 
                body_size > 0)
        
    except Exception as e:
        return False

# =============================================================================
# NUMPY ACCELERATION (OPTIONAL)
# =============================================================================

def _get_numpy_series(self, series_name, length):
    """Ottiene series come numpy array per calcoli accelerati"""
    try:
        if not self._use_numpy_acceleration:
            return None
        
        import numpy as np
        
        if series_name.lower() == "high":
            data_list = list(self.price_tracker._high_buffer)[-length:]
        elif series_name.lower() == "low":
            data_list = list(self.price_tracker._low_buffer)[-length:]
        elif series_name.lower() == "close":
            data_list = list(self.price_tracker._close_buffer)[-length:]
        elif series_name.lower() == "open":
            data_list = list(self.price_tracker._open_buffer)[-length:]
        elif series_name.lower() == "volume":
            data_list = list(self.price_tracker._volume_buffer)[-length:]
        else:
            return None
        
        return np.array(data_list, dtype=np.float64)
        
    except (ImportError, Exception):
        return None

def HighestNumpy(self, series_name="high", length=10):
    """Numpy-accelerated highest calculation"""
    try:
        np_array = self._get_numpy_series(series_name, length)
        if np_array is not None:
            return float(np.max(np_array))
        else:
            return self.Highest(series_name, length)
            
    except Exception:
        return self.Highest(series_name, length)

def LowestNumpy(self, series_name="low", length=10):
    """Numpy-accelerated lowest calculation"""
    try:
        np_array = self._get_numpy_series(series_name, length)
        if np_array is not None:
            return float(np.min(np_array))
        else:
            return self.Lowest(series_name, length)
            
    except Exception:
        return self.Lowest(series_name, length)

# =============================================================================
# PERFORMANCE AND UTILITY FUNCTIONS
# =============================================================================

def get_price_action_stats(self):
    """Restituisce statistiche performance del sistema price action"""
    try:
        cache_hit_rate = 0.0
        if self.price_tracker._calculations_count > 0:
            cache_hit_rate = (self.price_tracker._cache_hits / self.price_tracker._calculations_count) * 100.0
        
        return