# ========================================================================
# ADVANCED MATH & STATISTICAL FUNCTIONS - EasyLanguage for Backtrader
# Funzioni matematiche avanzate e analisi statistiche
# ========================================================================

import math
import numpy as np
from collections import deque
from typing import Optional, List, Tuple, Union, Dict, Any
from dataclasses import dataclass
import statistics
from scipy import stats
import warnings

# Suppress scipy warnings per performance
warnings.filterwarnings('ignore', category=RuntimeWarning)

@dataclass
class RegressionResult:
    """Risultato di una regressione lineare"""
    slope: float
    intercept: float
    r_squared: float
    correlation: float
    std_error: float
    p_value: float
    
class StatisticalBuffer:
    """Buffer ottimizzato per calcoli statistici su finestre mobili"""
    
    def __init__(self, max_size: int = 1000):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
        
    def append(self, value: float):
        """Aggiunge un valore al buffer"""
        if not math.isnan(value) and math.isfinite(value):
            self.buffer.append(float(value))
    
    def get_window(self, length: int) -> List[float]:
        """Restituisce una finestra degli ultimi N valori"""
        if length <= 0 or len(self.buffer) < length:
            return list(self.buffer)
        return list(self.buffer)[-length:]
    
    def size(self) -> int:
        """Restituisce la dimensione corrente del buffer"""
        return len(self.buffer)

class MathStatTracker:
    """Helper class per tracking di calcoli matematici e statistici"""
    
    def __init__(self, max_buffer_size: int = 1000):
        self.price_buffer = StatisticalBuffer(max_buffer_size)
        self.volume_buffer = StatisticalBuffer(max_buffer_size)
        self.return_buffer = StatisticalBuffer(max_buffer_size)
        self.custom_buffers = {}
        
        # Cache per calcoli pesanti
        self._correlation_cache = {}
        self._regression_cache = {}
        self._cache_max_age = 10  # bars
        self._last_cache_clear = 0
        
    def add_price_data(self, price: float, volume: float = 0.0):
        """Aggiunge dati di prezzo e volume"""
        self.price_buffer.append(price)
        if volume > 0:
            self.volume_buffer.append(volume)
        
        # Calcola return se abbiamo dati sufficienti
        if self.price_buffer.size() >= 2:
            prev_prices = self.price_buffer.get_window(2)
            if len(prev_prices) >= 2:
                ret = (prev_prices[-1] / prev_prices[-2]) - 1.0
                self.return_buffer.append(ret)
    
    def get_custom_buffer(self, name: str) -> StatisticalBuffer:
        """Ottiene o crea un buffer personalizzato"""
        if name not in self.custom_buffers:
            self.custom_buffers[name] = StatisticalBuffer()
        return self.custom_buffers[name]
    
    def clear_cache(self, force: bool = False):
        """Pulisce la cache se necessario"""
        if force or self._last_cache_clear > self._cache_max_age:
            self._correlation_cache.clear()
            self._regression_cache.clear()
            self._last_cache_clear = 0
        else:
            self._last_cache_clear += 1

# ========================================================================
# METODI DA AGGIUNGERE ALLA CLASSE CompiledStrategy
# ========================================================================

def __init_math_statistical_tracking__(self, max_buffer_size: int = 1000):
    """Inizializza il sistema di tracking matematico e statistico"""
    self._math_stat_tracker = MathStatTracker(max_buffer_size)
    self._current_bar = 0

def _update_math_stat_data(self):
    """Aggiorna i dati per i calcoli statistici (chiamare in next())"""
    try:
        current_price = float(self.data.close[0])
        current_volume = float(self.data.volume[0]) if hasattr(self.data, 'volume') else 0.0
        
        self._math_stat_tracker.add_price_data(current_price, current_volume)
        self._current_bar += 1
        
        # Pulizia cache periodica
        if self._current_bar % 50 == 0:
            self._math_stat_tracker.clear_cache()
            
    except Exception as e:
        self._handle_error(f"Math stat data update error: {e}")

# ========================================================================
# CORRELATION & COVARIANCE FUNCTIONS
# ========================================================================

def Correlation(self, series1_name: str, series2_name: str, length: int) -> float:
    """Calcola la correlazione tra due serie di dati"""
    try:
        if length <= 1:
            return 0.0
        
        # Check cache
        cache_key = f"{series1_name}_{series2_name}_{length}_{self._current_bar}"
        if cache_key in self._math_stat_tracker._correlation_cache:
            return self._math_stat_tracker._correlation_cache[cache_key]
        
        # Ottieni i buffer delle serie
        buffer1 = self._math_stat_tracker.get_custom_buffer(series1_name)
        buffer2 = self._math_stat_tracker.get_custom_buffer(series2_name)
        
        data1 = buffer1.get_window(length)
        data2 = buffer2.get_window(length)
        
        if len(data1) < 2 or len(data2) < 2 or len(data1) != len(data2):
            return 0.0
        
        # Calcola correlazione usando numpy per performance
        correlation = float(np.corrcoef(data1, data2)[0, 1])
        
        if math.isnan(correlation):
            correlation = 0.0
        
        # Cache result
        self._math_stat_tracker._correlation_cache[cache_key] = correlation
        return correlation
        
    except Exception as e:
        self._handle_error(f"Correlation error: {e}")
        return 0.0

def Covariance(self, series1_name: str, series2_name: str, length: int) -> float:
    """Calcola la covarianza tra due serie di dati"""
    try:
        if length <= 1:
            return 0.0
        
        buffer1 = self._math_stat_tracker.get_custom_buffer(series1_name)
        buffer2 = self._math_stat_tracker.get_custom_buffer(series2_name)
        
        data1 = buffer1.get_window(length)
        data2 = buffer2.get_window(length)
        
        if len(data1) < 2 or len(data2) < 2 or len(data1) != len(data2):
            return 0.0
        
        # Calcola covarianza
        covariance = float(np.cov(data1, data2)[0, 1])
        
        return covariance if not math.isnan(covariance) else 0.0
        
    except Exception as e:
        self._handle_error(f"Covariance error: {e}")
        return 0.0

def PriceCorrelation(self, length: int) -> float:
    """Calcola l'autocorrelazione dei prezzi"""
    try:
        if length <= 1:
            return 0.0
        
        prices = self._math_stat_tracker.price_buffer.get_window(length * 2)
        if len(prices) < length * 2:
            return 0.0
        
        # Split in due serie per autocorrelazione
        series1 = prices[:-length]  # Prima metà
        series2 = prices[length:]   # Seconda metà
        
        if len(series1) != len(series2):
            return 0.0
        
        correlation = float(np.corrcoef(series1, series2)[0, 1])
        return correlation if not math.isnan(correlation) else 0.0
        
    except Exception as e:
        self._handle_error(f"PriceCorrelation error: {e}")
        return 0.0

# ========================================================================
# LINEAR REGRESSION FUNCTIONS
# ========================================================================

def LinearRegSlope(self, series_name: str, length: int) -> float:
    """Calcola la pendenza della regressione lineare"""
    try:
        regression = self._calculate_linear_regression(series_name, length)
        return regression.slope if regression else 0.0
        
    except Exception as e:
        self._handle_error(f"LinearRegSlope error: {e}")
        return 0.0

def LinearRegIntercept(self, series_name: str, length: int) -> float:
    """Calcola l'intercetta della regressione lineare"""
    try:
        regression = self._calculate_linear_regression(series_name, length)
        return regression.intercept if regression else 0.0
        
    except Exception as e:
        self._handle_error(f"LinearRegIntercept error: {e}")
        return 0.0

def LinearRegValue(self, series_name: str, length: int, bar_ago: int = 0) -> float:
    """Calcola il valore della regressione lineare per una barra specifica"""
    try:
        regression = self._calculate_linear_regression(series_name, length)
        if not regression:
            return 0.0
        
        # Calcola il valore per la barra richiesta
        x = length - 1 - bar_ago  # Posizione nella serie
        return regression.slope * x + regression.intercept
        
    except Exception as e:
        self._handle_error(f"LinearRegValue error: {e}")
        return 0.0

def RSquare(self, series_name: str, length: int) -> float:
    """Calcola il coefficiente di determinazione R²"""
    try:
        regression = self._calculate_linear_regression(series_name, length)
        return regression.r_squared if regression else 0.0
        
    except Exception as e:
        self._handle_error(f"RSquare error: {e}")
        return 0.0

def LinearRegAngle(self, series_name: str, length: int) -> float:
    """Calcola l'angolo della regressione lineare in gradi"""
    try:
        regression = self._calculate_linear_regression(series_name, length)
        if not regression:
            return 0.0
        
        # Converti pendenza in angolo
        angle_radians = math.atan(regression.slope)
        angle_degrees = math.degrees(angle_radians)
        
        return angle_degrees
        
    except Exception as e:
        self._handle_error(f"LinearRegAngle error: {e}")
        return 0.0

def _calculate_linear_regression(self, series_name: str, length: int) -> Optional[RegressionResult]:
    """Calcola la regressione lineare completa"""
    try:
        if length <= 2:
            return None
        
        # Check cache
        cache_key = f"reg_{series_name}_{length}_{self._current_bar}"
        if cache_key in self._math_stat_tracker._regression_cache:
            return self._math_stat_tracker._regression_cache[cache_key]
        
        # Ottieni i dati
        if series_name == "PRICE":
            data = self._math_stat_tracker.price_buffer.get_window(length)
        elif series_name == "VOLUME":
            data = self._math_stat_tracker.volume_buffer.get_window(length)
        else:
            buffer = self._math_stat_tracker.get_custom_buffer(series_name)
            data = buffer.get_window(length)
        
        if len(data) < length:
            return None
        
        # Prepara dati per regressione
        x = np.arange(len(data), dtype=float)
        y = np.array(data, dtype=float)
        
        # Calcola regressione usando scipy per accuracy
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        result = RegressionResult(
            slope=float(slope),
            intercept=float(intercept),
            r_squared=float(r_value ** 2),
            correlation=float(r_value),
            std_error=float(std_err),
            p_value=float(p_value)
        )
        
        # Cache result
        self._math_stat_tracker._regression_cache[cache_key] = result
        return result
        
    except Exception as e:
        self._handle_error(f"Linear regression calculation error: {e}")
        return None

# ========================================================================
# STANDARD DEVIATION & VARIANCE FUNCTIONS
# ========================================================================

def StandardDeviation(self, series_name: str, length: int, sample: bool = True) -> float:
    """Calcola la deviazione standard"""
    try:
        if length <= 1:
            return 0.0
        
        if series_name == "PRICE":
            data = self._math_stat_tracker.price_buffer.get_window(length)
        elif series_name == "VOLUME":
            data = self._math_stat_tracker.volume_buffer.get_window(length)
        elif series_name == "RETURNS":
            data = self._math_stat_tracker.return_buffer.get_window(length)
        else:
            buffer = self._math_stat_tracker.get_custom_buffer(series_name)
            data = buffer.get_window(length)
        
        if len(data) < 2:
            return 0.0
        
        # Calcola deviazione standard
        if sample and len(data) > 1:
            # Sample standard deviation (N-1)
            std_dev = float(np.std(data, ddof=1))
        else:
            # Population standard deviation (N)
            std_dev = float(np.std(data, ddof=0))
        
        return std_dev if not math.isnan(std_dev) else 0.0
        
    except Exception as e:
        self._handle_error(f"StandardDeviation error: {e}")
        return 0.0

def Variance(self, series_name: str, length: int, sample: bool = True) -> float:
    """Calcola la varianza"""
    try:
        if length <= 1:
            return 0.0
        
        if series_name == "PRICE":
            data = self._math_stat_tracker.price_buffer.get_window(length)
        elif series_name == "VOLUME":
            data = self._math_stat_tracker.volume_buffer.get_window(length)
        elif series_name == "RETURNS":
            data = self._math_stat_tracker.return_buffer.get_window(length)
        else:
            buffer = self._math_stat_tracker.get_custom_buffer(series_name)
            data = buffer.get_window(length)
        
        if len(data) < 2:
            return 0.0
        
        # Calcola varianza
        if sample and len(data) > 1:
            # Sample variance (N-1)
            variance = float(np.var(data, ddof=1))
        else:
            # Population variance (N)
            variance = float(np.var(data, ddof=0))
        
        return variance if not math.isnan(variance) else 0.0
        
    except Exception as e:
        self._handle_error(f"Variance error: {e}")
        return 0.0

def CoefficientOfVariation(self, series_name: str, length: int) -> float:
    """Calcola il coefficiente di variazione (CV = std/mean)"""
    try:
        if length <= 1:
            return 0.0
        
        std_dev = self.StandardDeviation(series_name, length)
        mean_val = self.Average(series_name, length)
        
        if mean_val == 0.0:
            return 0.0
        
        cv = std_dev / abs(mean_val)
        return cv if not math.isnan(cv) else 0.0
        
    except Exception as e:
        self._handle_error(f"CoefficientOfVariation error: {e}")
        return 0.0

# ========================================================================
# SKEWNESS & KURTOSIS FUNCTIONS
# ========================================================================

def Skewness(self, series_name: str, length: int) -> float:
    """Calcola l'asimmetria (skewness) della distribuzione"""
    try:
        if length <= 3:
            return 0.0
        
        if series_name == "PRICE":
            data = self._math_stat_tracker.price_buffer.get_window(length)
        elif series_name == "VOLUME":
            data = self._math_stat_tracker.volume_buffer.get_window(length)
        elif series_name == "RETURNS":
            data = self._math_stat_tracker.return_buffer.get_window(length)
        else:
            buffer = self._math_stat_tracker.get_custom_buffer(series_name)
            data = buffer.get_window(length)
        
        if len(data) < 3:
            return 0.0
        
        # Calcola skewness usando scipy
        skew = float(stats.skew(data))
        return skew if not math.isnan(skew) else 0.0
        
    except Exception as e:
        self._handle_error(f"Skewness error: {e}")
        return 0.0

def Kurtosis(self, series_name: str, length: int) -> float:
    """Calcola la curtosi (kurtosis) della distribuzione"""
    try:
        if length <= 4:
            return 0.0
        
        if series_name == "PRICE":
            data = self._math_stat_tracker.price_buffer.get_window(length)
        elif series_name == "VOLUME":
            data = self._math_stat_tracker.volume_buffer.get_window(length)
        elif series_name == "RETURNS":
            data = self._math_stat_tracker.return_buffer.get_window(length)
        else:
            buffer = self._math_stat_tracker.get_custom_buffer(series_name)
            data = buffer.get_window(length)
        
        if len(data) < 4:
            return 0.0
        
        # Calcola kurtosis usando scipy (excess kurtosis, 0 = normal distribution)
        kurt = float(stats.kurtosis(data))
        return kurt if not math.isnan(kurt) else 0.0
        
    except Exception as e:
        self._handle_error(f"Kurtosis error: {e}")
        return 0.0

def JarqueBera(self, series_name: str, length: int) -> Tuple[float, float]:
    """Test di normalità Jarque-Bera, restituisce (statistic, p_value)"""
    try:
        if length <= 6:
            return 0.0, 1.0
        
        if series_name == "RETURNS":
            data = self._math_stat_tracker.return_buffer.get_window(length)
        else:
            buffer = self._math_stat_tracker.get_custom_buffer(series_name)
            data = buffer.get_window(length)
        
        if len(data) < 6:
            return 0.0, 1.0
        
        # Test Jarque-Bera
        jb_stat, p_value = stats.jarque_bera(data)
        
        return float(jb_stat), float(p_value)
        
    except Exception as e:
        self._handle_error(f"JarqueBera error: {e}")
        return 0.0, 1.0

# ========================================================================
# PERCENTILE & QUANTILE FUNCTIONS
# ========================================================================

def Percentile(self, series_name: str, length: int, percentile: float) -> float:
    """Calcola il percentile specificato"""
    try:
        if length <= 0 or percentile < 0 or percentile > 100:
            return 0.0
        
        if series_name == "PRICE":
            data = self._math_stat_tracker.price_buffer.get_window(length)
        elif series_name == "VOLUME":
            data = self._math_stat_tracker.volume_buffer.get_window(length)
        elif series_name == "RETURNS":
            data = self._math_stat_tracker.return_buffer.get_window(length)
        else:
            buffer = self._math_stat_tracker.get_custom_buffer(series_name)
            data = buffer.get_window(length)
        
        if len(data) == 0:
            return 0.0
        
        # Calcola percentile
        result = float(np.percentile(data, percentile))
        return result if not math.isnan(result) else 0.0
        
    except Exception as e:
        self._handle_error(f"Percentile error: {e}")
        return 0.0

def Quartile(self, series_name: str, length: int, quartile: int) -> float:
    """Calcola il quartile specificato (1, 2, 3)"""
    try:
        if quartile not in [1, 2, 3]:
            return 0.0
        
        percentiles = {1: 25.0, 2: 50.0, 3: 75.0}
        return self.Percentile(series_name, length, percentiles[quartile])
        
    except Exception as e:
        self._handle_error(f"Quartile error: {e}")
        return 0.0

def InterquartileRange(self, series_name: str, length: int) -> float:
    """Calcola la distanza interquartile (IQR = Q3 - Q1)"""
    try:
        q1 = self.Quartile(series_name, length, 1)
        q3 = self.Quartile(series_name, length, 3)
        
        return q3 - q1
        
    except Exception as e:
        self._handle_error(f"InterquartileRange error: {e}")
        return 0.0

def MedianAbsoluteDeviation(self, series_name: str, length: int) -> float:
    """Calcola la deviazione assoluta mediana (MAD)"""
    try:
        if length <= 1:
            return 0.0
        
        if series_name == "PRICE":
            data = self._math_stat_tracker.price_buffer.get_window(length)
        elif series_name == "VOLUME":
            data = self._math_stat_tracker.volume_buffer.get_window(length)
        elif series_name == "RETURNS":
            data = self._math_stat_tracker.return_buffer.get_window(length)
        else:
            buffer = self._math_stat_tracker.get_custom_buffer(series_name)
            data = buffer.get_window(length)
        
        if len(data) == 0:
            return 0.0
        
        # Calcola MAD
        median = float(np.median(data))
        deviations = [abs(x - median) for x in data]
        mad = float(np.median(deviations))
        
        return mad if not math.isnan(mad) else 0.0
        
    except Exception as e:
        self._handle_error(f"MedianAbsoluteDeviation error: {e}")
        return 0.0

# ========================================================================
# MOVING STATISTICS FUNCTIONS
# ========================================================================

def MovingStandardDeviation(self, series_name: str, length: int, lookback: int) -> float:
    """Calcola la deviazione standard mobile su un lookback period"""
    try:
        if lookback <= 0 or length <= 0:
            return 0.0
        
        # Calcola std dev per ogni periodo nel lookback
        std_devs = []
        for i in range(lookback):
            # Ottieni dati con offset
            if series_name == "PRICE":
                all_data = self._math_stat_tracker.price_buffer.get_window(length + i)
                if len(all_data) >= length + i:
                    data = all_data[i:i+length]
                else:
                    continue
            else:
                buffer = self._math_stat_tracker.get_custom_buffer(series_name)
                all_data = buffer.get_window(length + i)
                if len(all_data) >= length + i:
                    data = all_data[i:i+length]
                else:
                    continue
            
            if len(data) >= 2:
                std_dev = float(np.std(data, ddof=1))
                if not math.isnan(std_dev):
                    std_devs.append(std_dev)
        
        if not std_devs:
            return 0.0
        
        # Media delle deviazioni standard
        return sum(std_devs) / len(std_devs)
        
    except Exception as e:
        self._handle_error(f"MovingStandardDeviation error: {e}")
        return 0.0

def MovingCorrelation(self, series1_name: str, series2_name: str, length: int, lookback: int) -> float:
    """Calcola la correlazione mobile su un lookback period"""
    try:
        if lookback <= 0 or length <= 1:
            return 0.0
        
        correlations = []
        for i in range(lookback):
            # Calcola correlazione con offset
            buffer1 = self._math_stat_tracker.get_custom_buffer(series1_name)
            buffer2 = self._math_stat_tracker.get_custom_buffer(series2_name)
            
            data1 = buffer1.get_window(length + i)[i:i+length] if buffer1.size() >= length + i else []
            data2 = buffer2.get_window(length + i)[i:i+length] if buffer2.size() >= length + i else []
            
            if len(data1) >= 2 and len(data2) >= 2 and len(data1) == len(data2):
                corr = float(np.corrcoef(data1, data2)[0, 1])
                if not math.isnan(corr):
                    correlations.append(corr)
        
        if not correlations:
            return 0.0
        
        return sum(correlations) / len(correlations)
        
    except Exception as e:
        self._handle_error(f"MovingCorrelation error: {e}")
        return 0.0

def RollingBeta(self, market_series: str, stock_series: str, length: int) -> float:
    """Calcola il beta rolling (Cov(stock,market) / Var(market))"""
    try:
        if length <= 2:
            return 0.0
        
        cov = self.Covariance(stock_series, market_series, length)
        var = self.Variance(market_series, length)
        
        if var == 0.0:
            return 0.0
        
        beta = cov / var
        return beta if not math.isnan(beta) else 0.0
        
    except Exception as e:
        self._handle_error(f"RollingBeta error: {e}")
        return 0.0

# ========================================================================
# HELPER FUNCTIONS FOR SERIES MANAGEMENT
# ========================================================================

def AddToSeries(self, series_name: str, value: float) -> bool:
    """Aggiunge un valore a una serie personalizzata"""
    try:
        buffer = self._math_stat_tracker.get_custom_buffer(series_name)
        buffer.append(float(value))
        return True
        
    except Exception as e:
        self._handle_error(f"AddToSeries error: {e}")
        return False

def GetSeriesValue(self, series_name: str, bars_ago: int = 0) -> float:
    """Ottiene un valore da una serie personalizzata"""
    try:
        if bars_ago < 0:
            return 0.0
        
        if series_name == "PRICE":
            buffer = self._math_stat_tracker.price_buffer
        elif series_name == "VOLUME":
            buffer = self._math_stat_tracker.volume_buffer
        elif series_name == "RETURNS":
            buffer = self._math_stat_tracker.return_buffer
        else:
            buffer = self._math_stat_tracker.get_custom_buffer(series_name)
        
        if buffer.size() <= bars_ago:
            return 0.0
        
        data = buffer.get_window(bars_ago + 1)
        return data[0] if data else 0.0
        
    except Exception as e:
        self._handle_error(f"GetSeriesValue error: {e}")
        return 0.0

def Average(self, series_name: str, length: int) -> float:
    """Calcola la media di una serie"""
    try:
        if length <= 0:
            return 0.0
        
        if series_name == "PRICE":
            data = self._math_stat_tracker.price_buffer.get_window(length)
        elif series_name == "VOLUME":
            data = self._math_stat_tracker.volume_buffer.get_window(length)
        elif series_name == "RETURNS":
            data = self._math_stat_tracker.return_buffer.get_window(length)
        else:
            buffer = self._math_stat_tracker.get_custom_buffer(series_name)
            data = buffer.get_window(length)
        
        if not data:
            return 0.0
        
        avg = sum(data) / len(data)
        return avg if not math.isnan(avg) else 0.0
        
    except Exception as e:
        self._handle_error(f"Average error: {e}")
        return 0.0

def Median(self, series_name: str, length: int) -> float:
    """Calcola la mediana di una serie"""
    try:
        if length <= 0:
            return 0.0
        
        if series_name == "PRICE":
            data = self._math_stat_tracker.price_buffer.get_window(length)
        elif series_name == "VOLUME":
            data = self._math_stat_tracker.volume_buffer.get_window(length)
        elif series_name == "RETURNS":
            data = self._math_stat_tracker.return_buffer.get_window(length)
        else:
            buffer = self._math_stat_tracker.get_custom_buffer(series_name)
            data = buffer.get_window(length)
        
        if not data:
            return 0.0
        
        median = float(np.median(data))
        return median if not math.isnan(median) else 0.0
        
    except Exception as e:
        self._handle_error(f"Median error: {e}")
        return 0.0

def Mode(self, series_name: str, length: int) -> float:
    """Calcola la moda di una serie (valore più frequente)"""
    try:
        if length <= 0:
            return 0.0
        
        if series_name == "PRICE":
            data = self._math_stat_tracker.price_buffer.get_window(length)
        elif series_name == "VOLUME":
            data = self._math_stat_tracker.volume_buffer.get_window(length)
        elif series_name == "RETURNS":
            data = self._math_stat_tracker.return_buffer.get_window(length)
        else:
            buffer = self._math_stat_tracker.get_custom_buffer(series_name)
            data = buffer.get_window(length)
        
        if not data:
            return 0.0
        
        # Per dati continui, arrotonda e trova la moda
        rounded_data = [round(x, 4) for x in data]  # 4 decimali
        try:
            mode_result = statistics.mode(rounded_data)
            return float(mode_result)
        except statistics.StatisticsError:
            # Se non c'è una moda chiara, restituisce il primo valore
            return data[0] if data else 0.0
        
    except Exception as e:
        self._handle_error(f"Mode error: {e}")
        return 0.0

# ========================================================================
# ADVANCED MATHEMATICAL FUNCTIONS
# ========================================================================

def ZScore(self, series_name: str, length: int, value: Optional[float] = None) -> float:
    """Calcola il Z-Score per un valore (o il valore corrente)"""
    try:
        if length <= 1:
            return 0.0
        
        if value is None:
            # Usa il valore corrente della serie
            if series_name == "PRICE":
                value = float(self.data.close[0])
            elif series_name == "VOLUME":
                value = float(self.data.volume[0])
            else:
                buffer = self._math_stat_tracker.get_custom_buffer(series_name)
                if buffer.size() > 0:
                    data = buffer.get_window(1)
                    value = data[0] if data else 0.0
                else:
                    return 0.0
        
        mean = self.Average(series_name, length)
        std_dev = self.StandardDeviation(series_name, length)
        
        if std_dev == 0.0:
            return 0.0
        
        z_score = (value - mean) / std_dev
        return z_score if not math.isnan(z_score) else 0.0
        
    except Exception as e:
        self._handle_error(f"ZScore error: {e}")
        return 0.0

def NormalizeValue(self, value: float, series_name: str, length: int, method: str = "zscore") -> float:
    """Normalizza un valore usando diversi metodi"""
    try:
        if length <= 1:
            return value
        
        if method.lower() == "zscore":
            return self.ZScore(series_name, length, value)
        
        elif method.lower() == "minmax":
            # Min-Max normalization [0, 1]
            if series_name == "PRICE":
                data = self._math_stat_tracker.price_buffer.get_window(length)
            elif series_name == "VOLUME":
                data = self._math_stat_tracker.volume_buffer.get_window(length)
            else:
                buffer = self._math_stat_tracker.get_custom_buffer(series_name)
                data = buffer.get_window(length)
            
            if not data:
                return value
            
            min_val = min(data)
            max_val = max(data)
            
            if max_val == min_val:
                return 0.0
            
            normalized = (value - min_val) / (max_val - min_val)
            return normalized
        
        elif method.lower() == "robust":
            # Robust normalization usando mediana e IQR
            median = self.Median(series_name, length)
            iqr = self.InterquartileRange(series_name, length)
            
            if iqr == 0.0:
                return 0.0
            
            normalized = (value - median) / iqr
            return normalized
        
        else:
            return value
        
    except Exception as e:
        self._handle_error(f"NormalizeValue error: {e}")
        return value

def OutlierDetection(self, series_name: str, length: int, method: str = "zscore", threshold: float = 3.0) -> bool:
    """Detecta se il valore corrente è un outlier"""
    try:
        if length <= 1:
            return False
        
        current_value = self.GetSeriesValue(series_name, 0)
        
        if method.lower() == "zscore":
            z_score = self.ZScore(series_name, length, current_value)
            return abs(z_score) > threshold
        
        elif method.lower() == "iqr":
            q1 = self.Quartile(series_name, length, 1)
            q3 = self.Quartile(series_name, length, 3)
            iqr = q3 - q1
            
            if iqr == 0.0:
                return False
            
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            
            return current_value < lower_bound or current_value > upper_bound
        
        elif method.lower() == "mad":
            median = self.Median(series_name, length)
            mad = self.MedianAbsoluteDeviation(series_name, length)
            
            if mad == 0.0:
                return False
            
            # Modified Z-score using MAD
            modified_z = 0.6745 * (current_value - median) / mad
            return abs(modified_z) > threshold
        
        return False
        
    except Exception as e:
        self._handle_error(f"OutlierDetection error: {e}")
        return False

def TrendStrength(self, series_name: str, length: int) -> float:
    """Calcola la forza del trend usando R²"""
    try:
        if length <= 2:
            return 0.0
        
        r_squared = self.RSquare(series_name, length)
        slope = self.LinearRegSlope(series_name, length)
        
        # Trend strength considera sia R² che direzione
        trend_strength = r_squared * (1.0 if slope >= 0 else -1.0)
        
        return trend_strength
        
    except Exception as e:
        self._handle_error(f"TrendStrength error: {e}")
        return 0.0

def CyclicalIndicator(self, series_name: str, length: int) -> float:
    """Calcola un indicatore di comportamento ciclico"""
    try:
        if length <= 4:
            return 0.0
        
        # Usa autocorrelazione per detectare cicli
        autocorr = self.PriceCorrelation(length // 2)
        
        # Combina con detrended data variance
        detrended_variance = self._calculate_detrended_variance(series_name, length)
        total_variance = self.Variance(series_name, length)
        
        if total_variance == 0.0:
            return 0.0
        
        cyclical_component = detrended_variance / total_variance
        cyclical_indicator = autocorr * cyclical_component
        
        return cyclical_indicator if not math.isnan(cyclical_indicator) else 0.0
        
    except Exception as e:
        self._handle_error(f"CyclicalIndicator error: {e}")
        return 0.0

def _calculate_detrended_variance(self, series_name: str, length: int) -> float:
    """Calcola la varianza dei dati detrended"""
    try:
        # Ottieni dati originali
        if series_name == "PRICE":
            data = self._math_stat_tracker.price_buffer.get_window(length)
        else:
            buffer = self._math_stat_tracker.get_custom_buffer(series_name)
            data = buffer.get_window(length)
        
        if len(data) < 3:
            return 0.0
        
        # Calcola trend lineare
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data, 1)
        trend = np.polyval(coeffs, x)
        
        # Calcola residui (dati detrended)
        residuals = np.array(data) - trend
        
        # Varianza dei residui
        detrended_var = float(np.var(residuals, ddof=1))
        return detrended_var if not math.isnan(detrended_var) else 0.0
        
    except Exception as e:
        self._handle_error(f"Detrended variance calculation error: {e}")
        return 0.0

# ========================================================================
# STATISTICAL TESTS FUNCTIONS
# ========================================================================

def TTest(self, series1_name: str, series2_name: str, length: int) -> Tuple[float, float]:
    """T-test per confrontare due serie, restituisce (statistic, p_value)"""
    try:
        if length <= 2:
            return 0.0, 1.0
        
        buffer1 = self._math_stat_tracker.get_custom_buffer(series1_name)
        buffer2 = self._math_stat_tracker.get_custom_buffer(series2_name)
        
        data1 = buffer1.get_window(length)
        data2 = buffer2.get_window(length)
        
        if len(data1) < 2 or len(data2) < 2:
            return 0.0, 1.0
        
        # Independent t-test
        t_stat, p_value = stats.ttest_ind(data1, data2)
        
        return float(t_stat), float(p_value)
        
    except Exception as e:
        self._handle_error(f"TTest error: {e}")
        return 0.0, 1.0

def KSTest(self, series_name: str, length: int, distribution: str = "norm") -> Tuple[float, float]:
    """Kolmogorov-Smirnov test di normalità, restituisce (statistic, p_value)"""
    try:
        if length <= 3:
            return 0.0, 1.0
        
        if series_name == "RETURNS":
            data = self._math_stat_tracker.return_buffer.get_window(length)
        else:
            buffer = self._math_stat_tracker.get_custom_buffer(series_name)
            data = buffer.get_window(length)
        
        if len(data) < 3:
            return 0.0, 1.0
        
        if distribution.lower() == "norm":
            # Test contro distribuzione normale
            mean = np.mean(data)
            std = np.std(data)
            ks_stat, p_value = stats.kstest(data, lambda x: stats.norm.cdf(x, mean, std))
        else:
            # Altri distributions possono essere aggiunti
            ks_stat, p_value = stats.kstest(data, distribution)
        
        return float(ks_stat), float(p_value)
        
    except Exception as e:
        self._handle_error(f"KSTest error: {e}")
        return 0.0, 1.0

def AndersonDarling(self, series_name: str, length: int) -> Tuple[float, float]:
    """Anderson-Darling test di normalità"""
    try:
        if length <= 3:
            return 0.0, 1.0
        
        if series_name == "RETURNS":
            data = self._math_stat_tracker.return_buffer.get_window(length)
        else:
            buffer = self._math_stat_tracker.get_custom_buffer(series_name)
            data = buffer.get_window(length)
        
        if len(data) < 3:
            return 0.0, 1.0
        
        # Anderson-Darling test
        ad_stat, critical_values, significance_level = stats.anderson(data, dist='norm')
        
        # Approssimiamo p-value basato su critical values
        p_value = 0.05  # Default
        if ad_stat < critical_values[2]:  # 5% level
            p_value = 0.05
        elif ad_stat < critical_values[1]:  # 2.5% level  
            p_value = 0.025
        elif ad_stat < critical_values[0]:  # 1% level
            p_value = 0.01
        else:
            p_value = 0.001
        
        return float(ad_stat), float(p_value)
        
    except Exception as e:
        self._handle_error(f"AndersonDarling error: {e}")
        return 0.0, 1.0

# ========================================================================
# ESEMPI DI USO IN EASYLANGUAGE
# ========================================================================

"""
ESEMPIO 1 - Analisi correlazione tra asset:
vars: 
    Correlation1(0), CorrThreshold(0.7),
    IsCorrelated(false);

// Aggiungi dati a serie personalizzate
AddToSeries("SPY_CLOSE", SPY_Close);
AddToSeries("QQQ_CLOSE", QQQ_Close);

// Calcola correlazione rolling
Correlation1 = Correlation("SPY_CLOSE", "QQQ_CLOSE", 20);
IsCorrelated = AbsValue(Correlation1) > CorrThreshold;

if IsCorrelated then begin
    Print("High correlation detected: ", NumToStr(Correlation1, 3));
    // Strategia pairs trading
    if Correlation1 > CorrThreshold and Close > Close[1] then
        Buy;
end;

ESEMPIO 2 - Mean reversion con Z-Score:
vars: 
    ZScore1(0), ZThreshold(2.0),
    IsOverbought(false), IsOversold(false);

// Calcola Z-Score del prezzo corrente
ZScore1 = ZScore("PRICE", 50);

IsOverbought = ZScore1 > ZThreshold;
IsOversold = ZScore1 < -ZThreshold;

if IsOversold then begin
    Buy("MeanRev_Long");
    Alert("Oversold condition - Z-Score: " + NumToStr(ZScore1, 2));
end;

if IsOverbought then begin
    Sell Short("MeanRev_Short");  
    Alert("Overbought condition - Z-Score: " + NumToStr(ZScore1, 2));
end;

ESEMPIO 3 - Trend strength analysis:
vars: 
    TrendStr(0), RSquared1(0), Slope1(0),
    StrongTrend(false), TrendDirection(0);

// Aggiungi prezzo a serie per analisi
AddToSeries("PRICE", Close);

TrendStr = TrendStrength("PRICE", 30);
RSquared1 = RSquare("PRICE", 30);
Slope1 = LinearRegSlope("PRICE", 30);

StrongTrend = AbsValue(TrendStr) > 0.6;
TrendDirection = 1;
if Slope1 < 0 then TrendDirection = -1;

if StrongTrend and TrendDirection = 1 then begin
    // Trend following strategy
    if Close > LinearRegValue("PRICE", 30, 0) then
        Buy("Trend_Long");
end;

ESEMPIO 4 - Volatility clustering detection:
vars: 
    Returns1(0), ReturnStdDev(0), ReturnSkew(0),
    ReturnKurt(0), IsHighVol(false);

// Calcola return
if Close[1] > 0 then begin
    Returns1 = (Close / Close[1]) - 1;
    AddToSeries("RETURNS", Returns1);
end;

// Analisi distribuzione returns
ReturnStdDev = StandardDeviation("RETURNS", 20);
ReturnSkew = Skewness("RETURNS", 20);  
ReturnKurt = Kurtosis("RETURNS", 20);

IsHighVol = ReturnStdDev > Average(ReturnStdDev, 50) * 1.5;

if IsHighVol then begin
    // Riduci posizioni durante alta volatilità
    Print("High volatility period detected");
    Print("Std Dev: ", NumToStr(ReturnStdDev, 4));
    Print("Skewness: ", NumToStr(ReturnSkew, 3));
    Print("Kurtosis: ", NumToStr(ReturnKurt, 3));
end;

ESEMPIO 5 - Outlier detection e filtering:
vars: 
    IsOutlier(false), FilteredPrice(0),
    CleanPrice(0);

// Aggiungi prezzo corrente
AddToSeries("PRICE", Close);

// Detecta outliers
IsOutlier = OutlierDetection("PRICE", 30, "zscore", 2.5);

if not IsOutlier then begin
    CleanPrice = Close;
    // Usa solo dati "puliti" per trading
    if CleanPrice > Average(CleanPrice, 20) then Buy;
end else begin
    Alert("Outlier detected: " + NumToStr(Close, 2));
    CleanPrice = Median("PRICE", 5);  // Usa mediana come fallback
end;

ESEMPIO 6 - Regime detection con statistical tests:
vars: 
    JBStat(0), JBPValue(0), IsNormalDistrib(false),
    KSStat(0), KSPValue(0), RegimeChange(false);

// Test distribuzione returns
if GetSeriesValue("RETURNS", 0) <> 0 then begin
    Value1 = JarqueBera("RETURNS", 50);
    JBStat = Value1;  // Prima parte del risultato
    // JBPValue sarebbe la seconda parte
    
    Value2 = KSTest("RETURNS", 50);
    KSStat = Value2;
    
    IsNormalDistrib = JBStat < 6.0;  // Soglia approssimativa
    
    if not IsNormalDistrib then begin
        Print("Non-normal return distribution detected");
        // Adatta strategia per regime non-normale
        RegimeChange = true;
    end;
end;

ESEMPIO 7 - Portfolio correlation matrix:
vars: 
    CorrSPY_QQQ(0), CorrSPY_IWM(0), CorrQQQ_IWM(0),
    AvgCorrelation(0), DiversificationRatio(0);

// Mantieni dati di multipli asset
AddToSeries("SPY", SPY_Close);
AddToSeries("QQQ", QQQ_Close);  
AddToSeries("IWM", IWM_Close);

// Calcola correlazioni
CorrSPY_QQQ = Correlation("SPY", "QQQ", 30);
CorrSPY_IWM = Correlation("SPY", "IWM", 30);
CorrQQQ_IWM = Correlation("QQQ", "IWM", 30);

AvgCorrelation = (CorrSPY_QQQ + CorrSPY_IWM + CorrQQQ_IWM) / 3;

// Calcola diversification ratio semplificato
DiversificationRatio = 1 - AvgCorrelation;

if DiversificationRatio < 0.3 then begin
    Alert("Low diversification - High correlation detected");
    // Riduci esposizione o cerca asset alternativi
end;

ESEMPIO 8 - Advanced momentum con regression analysis:
vars: 
    RegSlope(0), RegAngle(0), RegRSquare(0),
    MomentumStrength(0), QualityMomentum(false);

AddToSeries("PRICE", Close);

RegSlope = LinearRegSlope("PRICE", 20);
RegAngle = LinearRegAngle("PRICE", 20);  
RegRSquare = RSquare("PRICE", 20);

// Momentum quality = slope * R²
MomentumStrength = RegSlope * RegRSquare;
QualityMomentum = MomentumStrength > 0 and RegRSquare > 0.7;

if QualityMomentum then begin
    if RegAngle > 15 then begin  // Trend > 15 gradi
        Buy("Quality_Momentum");
        Print("Quality momentum detected - Angle: ", NumToStr(RegAngle, 1));
    end;
end;
"""

# ========================================================================
# TEST CASES
# ========================================================================

def run_math_statistical_tests():
    """Test suite per le funzioni matematiche e statistiche"""
    print("Running Math & Statistical Functions Tests...")
    
    # Test 1: Inizializzazione
    class MockStrategy:
        def __init__(self):
            self.__init_math_statistical_tracking__()
            
        def _handle_error(self, msg):
            print(f"Error: {msg}")
            
        class MockData:
            def __init__(self):
                self.close = [100.0]
                self.volume = [1000.0]
                
        def __init__(self):
            self.__init_math_statistical_tracking__()
            self.data = self.MockData()
            
    strategy = MockStrategy()
    
    # Test buffer creation
    buffer = strategy._math_stat_tracker.get_custom_buffer("TEST")
    assert isinstance(buffer, StatisticalBuffer), "Buffer creation failed"
    
    print("✓ Initialization tests passed")
    
    # Test 2: Serie management
    test_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    
    for value in test_data:
        strategy.AddToSeries("TEST_SERIES", value)
    
    # Test average
    avg = strategy.Average("TEST_SERIES", 5)
    expected_avg = sum(test_data[-5:]) / 5  # Ultimi 5 valori: 6,7,8,9,10 = 8.0
    assert abs(avg - expected_avg) < 0.001, f"Average test failed: expected {expected_avg}, got {avg}"
    
    print("✓ Series management tests passed")
    
    # Test 3: Statistical calculations
    std_dev = strategy.StandardDeviation("TEST_SERIES", 5)
    expected_std = np.std([6.0, 7.0, 8.0, 9.0, 10.0], ddof=1)
    assert abs(std_dev - expected_std) < 0.001, f"StdDev test failed: expected {expected_std}, got {std_dev}"
    
    # Test median
    median = strategy.Median("TEST_SERIES", 5)
    expected_median = 8.0  # Mediana di [6,7,8,9,10]
    assert abs(median - expected_median) < 0.001, f"Median test failed: expected {expected_median}, got {median}"
    
    print("✓ Basic statistical calculations tests passed")
    
    # Test 4: Regression calculations
    # Crea serie con trend lineare chiaro
    linear_data = [i * 2 + 1 for i in range(10)]  # y = 2x + 1
    for value in linear_data:
        strategy.AddToSeries("LINEAR_SERIES", value)
    
    slope = strategy.LinearRegSlope("LINEAR_SERIES", 10)
    # La pendenza dovrebbe essere circa 2.0
    assert abs(slope - 2.0) < 0.1, f"Linear regression slope test failed: expected ~2.0, got {slope}"
    
    r_squared = strategy.RSquare("LINEAR_SERIES", 10)
    # R² dovrebbe essere molto vicino a 1.0 per dati lineari perfetti
    assert r_squared > 0.99, f"R-squared test failed: expected >0.99, got {r_squared}"
    
    print("✓ Regression analysis tests passed")
    
    # Test 5: Percentile calculations
    percentile_25 = strategy.Percentile("TEST_SERIES", 10, 25)
    percentile_75 = strategy.Percentile("TEST_SERIES", 10, 75)
    
    expected_p25 = np.percentile(test_data, 25)
    expected_p75 = np.percentile(test_data, 75)
    
    assert abs(percentile_25 - expected_p25) < 0.001, f"P25 test failed: expected {expected_p25}, got {percentile_25}"
    assert abs(percentile_75 - expected_p75) < 0.001, f"P75 test failed: expected {expected_p75}, got {percentile_75}"
    
    print("✓ Percentile calculations tests passed")
    
    # Test 6: Z-Score calculation
    # Aggiungi valore anomalo per test Z-Score
    strategy.AddToSeries("ZSCORE_TEST", 50.0)  # Outlier
    for i in range(1, 11):
        strategy.AddToSeries("ZSCORE_TEST", float(i))  # Valori normali 1-10
    
    z_score = strategy.ZScore("ZSCORE_TEST", 10, 50.0)
    assert abs(z_score) > 2.0, f"Z-Score test failed: expected >2.0, got {abs(z_score)}"
    
    print("✓ Z-Score calculation tests passed")
    
    # Test 7: Correlation
    # Crea due serie correlate
    for i in range(20):
        strategy.AddToSeries("SERIES_A", float(i))
        strategy.AddToSeries("SERIES_B", float(i * 0.8 + 2))  # Correlazione positiva
    
    correlation = strategy.Correlation("SERIES_A", "SERIES_B", 20)
    assert correlation > 0.9, f"Correlation test failed: expected >0.9, got {correlation}"
    
    print("✓ Correlation calculation tests passed")
    
    print("All Math & Statistical Functions Tests PASSED! ✓")

if __name__ == "__main__":
    run_math_statistical_tests()

# ========================================================================
# ISTRUZIONI DI INTEGRAZIONE AVANZATE
# ========================================================================

"""
INTEGRAZIONE COMPLETA NELLA CLASSE CompiledStrategy:

1. DIPENDENZE NECESSARIE:
   pip install numpy scipy

2. IMPORT RICHIESTI:
   import math
   import numpy as np
   from collections import deque
   from typing import Optional, List, Tuple, Union, Dict, Any
   from dataclasses import dataclass
   import statistics
   from scipy import stats
   import warnings

3. INIZIALIZZAZIONE NELLA CLASSE:
   def __init__(self):
       super().__init__()
       # ... altre inizializzazioni ...
       self.__init_math_statistical_tracking__(1000)  # buffer size
       
   def next(self):
       # IMPORTANTE: Chiamare questo all'inizio di next()
       self._update_math_stat_data()
       
       # ... resto della logica strategy ...

4. FUNZIONI DISPONIBILI NEL PARSER:
   
   CORRELATION_FUNCTIONS = [
       'Correlation', 'Covariance', 'PriceCorrelation', 'MovingCorrelation', 'RollingBeta'
   ]
   
   REGRESSION_FUNCTIONS = [
       'LinearRegSlope', 'LinearRegIntercept', 'LinearRegValue', 'RSquare', 'LinearRegAngle'
   ]
   
   STATISTICAL_FUNCTIONS = [
       'StandardDeviation', 'Variance', 'CoefficientOfVariation', 'Skewness', 'Kurtosis',
       'JarqueBera', 'Percentile', 'Quartile', 'InterquartileRange', 'MedianAbsoluteDeviation',
       'Average', 'Median', 'Mode'
   ]
   
   ADVANCED_FUNCTIONS = [
       'ZScore', 'NormalizeValue', 'OutlierDetection', 'TrendStrength', 'CyclicalIndicator'
   ]
   
   SERIES_FUNCTIONS = [
       'AddToSeries', 'GetSeriesValue'
   ]
   
   TEST_FUNCTIONS = [
       'TTest', 'KSTest', 'AndersonDarling'
   ]

5. CODE GENERATION EXAMPLES:
   
   // EasyLanguage:
   Correlation1 = Correlation("SERIES1", "SERIES2", 20);
   
   // Python:
   Correlation1 = self.Correlation("SERIES1", "SERIES2", 20)
   
   // EasyLanguage:
   AddToSeries("MY_DATA", Close);
   ZScore1 = ZScore("MY_DATA", 50);
   
   // Python:
   self.AddToSeries("MY_DATA", self.data.close[0])
   ZScore1 = self.ZScore("MY_DATA", 50)

6. USAGE PATTERNS NEL PARSER:

   a) Serie Management Pattern:
   ```
   // EasyLanguage pattern:
   vars: MyValue(0);
   MyValue = Close * Volume;
   AddToSeries("CUSTOM_INDICATOR", MyValue);
   
   // Diventa:
   MyValue = self.data.close[0] * self.data.volume[0]
   self.AddToSeries("CUSTOM_INDICATOR", MyValue)
   ```

   b) Statistical Analysis Pattern:
   ```
   // EasyLanguage pattern:
   vars: Mean1(0), StdDev1(0), ZScore1(0);
   Mean1 = Average("PRICE", 20);
   StdDev1 = StandardDeviation("PRICE", 20);  
   ZScore1 = ZScore("PRICE", 20);
   
   // Diventa:
   Mean1 = self.Average("PRICE", 20)
   StdDev1 = self.StandardDeviation("PRICE", 20)
   ZScore1 = self.ZScore("PRICE", 20)
   ```

   c) Regression Analysis Pattern:
   ```
   // EasyLanguage pattern:
   vars: Slope1(0), RSquare1(0), RegValue(0);
   Slope1 = LinearRegSlope("PRICE", 30);
   RSquare1 = RSquare("PRICE", 30);
   RegValue = LinearRegValue("PRICE", 30, 0);
   
   // Diventa:
   Slope1 = self.LinearRegSlope("PRICE", 30)
   RSquare1 = self.RSquare("PRICE", 30)
   RegValue = self.LinearRegValue("PRICE", 30, 0)
   ```

7. PERFORMANCE OPTIMIZATIONS:

   - Cache automatica per calcoli costosi (correlazioni, regressioni)
   - Buffer circolari per memory efficiency
   - Calcoli lazy dove possibile
   - Batch processing per operazioni multiple

8. SERIES NAMING CONVENTIONS:

   Built-in series:
   - "PRICE" = prezzi di chiusura automatici
   - "VOLUME" = volumi automatici  
   - "RETURNS" = returns calcolati automaticamente
   
   Custom series: qualsiasi nome definito dall'utente
   - "SPY_CLOSE", "MOMENTUM", "CUSTOM_INDICATOR", etc.

9. MEMORY MANAGEMENT:

   Default buffer size: 1000 bars
   Configurabile in inizializzazione:
   
   self.__init_math_statistical_tracking__(2000)  # 2000 bars buffer

10. ERROR HANDLING:

    Tutte le funzioni includono:
    - Validazione input
    - Check NaN/Infinite values
    - Graceful degradation
    - Logging errors via _handle_error()

11. STATISTICAL ACCURACY:

    - Usa scipy.stats per accuracy scientifica
    - Sample vs population statistics supportate
    - Multiple statistical tests disponibili
    - Robust statistics (MAD, IQR) per outliers

12. INTEGRATION CHECKLIST:

    ✅ Aggiungi imports necessari
    ✅ Inizializza tracking in __init__()
    ✅ Chiama _update_math_stat_data() in next()
    ✅ Aggiungi function recognition nel parser
    ✅ Implementa code generation per tutte le funzioni
    ✅ Test con dati reali
    ✅ Verifica performance con buffer grandi
    ✅ Documenta custom series naming conventions

ESEMPIO COMPLETO DI INTEGRAZIONE:

```python
class CompiledStrategy(bt.Strategy):
    def __init__(self):
        super().__init__()
        
        # Performance tracking (modulo 1)
        self.__init_performance_tracking__()
        
        # Session tracking (modulo 2)  
        self.__init_session_tracking__("America/New_York")
        
        # Math & Statistical tracking (modulo 3)
        self.__init_math_statistical_tracking__(1000)
        
        # ... resto inizializzazioni strategy ...
        
    def next(self):
        # Update trackers
        self._update_math_stat_data()
        
        # Esempio uso combinato dei tre moduli:
        
        # 1. Performance analysis
        if self.TotalTrades() > 10:
            profit_factor = self.ProfitFactor()
            
        # 2. Session analysis  
        if self.IsInSession("NYSE_REGULAR"):
            # 3. Statistical analysis
            self.AddToSeries("PRICE", self.data.close[0])
            z_score = self.ZScore("PRICE", 50)
            
            # Combined logic
            if abs(z_score) > 2.0 and profit_factor > 1.2:
                # Mean reversion in profitable strategy
                if z_score > 2.0:
                    self.sell()  # Overbought
                elif z_score < -2.0:
                    self.buy()   # Oversold
```

ADVANCED USAGE EXAMPLES:

```easylanguage
// Multi-factor quantitative model
vars: 
    MomentumScore(0), MeanRevScore(0), VolScore(0),
    CompositeScore(0), TrendQuality(0);

// Momentum factor
AddToSeries("PRICE", Close);
MomentumScore = LinearRegSlope("PRICE", 20) * RSquare("PRICE", 20);

// Mean reversion factor  
MeanRevScore = -AbsValue(ZScore("PRICE", 50));

// Volatility factor
AddToSeries("RETURNS", (Close/Close[1])-1);
VolScore = -StandardDeviation("RETURNS", 20);

// Quality check
TrendQuality = RSquare("PRICE", 30);

// Composite scoring
CompositeScore = MomentumScore * 0.4 + MeanRevScore * 0.3 + VolScore * 0.3;

// Trading logic with quality filter
if TrendQuality > 0.6 then begin
    if CompositeScore > 0.5 then Buy;
    if CompositeScore < -0.5 then Sell Short;
end;

// Portfolio correlation monitoring
vars: CorrMatrix(0), DiversificationRatio(0);

AddToSeries("ASSET1", Asset1_Close);
AddToSeries("ASSET2", Asset2_Close); 
AddToSeries("ASSET3", Asset3_Close);

CorrMatrix = (Correlation("ASSET1","ASSET2",30) + 
              Correlation("ASSET1","ASSET3",30) + 
              Correlation("ASSET2","ASSET3",30)) / 3;

DiversificationRatio = 1 - AbsValue(CorrMatrix);

if DiversificationRatio < 0.3 then
    Alert("Portfolio concentration risk detected");
```

LIMITATIONS AND CONSIDERATIONS:

1. **Memory Usage**: Large buffers (>2000 bars) may impact performance
2. **Calculation Intensity**: Some functions (regression, correlation) are CPU intensive
3. **Data Requirements**: Many statistical functions need minimum data points
4. **Cache Management**: Cache automatically cleared every 50 bars
5. **NaN Handling**: Functions return 0.0 for invalid/NaN results
6. **Series Persistence**: Custom series persist throughout strategy lifetime

For complete mathematical reference:
- Correlation: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
- Linear Regression: https://en.wikipedia.org/wiki/Linear_regression
- Statistical Tests: https://docs.scipy.org/doc/scipy/reference/stats.html

Per il manuale EasyLanguage originale:
https://cdn.tradestation.com/uploads/EasyLanguage-Essentials.pdf (Chapters 12-14)
"""