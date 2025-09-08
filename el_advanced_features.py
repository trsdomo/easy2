"""
EasyLanguage Python Library - Advanced Features and Special Functions
Auto-generated on 2025-09-09 01:00:13

This module contains Advanced Features and Special Functions 
compatible with TradeStation EasyLanguage.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, List, Tuple, Any
from datetime import datetime, date, time
import math
import re

# Module-specific imports will be added as needed



# Functions in this module (50):
# - print_medium_priority_summary (from easylanguage_medium_priority_features.py)
# - compile_custom_indicators_system (from easylanguage_medium_priority_features.py)
# - compile_showme_paintbar_system (from easylanguage_medium_priority_features.py)
# - compile_session_info_system (from easylanguage_medium_priority_features.py)
# - compile_symbol_info_system (from easylanguage_medium_priority_features.py)
# - compile_volume_analysis_system (from easylanguage_medium_priority_features.py)
# - compile_all_medium_priority_features (from easylanguage_medium_priority_features.py)
# - Data (from easylanguage_ultime1.py)
# - ParameterOptimization (from easylanguage_ultime2.py)
# - WalkForwardAnalysis (from easylanguage_ultime2.py)
# - GeneticAlgorithmOptimization (from easylanguage_ultime2.py)
# - PerformanceMetricsTracking (from easylanguage_ultime2.py)
# - PortfolioManagerEnabled (from easylanguage_ultime2.py)
# - SharesPerDollar (from easylanguage_ultime2.py)
# - DollarRisk (from easylanguage_ultime2.py)
# - MaxShares (from easylanguage_ultime2.py)
# - PctEquity (from easylanguage_ultime2.py)
# - PercentProfit (from easylanguage_ultime2.py)
# - MaxDrawDown (from easylanguage_ultime2.py)
# - Slippage (from easylanguage_ultime2.py)
# - Commission (from easylanguage_ultime2.py)
# - Array_SetMaxIndex (from easylanguage_ultime2.py)
# - Array_Sort (from easylanguage_ultime2.py)
# - Array_Sum (from easylanguage_ultime2.py)
# - HighestArray (from easylanguage_ultime2.py)
# - LowestArray (from easylanguage_ultime2.py)
# - AverageArray (from easylanguage_ultime2.py)
# - SortArray (from easylanguage_ultime2.py)
# - Sort2DArray (from easylanguage_ultime2.py)
# - DateAdd (from easylanguage_ultime2.py)
# - DateDiff (from easylanguage_ultime2.py)
# - TimeAdd (from easylanguage_ultime2.py)
# - TimeDiff (from easylanguage_ultime2.py)
# - FormatTime (from easylanguage_ultime2.py)
# - _handle_error (from easylanguage_ultime2.py)
# - _optimize_on_data (from easylanguage_ultime2.py)
# - __init_missing_functions__ (from easylanguage_ultime2.py)
# - optimize_recursive (from easylanguage_ultime2.py)
# - parse_date (from easylanguage_ultime2.py)
# - time_to_minutes (from easylanguage_ultime2.py)
# - compile_enhanced_plot_system (from easylanguage_ultime_features_claude.py)
# - compile_color_system (from easylanguage_ultime_features_claude.py)
# - compile_quote_fields_system (from easylanguage_ultime_features_claude.py)
# - compile_fundamental_data_system (from easylanguage_ultime_features_claude.py)
# - compile_multimedia_commentary_system (from easylanguage_ultime_features_claude.py)
# - compile_advanced_stop_system (from easylanguage_ultime_features_claude.py)
# - compile_realtime_barstate_system (from easylanguage_ultime_features_claude.py)
# - compile_user_functions_system (from easylanguage_ultime_features_claude.py)
# - compile_option_system (from easylanguage_ultime_features_claude.py)
# - compile_advanced_math_system (from easylanguage_ultime_features_claude.py)


# Function: print_medium_priority_summary
# Source: easylanguage_medium_priority_features.py
def print_medium_priority_summary():
    """Stampa riepilogo priorità media"""
    print("=== MEDIUM PRIORITY FEATURES COMPLETED ===")
    for feature in MEDIUM_PRIORITY_COMPLETED:
        print(feature)
        
    print("\\n=== NEXT: LOW PRIORITY FEATURES ===")
    for feature in LOW_PRIORITY_NEXT:
        print(feature)
    
    print("\\n=== ESEMPI DI CODICE SUPPORTATI ===")
    print(MEDIUM_PRIORITY_EXAMPLES)


# Function: compile_custom_indicators_system
# Source: easylanguage_medium_priority_features.py
    def compile_custom_indicators_system(self):
        """Implementa sistema per indicatori personalizzati"""
        code = '''
    # ============== CUSTOM INDICATORS SYSTEM ==============
    
    def _init_custom_indicators(self):
        """Inizializza sistema indicatori personalizzati"""
        self._custom_indicators = {}
        self._indicator_values = {}
        self._indicator_plots = {}
        
    def CreateIndicator(self, name, calculation_func, length=14, **params):
        """Crea un indicatore personalizzato"""
        self._custom_indicators[name] = {
            'func': calculation_func,
            'length': length,
            'params': params,
            'history': [],
            'current_value': 0
        }
        self.log(f"INDICATOR CREATED: {name}")
        return name
    
    def UpdateIndicator(self, name, value=None):
        """Aggiorna valore di un indicatore"""
        if name not in self._custom_indicators:
            return False
            
        if value is None:
            # Calcola automaticamente usando la funzione
            indicator = self._custom_indicators[name]
            try:
                value = indicator['func'](self, **indicator['params'])
            except:
                value = 0
        
        # Aggiorna storia
        indicator = self._custom_indicators[name]
        indicator['history'].append(value)
        indicator['current_value'] = value
        
        # Mantieni solo length elementi
        if len(indicator['history']) > indicator['length'] * 2:
            indicator['history'] = indicator['history'][-indicator['length']:]
        
        self._indicator_values[name] = value
        return True
    
    def GetIndicatorValue(self, name, bars_back=0):
        """Ottiene valore di un indicatore personalizzato"""
        if name not in self._custom_indicators:
            return 0
            
        indicator = self._custom_indicators[name]
        history = indicator['history']
        
        if len(history) == 0:
            return 0
        
        idx = len(history) - 1 - bars_back
        if idx >= 0 and idx < len(history):
            return history[idx]
        return 0
    
    # Indicatori tecnici standard implementabili
    def Custom_RSI(self, length=14, price_series=None):
        """RSI personalizzato"""
        if price_series is None:
            price_series = [self.dataclose[-i] for i in range(length + 1, -1, -1)]
        
        if len(price_series) < length + 1:
            return 50  # Valore neutro
        
        gains = []
        losses = []
        
        for i in range(1, len(price_series)):
            change = price_series[i] - price_series[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(-change)
        
        if len(gains) < length:
            return 50
            
        avg_gain = sum(gains[-length:]) / length
        avg_loss = sum(losses[-length:]) / length
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def Custom_MACD(self, fast=12, slow=26, signal=9):
        """MACD personalizzato"""
        if len(self.data) < slow:
            return {'macd': 0, 'signal': 0, 'histogram': 0}
        
        # EMA veloce
        fast_ema = self._calculate_ema(fast)
        slow_ema = self._calculate_ema(slow)
        
        macd_line = fast_ema - slow_ema
        
        # Segnale EMA del MACD (simulato)
        signal_line = macd_line * 0.9  # Semplificato
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line, 
            'histogram': histogram
        }
    
    def Custom_Stochastic(self, k_length=14, k_smooth=3, d_length=3):
        """Stocastico personalizzato"""
        if len(self.data) < k_length:
            return {'k': 50, 'd': 50}
        
        # Ottieni high/low per k_length barre
        highs = [self.datahigh[-i] for i in range(k_length, -1, -1)]
        lows = [self.datalow[-i] for i in range(k_length, -1, -1)]
        
        highest_high = max(highs)
        lowest_low = min(lows)
        
        if highest_high == lowest_low:
            k_raw = 50
        else:
            k_raw = ((self.dataclose[0] - lowest_low) / (highest_high - lowest_low)) * 100
        
        # %K smoothed
        k_smooth = k_raw  # Semplificato
        
        # %D (media mobile di %K)
        d_smooth = k_smooth  # Semplificato
        
        return {'k': k_smooth, 'd': d_smooth}
    
    def _calculate_ema(self, length):
        """Calcola EMA semplificato"""
        if len(self.data) < length:
            return self.dataclose[0]
            
        alpha = 2.0 / (length + 1)
        prices = [self.dataclose[-i] for i in range(length, -1, -1)]
        
        ema = prices[0]
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
            
        return ema
    
    def DefineIndicatorProperty(self, name, property_name, value):
        """Definisce proprietà di un indicatore"""
        if name not in self._custom_indicators:
            return False
            
        if 'properties' not in self._custom_indicators[name]:
            self._custom_indicators[name]['properties'] = {}
            
        self._custom_indicators[name]['properties'][property_name] = value
        return True
'''
        return code


# Function: compile_showme_paintbar_system
# Source: easylanguage_medium_priority_features.py
    def compile_showme_paintbar_system(self):
        """Implementa ShowMe e PaintBar studies"""
        code = '''
    # ============== SHOWME/PAINTBAR STUDIES ==============
    
    def _init_showme_paintbar(self):
        """Inizializza sistema ShowMe/PaintBar"""
        self._showme_markers = {}
        self._paintbar_conditions = {}
        self._showme_counter = 0
        self._paintbar_counter = 0
    
    def ShowMe_Create(self, condition, price=None, color="Yellow", shape="Dot", size=8, name=None):
        """Crea un marcatore ShowMe"""
        if not condition:
            return None
            
        self._showme_counter += 1
        
        if name is None:
            name = f"ShowMe_{self._showme_counter}"
        
        if price is None:
            price = self.dataclose[0]
        
        marker_id = f"sm_{self._showme_counter}"
        
        showme_marker = {
            'id': marker_id,
            'name': name,
            'date': self.data.datetime.datetime(0),
            'price': price,
            'color': color,
            'shape': shape,
            'size': size,
            'bar': len(self.data)
        }
        
        self._showme_markers[marker_id] = showme_marker
        
        self.log(f"SHOWME: {name} at {price:.4f} [{color} {shape}]")
        return marker_id
    
    def ShowMe_UpArrow(self, condition, price=None, color="Green"):
        """ShowMe freccia verso l'alto"""
        return self.ShowMe_Create(condition, price, color, "UpArrow", name="UpArrow")
    
    def ShowMe_DownArrow(self, condition, price=None, color="Red"):
        """ShowMe freccia verso il basso"""
        return self.ShowMe_Create(condition, price, color, "DownArrow", name="DownArrow")
    
    def ShowMe_Dot(self, condition, price=None, color="Yellow"):
        """ShowMe punto"""
        return self.ShowMe_Create(condition, price, color, "Dot", name="Dot")
    
    def ShowMe_Square(self, condition, price=None, color="Blue"):
        """ShowMe quadrato"""
        return self.ShowMe_Create(condition, price, color, "Square", name="Square")
    
    def ShowMe_Diamond(self, condition, price=None, color="Magenta"):
        """ShowMe diamante"""
        return self.ShowMe_Create(condition, price, color, "Diamond", name="Diamond")
    
    def PaintBar_Create(self, condition, color="Yellow", alpha=0.5, name=None):
        """Crea una PaintBar"""
        if not condition:
            return None
            
        self._paintbar_counter += 1
        
        if name is None:
            name = f"PaintBar_{self._paintbar_counter}"
        
        paintbar_id = f"pb_{self._paintbar_counter}"
        
        paintbar = {
            'id': paintbar_id,
            'name': name,
            'date': self.data.datetime.datetime(0),
            'high': self.datahigh[0],
            'low': self.datalow[0],
            'open': self.dataopen[0],
            'close': self.dataclose[0],
            'color': color,
            'alpha': alpha,
            'bar': len(self.data)
        }
        
        self._paintbar_conditions[paintbar_id] = paintbar
        
        self.log(f"PAINTBAR: {name} [{color}] OHLC: {paintbar['open']:.2f}/{paintbar['high']:.2f}/{paintbar['low']:.2f}/{paintbar['close']:.2f}")
        return paintbar_id
    
    def PaintBar_Bullish(self, condition, color="Green"):
        """PaintBar rialzista"""
        return self.PaintBar_Create(condition, color, name="Bullish")
    
    def PaintBar_Bearish(self, condition, color="Red"):
        """PaintBar ribassista"""
        return self.PaintBar_Create(condition, color, name="Bearish")
    
    def PaintBar_Neutral(self, condition, color="Gray"):
        """PaintBar neutrale"""
        return self.PaintBar_Create(condition, color, name="Neutral")
    
    def PaintBar_Volume(self, condition, color="Blue"):
        """PaintBar per volume"""
        return self.PaintBar_Create(condition, color, name="HighVolume")
    
    def GetShowMeCount(self):
        """Conta i ShowMe attivi"""
        return len(self._showme_markers)
    
    def GetPaintBarCount(self):
        """Conta le PaintBar attive"""
        return len(self._paintbar_conditions)
    
    def ClearShowMe(self, marker_id=None):
        """Rimuove ShowMe markers"""
        if marker_id:
            if marker_id in self._showme_markers:
                del self._showme_markers[marker_id]
                return True
        else:
            # Pulisci tutti
            self._showme_markers.clear()
            return True
        return False
    
    def ClearPaintBars(self, paintbar_id=None):
        """Rimuove PaintBars"""
        if paintbar_id:
            if paintbar_id in self._paintbar_conditions:
                del self._paintbar_conditions[paintbar_id]
                return True
        else:
            # Pulisci tutte
            self._paintbar_conditions.clear()
            return True
        return False
'''
        return code


# Function: compile_session_info_system
# Source: easylanguage_medium_priority_features.py
    def compile_session_info_system(self):
        """Implementa funzioni informazioni di sessione"""
        code = '''
    # ============== SESSION INFORMATION FUNCTIONS ==============
    
    def _init_session_info(self):
        """Inizializza sistema informazioni sessione"""
        self._session_times = {
            'regular_start': 930,    # 9:30 AM
            'regular_end': 1600,     # 4:00 PM  
            'premarket_start': 400,  # 4:00 AM
            'premarket_end': 930,    # 9:30 AM
            'aftermarket_start': 1600, # 4:00 PM
            'aftermarket_end': 2000,   # 8:00 PM
        }
        
        # Sessioni per futures (24h)
        self._futures_sessions = {
            'electronic_start': 1800,  # 6:00 PM (previous day)
            'electronic_end': 1700,    # 5:00 PM
            'pit_start': 830,          # 8:30 AM  
            'pit_end': 1315,           # 1:15 PM
        }
        
        self._current_session = 'regular'
        
    def SessionStartTime(self, session_num=1, session_type=1):
        """Ottiene l'orario di inizio sessione"""
        current_time = self.Time()
        
        if session_type == 1:  # Regular session
            return self._session_times['regular_start']
        elif session_type == 2:  # Pre-market
            return self._session_times['premarket_start']
        elif session_type == 3:  # After-market
            return self._session_times['aftermarket_start']
        else:
            return self._session_times['regular_start']
    
    def SessionEndTime(self, session_num=1, session_type=1):
        """Ottiene l'orario di fine sessione"""
        if session_type == 1:  # Regular session
            return self._session_times['regular_end']
        elif session_type == 2:  # Pre-market
            return self._session_times['premarket_end']
        elif session_type == 3:  # After-market
            return self._session_times['aftermarket_end']
        else:
            return self._session_times['regular_end']
    
    def IsRegularSession(self):
        """Verifica se siamo in sessione regolare"""
        current_time = self.Time()
        return (self._session_times['regular_start'] <= current_time <= self._session_times['regular_end'])
    
    def IsPreMarket(self):
        """Verifica se siamo in pre-market"""
        current_time = self.Time()
        return (self._session_times['premarket_start'] <= current_time < self._session_times['premarket_end'])
    
    def IsAfterMarket(self):
        """Verifica se siamo in after-market"""
        current_time = self.Time()
        return (self._session_times['aftermarket_start'] <= current_time <= self._session_times['aftermarket_end'])
    
    def GetCurrentSession(self):
        """Ottiene la sessione corrente"""
        if self.IsRegularSession():
            return 'regular'
        elif self.IsPreMarket():
            return 'premarket'
        elif self.IsAfterMarket():
            return 'aftermarket'
        else:
            return 'closed'
    
    def SessionFirstBarTime(self):
        """Orario della prima barra della sessione"""
        return self.SessionStartTime()
    
    def SessionLastBarTime(self):
        """Orario dell'ultima barra della sessione"""
        return self.SessionEndTime()
    
    def BarsInSession(self):
        """Numero di barre nella sessione corrente"""
        # Calcolo semplificato basato su timeframe 1 minuto
        session_start = self.SessionStartTime()
        session_end = self.SessionEndTime()
        
        # Converti in minuti
        start_minutes = (session_start // 100) * 60 + (session_start % 100)
        end_minutes = (session_end // 100) * 60 + (session_end % 100)
        
        return max(0, end_minutes - start_minutes)
    
    def MinutesFromSessionStart(self):
        """Minuti dall'inizio della sessione"""
        current_time = self.Time()
        session_start = self.SessionStartTime()
        
        current_minutes = (current_time // 100) * 60 + (current_time % 100)
        start_minutes = (session_start // 100) * 60 + (session_start % 100)
        
        return max(0, current_minutes - start_minutes)
    
    def MinutesToSessionEnd(self):
        """Minuti alla fine della sessione"""
        current_time = self.Time()
        session_end = self.SessionEndTime()
        
        current_minutes = (current_time // 100) * 60 + (current_time % 100)
        end_minutes = (session_end // 100) * 60 + (session_end % 100)
        
        return max(0, end_minutes - current_minutes)
    
    def IsFirstBarOfSession(self):
        """Verifica se è la prima barra della sessione"""
        return self.MinutesFromSessionStart() <= 1
    
    def IsLastBarOfSession(self):
        """Verifica se è l'ultima barra della sessione"""
        return self.MinutesToSessionEnd() <= 1
    
    def SetCustomSessionTimes(self, start_time, end_time, session_type='custom'):
        """Imposta orari di sessione personalizzati"""
        if session_type == 'custom':
            self._session_times['custom_start'] = start_time
            self._session_times['custom_end'] = end_time
        
        self.log(f"CUSTOM SESSION: {start_time} - {end_time}")
        return True
'''
        return code


# Function: compile_symbol_info_system
# Source: easylanguage_medium_priority_features.py
    def compile_symbol_info_system(self):
        """Implementa funzioni informazioni simboli"""
        code = '''
    # ============== SYMBOL INFORMATION FUNCTIONS ==============
    
    def _init_symbol_info(self):
        """Inizializza sistema informazioni simbolo"""
        self._symbol_info = {
            'name': getattr(self.data, '_name', 'UNKNOWN'),
            'big_point_value': 1.0,
            'point_value': 0.01,
            'min_move': 0.01,
            'price_scale': 100,
            'currency': 'USD',
            'exchange': 'UNKNOWN',
            'sector': 'UNKNOWN',
            'tick_size': 0.01,
            'contract_size': 1,
            'margin_requirement': 0
        }
    
    def GetSymbolName(self):
        """Ottiene il nome del simbolo"""
        return self._symbol_info['name']
    
    def BigPointValue(self):
        """Valore di un punto intero"""
        return self._symbol_info['big_point_value']
    
    def PointValue(self):
        """Valore di un punto"""
        return self._symbol_info['point_value']
    
    def MinMove(self):
        """Movimento minimo del prezzo"""
        return self._symbol_info['min_move']
    
    def PriceScale(self):
        """Scala dei prezzi"""
        return self._symbol_info['price_scale']
    
    def GetCurrency(self):
        """Valuta del simbolo"""
        return self._symbol_info['currency']
    
    def GetExchange(self):
        """Exchange del simbolo"""
        return self._symbol_info['exchange']
    
    def GetSector(self):
        """Settore del simbolo"""
        return self._symbol_info['sector']
    
    def TickSize(self):
        """Dimensione del tick"""
        return self._symbol_info['tick_size']
    
    def ContractSize(self):
        """Dimensione del contratto (per futures)"""
        return self._symbol_info['contract_size']
    
    def MarginRequirement(self):
        """Margine richiesto"""
        return self._symbol_info['margin_requirement']
    
    def SetSymbolInfo(self, property_name, value):
        """Imposta informazioni simbolo personalizzate"""
        if property_name in self._symbol_info:
            self._symbol_info[property_name] = value
            self.log(f"SYMBOL INFO: {property_name} = {value}")
            return True
        return False
    
    def IsStock(self):
        """Verifica se è un'azione"""
        symbol = self.GetSymbolName().upper()
        # Logica semplificata - potrebbe essere migliorata
        return len(symbol) <= 5 and symbol.isalpha()
    
    def IsFuture(self):
        """Verifica se è un future"""
        symbol = self.GetSymbolName().upper()
        # Logica semplificata basata su pattern comuni
        future_patterns = ['ES', 'NQ', 'YM', 'RTY', 'GC', 'SI', 'CL', 'NG']
        return any(pattern in symbol for pattern in future_patterns)
    
    def IsForex(self):
        """Verifica se è forex"""
        symbol = self.GetSymbolName().upper()
        # Logica per coppie forex
        forex_patterns = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']
        return len(symbol) == 6 and any(currency in symbol for currency in forex_patterns)
    
    def IsOption(self):
        """Verifica se è un'opzione"""
        symbol = self.GetSymbolName().upper()
        # Logica semplificata per opzioni
        return 'C' in symbol or 'P' in symbol or 'CALL' in symbol or 'PUT' in symbol
    
    def GetInstrumentType(self):
        """Ottiene il tipo di strumento"""
        if self.IsStock():
            return 'STOCK'
        elif self.IsFuture():
            return 'FUTURE'
        elif self.IsForex():
            return 'FOREX'
        elif self.IsOption():
            return 'OPTION'
        else:
            return 'UNKNOWN'
    
    def CalculatePositionValue(self, shares):
        """Calcola valore della posizione"""
        return shares * self.dataclose[0] * self.PointValue()
    
    def CalculateCommission(self, shares, commission_per_share=0.01):
        """Calcola commissioni stimate"""
        return shares * commission_per_share
    
    def CalculateMargin(self, shares):
        """Calcola margine richiesto"""
        return shares * self.MarginRequirement()
'''
        return code


# Function: compile_volume_analysis_system
# Source: easylanguage_medium_priority_features.py
    def compile_volume_analysis_system(self):
        """Implementa funzioni analisi volumi"""
        code = '''
    # ============== VOLUME ANALYSIS FUNCTIONS ==============
    
    def _init_volume_analysis(self):
        """Inizializza sistema analisi volumi"""
        self._volume_history = []
        self._volume_profile = {}
        self._vwap_data = []
        self._volume_indicators = {}
    
    def VWAP(self, length=None):
        """Volume Weighted Average Price"""
        if length is None:
            # VWAP dalla sessione corrente (semplificato)
            length = min(len(self.data), self.MinutesFromSessionStart() + 1)
        
        if length <= 0:
            return self.dataclose[0]
        
        total_volume = 0
        total_pv = 0  # Price * Volume
        
        for i in range(length):
            try:
                price = (self.datahigh[-i] + self.datalow[-i] + self.dataclose[-i]) / 3
                volume = self.datavolume[-i] if self.datavolume[-i] > 0 else 1
                
                total_pv += price * volume
                total_volume += volume
            except:
                continue
        
        return total_pv / total_volume if total_volume > 0 else self.dataclose[0]
    
    def VolumeAverage(self, length=20):
        """Media mobile del volume"""
        if length <= 0 or len(self.data) < length:
            return self.datavolume[0]
        
        total_volume = 0
        for i in range(length):
            try:
                total_volume += self.datavolume[-i]
            except:
                continue
        
        return total_volume / length
    
    def VolumeRatio(self, current_volume=None, average_length=20):
        """Rapporto volume corrente / volume medio"""
        if current_volume is None:
            current_volume = self.datavolume[0]
        
        avg_volume = self.VolumeAverage(average_length)
        
        return current_volume / avg_volume if avg_volume > 0 else 1.0
    
    def IsHighVolume(self, threshold=1.5, average_length=20):
        """Verifica se il volume è alto"""
        return self.VolumeRatio(None, average_length) >= threshold
    
    def IsLowVolume(self, threshold=0.5, average_length=20):
        """Verifica se il volume è basso"""
        return self.VolumeRatio(None, average_length) <= threshold
    
    def VolumeSpike(self, spike_threshold=2.0, length=20):
        """Rileva picchi di volume"""
        return self.VolumeRatio(None, length) >= spike_threshold
    
    def AccumulationDistribution(self, length=14):
        """Indicatore Accumulation/Distribution"""
        if length <= 0 or len(self.data) < length:
            return 0
        
        ad_total = 0
        
        for i in range(length):
            try:
                high = self.datahigh[-i]
                low = self.datalow[-i]
                close = self.dataclose[-i]
                volume = self.datavolume[-i]
                
                if high != low:
                    money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
                    money_flow_volume = money_flow_multiplier * volume
                    ad_total += money_flow_volume
            except:
                continue
        
        return ad_total
    
    def OnBalanceVolume(self, length=14):
        """On Balance Volume"""
        if length <= 0 or len(self.data) < length:
            return 0
        
        obv = 0
        
        for i in range(length - 1, 0, -1):
            try:
                current_close = self.dataclose[-i]
                previous_close = self.dataclose[-i-1]
                volume = self.datavolume[-i]
                
                if current_close > previous_close:
                    obv += volume
                elif current_close < previous_close:
                    obv -= volume
                # Se uguale, OBV rimane invariato
            except:
                continue
        
        return obv
    
    def VolumeProfile(self, price_levels=20):
        """Volume Profile semplificato"""
        if len(self.data) < 2:
            return {}
        
        # Calcola range di prezzo
        min_price = min([self.datalow[-i] for i in range(min(50, len(self.data)))])
        max_price = max([self.datahigh[-i] for i in range(min(50, len(self.data)))])
        
        if max_price == min_price:
            return {max_price: self.datavolume[0]}
        
        price_step = (max_price - min_price) / price_levels
        volume_profile = {}
        
        # Inizializza livelli di prezzo
        for i in range(price_levels):
            price_level = min_price + (i * price_step)
            volume_profile[round(price_level, 4)] = 0
        
        # Distribuisci volume per livelli di prezzo
        for i in range(min(50, len(self.data))):
            try:
                typical_price = (self.datahigh[-i] + self.datalow[-i] + self.dataclose[-i]) / 3
                volume = self.datavolume[-i]
                
                # Trova il livello di prezzo più vicino
                closest_level = min(volume_profile.keys(), key=lambda x: abs(x - typical_price))
                volume_profile[closest_level] += volume
            except:
                continue
        
        return volume_profile
    
    def GetVolumeAtPrice(self, target_price, tolerance=0.1):
        """Ottiene volume a un prezzo specifico"""
        profile = self.VolumeProfile()
        
        total_volume = 0
        for price, volume in profile.items():
            if abs(price - target_price) <= tolerance:
                total_volume += volume
        
        return total_volume
    
    def HighestVolumePrice(self):
        """Prezzo con il volume più alto (POC - Point of Control)"""
        profile = self.VolumeProfile()
        
        if not profile:
            return self.dataclose[0]
        
        max_volume_price = max(profile.items(), key=lambda x: x[1])
        return max_volume_price[0]
    
    def VolumeWeightedMovingAverage(self, length=20):
        """Volume Weighted Moving Average"""
        if length <= 0 or len(self.data) < length:
            return self.dataclose[0]
        
        total_weighted_price = 0
        total_volume = 0
        
        for i in range(length):
            try:
                price = self.dataclose[-i]
                volume = self.datavolume[-i] if self.datavolume[-i] > 0 else 1
                
                total_weighted_price += price * volume
                total_volume += volume
            except:
                continue
        
        return total_weighted_price / total_volume if total_volume > 0 else self.dataclose[0]
    
    def UpVolume(self, bars_back=0):
        """Volume delle barre rialziste"""
        try:
            close = self.dataclose[-bars_back] if bars_back > 0 else self.dataclose[0]
            open_price = self.dataopen[-bars_back] if bars_back > 0 else self.dataopen[0]
            volume = self.datavolume[-bars_back] if bars_back > 0 else self.datavolume[0]
            
            return volume if close > open_price else 0
        except:
            return 0
    
    def DownVolume(self, bars_back=0):
        """Volume delle barre ribassiste"""
        try:
            close = self.dataclose[-bars_back] if bars_back > 0 else self.dataclose[0]
            open_price = self.dataopen[-bars_back] if bars_back > 0 else self.dataopen[0]
            volume = self.datavolume[-bars_back] if bars_back > 0 else self.datavolume[0]
            
            return volume if close < open_price else 0
        except:
            return 0
    
    def UpDownVolumeRatio(self, length=14):
        """Rapporto Up Volume / Down Volume"""
        if length <= 0:
            return 1.0
        
        up_vol_total = 0
        down_vol_total = 0
        
        for i in range(length):
            up_vol_total += self.UpVolume(i)
            down_vol_total += self.DownVolume(i)
        
        if down_vol_total == 0:
            return float('inf') if up_vol_total > 0 else 1.0
        
        return up_vol_total / down_vol_total
    
    def VolumeOscillator(self, short_length=5, long_length=10):
        """Oscillatore del volume"""
        short_vol_avg = self.VolumeAverage(short_length)
        long_vol_avg = self.VolumeAverage(long_length)
        
        if long_vol_avg == 0:
            return 0
        
        return ((short_vol_avg - long_vol_avg) / long_vol_avg) * 100
    
    def VolumeTrend(self, length=14):
        """Trend del volume (positivo = crescente, negativo = calante)"""
        if length <= 1 or len(self.data) < length:
            return 0
        
        recent_avg = self.VolumeAverage(length // 2)
        older_avg = 0
        
        # Calcola media del volume più vecchio
        total = 0
        count = 0
        for i in range(length // 2, length):
            try:
                total += self.datavolume[-i]
                count += 1
            except:
                continue
        
        older_avg = total / count if count > 0 else recent_avg
        
        if older_avg == 0:
            return 0
        
        return ((recent_avg - older_avg) / older_avg) * 100
'''
        return code


# Function: compile_all_medium_priority_features
# Source: easylanguage_medium_priority_features.py
    def compile_all_medium_priority_features(self):
        """Compila tutte le funzionalità a priorità media"""
        
        custom_indicators = CustomIndicatorsCompiler()
        showme_paintbar = ShowMePaintBarCompiler()
        session_info = SessionInfoCompiler()
        symbol_info = SymbolInfoCompiler()
        volume_analysis = VolumeAnalysisCompiler()
        
        return {
            'custom_indicators': custom_indicators.compile_custom_indicators_system(),
            'showme_paintbar': showme_paintbar.compile_showme_paintbar_system(),
            'session_info': session_info.compile_session_info_system(),
            'symbol_info': symbol_info.compile_symbol_info_system(),
            'volume_analysis': volume_analysis.compile_volume_analysis_system()
        }


# Function: Data
# Source: easylanguage_ultime1.py
def Data(self, data_number: int):
    """Reference to specific data stream (Data1, Data2, etc.)"""
    if data_number == 1:
        return self.data  # Primary data stream
    else:
        return self._multi_data.get(data_number, self.data)


# Function: ParameterOptimization
# Source: easylanguage_ultime2.py
def ParameterOptimization(self, param_ranges, objective_function, max_iterations=1000):
    """Parameter optimization loops with grid search"""
    try:
        best_result = None
        best_value = float('-inf')
        iteration = 0
        
        def optimize_recursive(remaining_ranges, current_params):
            nonlocal best_result, best_value, iteration
            
            if iteration >= max_iterations:
                return
                
            if not remaining_ranges:
                # Evaluate current parameter combination
                result = objective_function(current_params)
                if result > best_value:
                    best_value = result
                    best_result = current_params.copy()
                iteration += 1
                return
            
            param_name, param_range = next(iter(remaining_ranges.items()))
            remaining = {k: v for k, v in remaining_ranges.items() if k != param_name}
            
            for value in param_range:
                current_params[param_name] = value
                optimize_recursive(remaining, current_params)
        
        optimize_recursive(param_ranges, {})
        
        return OptimizationResult(
            parameters=best_result or {},
            net_profit=best_value,
            max_drawdown=0.0,
            profit_factor=1.0,
            total_trades=0,
            win_rate=0.0
        )
        
    except Exception as e:
        self._handle_error("ParameterOptimization", e)
        return None


# Function: WalkForwardAnalysis
# Source: easylanguage_ultime2.py
def WalkForwardAnalysis(self, data, strategy_func, in_sample_bars=252, out_sample_bars=63):
    """Walk-forward analysis implementation"""
    try:
        results = []
        total_bars = len(data)
        start_bar = 0
        
        while start_bar + in_sample_bars + out_sample_bars <= total_bars:
            # In-sample optimization period
            in_sample_data = data[start_bar:start_bar + in_sample_bars]
            
            # Out-of-sample testing period  
            out_sample_data = data[start_bar + in_sample_bars:start_bar + in_sample_bars + out_sample_bars]
            
            # Optimize on in-sample data
            best_params = self._optimize_on_data(in_sample_data, strategy_func)
            
            # Test on out-of-sample data
            out_sample_result = strategy_func(out_sample_data, best_params)
            
            results.append({
                'in_sample_start': start_bar,
                'out_sample_start': start_bar + in_sample_bars,
                'parameters': best_params,
                'out_sample_pnl': out_sample_result
            })
            
            start_bar += out_sample_bars
        
        return results
        
    except Exception as e:
        self._handle_error("WalkForwardAnalysis", e)
        return []


# Function: GeneticAlgorithmOptimization
# Source: easylanguage_ultime2.py
def GeneticAlgorithmOptimization(self, param_bounds, fitness_func, population_size=50, generations=100):
    """Genetic algorithm optimization"""
    try:
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = {}
            for param, (min_val, max_val) in param_bounds.items():
                individual[param] = random.uniform(min_val, max_val)
            population.append(individual)
        
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = [(individual, fitness_func(individual)) for individual in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Selection (top 50%)
            survivors = [ind for ind, _ in fitness_scores[:population_size//2]]
            
            # Crossover and mutation
            new_population = survivors.copy()
            while len(new_population) < population_size:
                parent1 = random.choice(survivors)
                parent2 = random.choice(survivors)
                
                # Crossover
                child = {}
                for param in param_bounds:
                    if random.random() < 0.5:
                        child[param] = parent1[param]
                    else:
                        child[param] = parent2[param]
                
                # Mutation (10% chance)
                if random.random() < 0.1:
                    param = random.choice(list(param_bounds.keys()))
                    min_val, max_val = param_bounds[param]
                    child[param] = random.uniform(min_val, max_val)
                
                new_population.append(child)
            
            population = new_population
        
        # Return best individual
        final_fitness = [(ind, fitness_func(ind)) for ind in population]
        best_individual, best_fitness = max(final_fitness, key=lambda x: x[1])
        
        return OptimizationResult(
            parameters=best_individual,
            net_profit=best_fitness,
            max_drawdown=0.0,
            profit_factor=1.0,
            total_trades=0,
            win_rate=0.0
        )
        
    except Exception as e:
        self._handle_error("GeneticAlgorithmOptimization", e)
        return None


# Function: PerformanceMetricsTracking
# Source: easylanguage_ultime2.py
def PerformanceMetricsTracking(self, trades_list):
    """Performance metrics tracking"""
    try:
        if not hasattr(self, '_performance_metrics'):
            self._performance_metrics = {
                'trades': [],
                'equity_curve': [],
                'initial_capital': 100000.0,
                'current_equity': 100000.0
            }
        
        # Add trades to tracking
        for trade in trades_list:
            self._performance_metrics['trades'].append(trade)
            pnl = trade.get('pnl', 0)
            self._performance_metrics['current_equity'] += pnl
            self._performance_metrics['equity_curve'].append(self._performance_metrics['current_equity'])
        
        return True
        
    except Exception as e:
        self._handle_error("PerformanceMetricsTracking", e)
        return False


# Function: PortfolioManagerEnabled
# Source: easylanguage_ultime2.py
def PortfolioManagerEnabled(self, enabled=True):
    """Enable/disable portfolio manager"""
    if not hasattr(self, '_portfolio_manager'):
        self._portfolio_manager = {'enabled': False}
    self._portfolio_manager['enabled'] = enabled
    return enabled


# Function: SharesPerDollar
# Source: easylanguage_ultime2.py
def SharesPerDollar(self, shares=0.01):
    """Set shares per dollar for position sizing"""
    if not hasattr(self, '_portfolio_manager'):
        self._portfolio_manager = {}
    self._portfolio_manager['shares_per_dollar'] = shares
    return shares


# Function: DollarRisk
# Source: easylanguage_ultime2.py
def DollarRisk(self, risk=1000.0):
    """Set dollar risk per position"""
    if not hasattr(self, '_portfolio_manager'):
        self._portfolio_manager = {}
    self._portfolio_manager['dollar_risk'] = risk
    return risk


# Function: MaxShares
# Source: easylanguage_ultime2.py
def MaxShares(self, shares=1000):
    """Set maximum shares per position"""
    if not hasattr(self, '_portfolio_manager'):
        self._portfolio_manager = {}
    self._portfolio_manager['max_shares'] = shares
    return shares


# Function: PctEquity
# Source: easylanguage_ultime2.py
def PctEquity(self, percentage=0.1):
    """Set percentage of equity to risk"""
    if not hasattr(self, '_portfolio_manager'):
        self._portfolio_manager = {}
    self._portfolio_manager['pct_equity'] = percentage
    return percentage


# Function: PercentProfit
# Source: easylanguage_ultime2.py
def PercentProfit(self):
    """Calculate profit percentage"""
    try:
        if not hasattr(self, '_performance_metrics'):
            return 0.0
        
        initial = self._performance_metrics.get('initial_capital', 100000.0)
        current = self._performance_metrics.get('current_equity', 100000.0)
        return ((current - initial) / initial) * 100.0
        
    except Exception as e:
        self._handle_error("PercentProfit", e)
        return 0.0


# Function: MaxDrawDown
# Source: easylanguage_ultime2.py
def MaxDrawDown(self):
    """Calculate maximum drawdown"""
    try:
        if not hasattr(self, '_performance_metrics'):
            return 0.0
        
        equity_curve = self._performance_metrics.get('equity_curve', [])
        if not equity_curve:
            return 0.0
        
        max_dd = 0.0
        peak = equity_curve[0]
        
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd * 100.0  # Return as percentage
        
    except Exception as e:
        self._handle_error("MaxDrawDown", e)
        return 0.0


# Function: Slippage
# Source: easylanguage_ultime2.py
def Slippage(self, amount=0.0):
    """Set/get slippage amount"""
    if not hasattr(self, '_trading_costs'):
        self._trading_costs = {}
    
    if amount != 0.0:
        self._trading_costs['slippage'] = amount
    
    return self._trading_costs.get('slippage', 0.0)


# Function: Commission
# Source: easylanguage_ultime2.py
def Commission(self, amount=0.0):
    """Set/get commission amount"""
    if not hasattr(self, '_trading_costs'):
        self._trading_costs = {}
    
    if amount != 0.0:
        self._trading_costs['commission'] = amount
    
    return self._trading_costs.get('commission', 0.0)


# Function: Array_SetMaxIndex
# Source: easylanguage_ultime2.py
def Array_SetMaxIndex(self, array_name, max_index):
    """Set maximum index for dynamic array"""
    try:
        if not hasattr(self, '_dynamic_arrays'):
            self._dynamic_arrays = {}
        
        if array_name not in self._dynamic_arrays:
            self._dynamic_arrays[array_name] = {}
        
        # Resize array to max_index + 1
        for i in range(max_index + 1):
            if i not in self._dynamic_arrays[array_name]:
                self._dynamic_arrays[array_name][i] = 0.0
        
        return True
        
    except Exception as e:
        self._handle_error("Array_SetMaxIndex", e)
        return False


# Function: Array_Sort
# Source: easylanguage_ultime2.py
def Array_Sort(self, array_name, ascending=True):
    """Sort dynamic array"""
    try:
        if not hasattr(self, '_dynamic_arrays') or array_name not in self._dynamic_arrays:
            return False
        
        array_dict = self._dynamic_arrays[array_name]
        sorted_values = sorted(array_dict.values(), reverse=not ascending)
        
        # Update array with sorted values
        for i, value in enumerate(sorted_values):
            array_dict[i] = value
        
        return True
        
    except Exception as e:
        self._handle_error("Array_Sort", e)
        return False


# Function: Array_Sum
# Source: easylanguage_ultime2.py
def Array_Sum(self, array_name, start_index=0, end_index=None):
    """Sum values in dynamic array"""
    try:
        if not hasattr(self, '_dynamic_arrays') or array_name not in self._dynamic_arrays:
            return 0.0
        
        array_dict = self._dynamic_arrays[array_name]
        
        if end_index is None:
            end_index = max(array_dict.keys()) if array_dict else 0
        
        total = 0.0
        for i in range(start_index, end_index + 1):
            if i in array_dict:
                total += array_dict[i]
        
        return total
        
    except Exception as e:
        self._handle_error("Array_Sum", e)
        return 0.0


# Function: HighestArray
# Source: easylanguage_ultime2.py
def HighestArray(self, array_name, length=None):
    """Find highest value in array"""
    try:
        if not hasattr(self, '_dynamic_arrays') or array_name not in self._dynamic_arrays:
            return 0.0
        
        array_dict = self._dynamic_arrays[array_name]
        values = list(array_dict.values())
        
        if length is not None and length < len(values):
            values = values[-length:]  # Take last 'length' values
        
        return max(values) if values else 0.0
        
    except Exception as e:
        self._handle_error("HighestArray", e)
        return 0.0


# Function: LowestArray
# Source: easylanguage_ultime2.py
def LowestArray(self, array_name, length=None):
    """Find lowest value in array"""
    try:
        if not hasattr(self, '_dynamic_arrays') or array_name not in self._dynamic_arrays:
            return 0.0
        
        array_dict = self._dynamic_arrays[array_name]
        values = list(array_dict.values())
        
        if length is not None and length < len(values):
            values = values[-length:]  # Take last 'length' values
        
        return min(values) if values else 0.0
        
    except Exception as e:
        self._handle_error("LowestArray", e)
        return 0.0


# Function: AverageArray
# Source: easylanguage_ultime2.py
def AverageArray(self, array_name, length=None):
    """Calculate average of array values"""
    try:
        if not hasattr(self, '_dynamic_arrays') or array_name not in self._dynamic_arrays:
            return 0.0
        
        array_dict = self._dynamic_arrays[array_name]
        values = list(array_dict.values())
        
        if length is not None and length < len(values):
            values = values[-length:]  # Take last 'length' values
        
        return sum(values) / len(values) if values else 0.0
        
    except Exception as e:
        self._handle_error("AverageArray", e)
        return 0.0


# Function: SortArray
# Source: easylanguage_ultime2.py
def SortArray(self, array_data, ascending=True):
    """Sort regular array (list)"""
    try:
        if isinstance(array_data, (list, tuple)):
            return sorted(array_data, reverse=not ascending)
        return array_data
        
    except Exception as e:
        self._handle_error("SortArray", e)
        return array_data


# Function: Sort2DArray
# Source: easylanguage_ultime2.py
def Sort2DArray(self, array_2d, column_index=0, ascending=True):
    """Sort 2D array by specified column"""
    try:
        if isinstance(array_2d, (list, tuple)) and array_2d:
            return sorted(array_2d, key=lambda x: x[column_index] if len(x) > column_index else 0, 
                         reverse=not ascending)
        return array_2d
        
    except Exception as e:
        self._handle_error("Sort2DArray", e)
        return array_2d


# Function: DateAdd
# Source: easylanguage_ultime2.py
def DateAdd(self, date_value, interval_type, number):
    """Add time interval to date"""
    try:
        # Convert EasyLanguage date format (YYYYMMDD) to datetime
        date_str = str(int(date_value))
        if len(date_str) == 7:  # Handle 1YYMMDD format
            date_str = '0' + date_str
        
        year = int(date_str[:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        
        dt = datetime.date(year, month, day)
        
        # Add interval
        interval_type = str(interval_type).lower()
        number = int(number)
        
        if interval_type in ['day', 'days', 'd']:
            new_date = dt + datetime.timedelta(days=number)
        elif interval_type in ['week', 'weeks', 'w']:
            new_date = dt + datetime.timedelta(weeks=number)
        elif interval_type in ['month', 'months', 'm']:
            # Approximate month addition
            new_date = dt + datetime.timedelta(days=number * 30)
        elif interval_type in ['year', 'years', 'y']:
            new_date = dt + datetime.timedelta(days=number * 365)
        else:
            new_date = dt + datetime.timedelta(days=number)
        
        # Convert back to EasyLanguage format
        return int(f"{new_date.year:04d}{new_date.month:02d}{new_date.day:02d}")
        
    except Exception as e:
        self._handle_error("DateAdd", e)
        return int(date_value)


# Function: DateDiff
# Source: easylanguage_ultime2.py
def DateDiff(self, date1, date2, interval_type='days'):
    """Calculate difference between two dates"""
    try:
        def parse_date(date_val):
            date_str = str(int(date_val))
            if len(date_str) == 7:
                date_str = '0' + date_str
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            return datetime.date(year, month, day)
        
        dt1 = parse_date(date1)
        dt2 = parse_date(date2)
        
        diff = dt2 - dt1
        
        interval_type = str(interval_type).lower()
        if interval_type in ['day', 'days', 'd']:
            return diff.days
        elif interval_type in ['week', 'weeks', 'w']:
            return diff.days // 7
        elif interval_type in ['month', 'months', 'm']:
            return diff.days // 30
        elif interval_type in ['year', 'years', 'y']:
            return diff.days // 365
        else:
            return diff.days
        
    except Exception as e:
        self._handle_error("DateDiff", e)
        return 0


# Function: TimeAdd
# Source: easylanguage_ultime2.py
def TimeAdd(self, time_value, minutes):
    """Add minutes to time"""
    try:
        time_val = int(time_value)
        hours = time_val // 100
        mins = time_val % 100
        
        total_minutes = hours * 60 + mins + int(minutes)
        
        new_hours = (total_minutes // 60) % 24
        new_mins = total_minutes % 60
        
        return new_hours * 100 + new_mins
        
    except Exception as e:
        self._handle_error("TimeAdd", e)
        return int(time_value)


# Function: TimeDiff
# Source: easylanguage_ultime2.py
def TimeDiff(self, time1, time2):
    """Calculate difference between two times in minutes"""
    try:
        def time_to_minutes(time_val):
            time_val = int(time_val)
            hours = time_val // 100
            mins = time_val % 100
            return hours * 60 + mins
        
        mins1 = time_to_minutes(time1)
        mins2 = time_to_minutes(time2)
        
        return mins2 - mins1
        
    except Exception as e:
        self._handle_error("TimeDiff", e)
        return 0


# Function: FormatTime
# Source: easylanguage_ultime2.py
def FormatTime(self, time_value, format_string="%H:%M"):
    """Format time with custom format string"""
    try:
        time_val = int(time_value)
        hours = time_val // 100
        minutes = time_val % 100
        
        dt = datetime.time(hours, minutes)
        return dt.strftime(format_string)
        
    except Exception as e:
        self._handle_error("FormatTime", e)
        return str(time_value)


# Function: _handle_error
# Source: easylanguage_ultime2.py
def _handle_error(self, function_name, error):
    """Handle errors in functions"""
    if not hasattr(self, '_function_errors'):
        self._function_errors = {}
    
    error_msg = f"Error in {function_name}: {str(error)}"
    self._function_errors[function_name] = error_msg
    
    if hasattr(self, 'debug_mode') and self.debug_mode:
        print(f"[ERROR] {error_msg}")


# Function: _optimize_on_data
# Source: easylanguage_ultime2.py
def _optimize_on_data(self, data, strategy_func):
    """Helper function for optimization"""
    try:
        # Simple optimization - return default parameters
        # In real implementation, this would run parameter optimization
        return {'param1': 10, 'param2': 20}
        
    except Exception as e:
        self._handle_error("_optimize_on_data", e)
        return {}


# Function: __init_missing_functions__
# Source: easylanguage_ultime2.py
def __init_missing_functions__(self):
    """Initialize all missing function systems"""
    # Initialize optimization support
    self.optimization_mode = OptimizationMode.NONE
    self.optimization_results = []
    
    # Initialize performance tracking
    self._performance_metrics = {
        'trades': [],
        'equity_curve': [],
        'initial_capital': 100000.0,
        'current_equity': 100000.0
    }
    
    # Initialize portfolio management
    self._portfolio_manager = {
        'enabled': False,
        'shares_per_dollar': 0.01,
        'dollar_risk': 1000.0,
        'max_shares': 1000,
        'pct_equity': 0.1
    }
    
    # Initialize trading costs
    self._trading_costs = {
        'slippage': 0.0,
        'commission': 0.0
    }
    
    # Initialize arrays
    self._dynamic_arrays = {}
    
    # Initialize error tracking
    self._function_errors = {}
    
    print("✓ Missing EasyLanguage Functions initialized")


# Function: optimize_recursive
# Source: easylanguage_ultime2.py
        def optimize_recursive(remaining_ranges, current_params):
            nonlocal best_result, best_value, iteration
            
            if iteration >= max_iterations:
                return
                
            if not remaining_ranges:
                # Evaluate current parameter combination
                result = objective_function(current_params)
                if result > best_value:
                    best_value = result
                    best_result = current_params.copy()
                iteration += 1
                return
            
            param_name, param_range = next(iter(remaining_ranges.items()))
            remaining = {k: v for k, v in remaining_ranges.items() if k != param_name}
            
            for value in param_range:
                current_params[param_name] = value
                optimize_recursive(remaining, current_params)


# Function: parse_date
# Source: easylanguage_ultime2.py
        def parse_date(date_val):
            date_str = str(int(date_val))
            if len(date_str) == 7:
                date_str = '0' + date_str
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            return datetime.date(year, month, day)


# Function: time_to_minutes
# Source: easylanguage_ultime2.py
        def time_to_minutes(time_val):
            time_val = int(time_val)
            hours = time_val // 100
            mins = time_val % 100
            return hours * 60 + mins


# Function: compile_enhanced_plot_system
# Source: easylanguage_ultime_features_claude.py
    def compile_enhanced_plot_system(self):
        code = '''
    # ============== ENHANCED PLOT SYSTEM ==============
    
    def _init_enhanced_plot_system(self):
        """Inizializza sistema plot avanzato"""
        self._plots = {}
        self._plot_colors = {}
        self._plot_styles = {}
        self._plot_widths = {}
        self._plot_names = {}
        self._noplot_flags = {}
        self._plot_displacement = {}
        
    def Plot1(self, value, plot_name="Plot1", color=None, bg_color=None, width=None, displacement=0):
        """Plot1 completo con tutte le opzioni EasyLanguage"""
        self._plots[1] = value
        self._plot_names[1] = plot_name
        
        if displacement != 0:
            self._plot_displacement[1] = displacement
        
        # Gestione colori
        if color is not None:
            self._plot_colors[1] = color
        if bg_color is not None:
            self._plot_bg_colors[1] = bg_color
        if width is not None:
            self._plot_widths[1] = width
            
        # Output effettivo
        self.log(f"PLOT1: {plot_name} = {value:.4f}")
        return value
    
    def Plot2(self, value, plot_name="Plot2", color=None, bg_color=None, width=None):
        """Plot2 con opzioni complete"""
        self._plots[2] = value
        self._plot_names[2] = plot_name
        if color: self._plot_colors[2] = color
        if bg_color: self._plot_bg_colors[2] = bg_color
        if width: self._plot_widths[2] = width
        self.log(f"PLOT2: {plot_name} = {value:.4f}")
        return value
    
    def Plot3(self, value, plot_name="Plot3", color=None, bg_color=None, width=None):
        """Plot3 con opzioni complete"""
        self._plots[3] = value
        self._plot_names[3] = plot_name
        if color: self._plot_colors[3] = color
        if bg_color: self._plot_bg_colors[3] = bg_color  
        if width: self._plot_widths[3] = width
        self.log(f"PLOT3: {plot_name} = {value:.4f}")
        return value
    
    # Continua fino a Plot99...
    
    def SetPlotColor(self, plot_number, color):
        """Imposta colore dinamico del plot"""
        self._plot_colors[plot_number] = color
        return True
    
    def SetPlotWidth(self, plot_number, width):
        """Imposta spessore dinamico del plot"""
        self._plot_widths[plot_number] = width
        return True
    
    def SetPlotBGColor(self, plot_number, bg_color):
        """Imposta colore sfondo plot (RadarScreen/OptionStation)"""
        if not hasattr(self, '_plot_bg_colors'):
            self._plot_bg_colors = {}
        self._plot_bg_colors[plot_number] = bg_color
        return True
    
    def NoPlot(self, plot_number):
        """Rimuove plot dalla barra corrente"""
        if plot_number in self._plots:
            del self._plots[plot_number]
        self._noplot_flags[plot_number] = True
        self.log(f"NOPLOT: Plot{plot_number} removed from current bar")
        return True
    
    def PlotPB(self, high_price, low_price, plot_name="PaintBar", color=None):
        """PlotPB per PaintBar studies"""
        paintbar_id = f"pb_{len(getattr(self, '_paintbars', {}))}"
        
        if not hasattr(self, '_paintbars'):
            self._paintbars = {}
            
        self._paintbars[paintbar_id] = {
            'high': high_price,
            'low': low_price,
            'name': plot_name,
            'color': color or 'Yellow',
            'bar': len(self.data)
        }
        
        self.log(f"PAINTBAR: {plot_name} from {low_price:.4f} to {high_price:.4f}")
        return paintbar_id
    
    def GetPlotValue(self, plot_number, bars_back=0):
        """Ottiene valore storico di un plot"""
        # In un'implementazione reale, dovremmo mantenere storia
        if plot_number in self._plots:
            return self._plots[plot_number]
        return 0
    
    def IsPlotVisible(self, plot_number):
        """Verifica se un plot è visibile"""
        return plot_number not in self._noplot_flags
'''
        return code


# Function: compile_color_system
# Source: easylanguage_ultime_features_claude.py
    def compile_color_system(self):
        code = '''
    # ============== 16 MILLION COLOR SYSTEM ==============
    
    def _init_color_system(self):
        """Inizializza sistema colori avanzato"""
        self._legacy_colors = {
            'Black': 1, 'Blue': 2, 'Cyan': 3, 'Green': 4,
            'Magenta': 5, 'Red': 6, 'Yellow': 7, 'White': 8,
            'DarkBlue': 9, 'DarkCyan': 10, 'DarkGreen': 11,
            'DarkMagenta': 12, 'DarkRed': 13, 'DarkBrown': 14,
            'DarkGray': 15, 'LightGray': 16
        }
        
        # RGB Values per colori standard
        self._rgb_colors = {
            'Black': 0, 'Blue': 16711680, 'Cyan': 16776960,
            'Green': 65280, 'Magenta': 16711935, 'Red': 255,
            'Yellow': 65535, 'White': 16777215, 'DarkBlue': 8388608,
            'DarkCyan': 8421376, 'DarkGreen': 32768, 'DarkMagenta': 8388736,
            'DarkRed': 128, 'DarkBrown': 32896, 'DarkGray': 8421504,
            'LightGray': 12632256
        }
        
        self._legacy_color_mode = False  # Default RGB mode
    
    def RGB(self, red, green, blue):
        """Crea colore RGB (16 milioni di colori)"""
        # Assicura valori validi 0-255
        red = max(0, min(255, int(red)))
        green = max(0, min(255, int(green)))
        blue = max(0, min(255, int(blue)))
        
        # Formula RGB standard
        rgb_value = (red << 16) + (green << 8) + blue
        return rgb_value
    
    def GradientColor(self, value, min_value, max_value, start_color, end_color):
        """Crea gradiente colore basato su valore"""
        if max_value == min_value:
            return start_color
        
        # Normalizza valore 0-1
        ratio = max(0, min(1, (value - min_value) / (max_value - min_value)))
        
        # Se sono colori RGB
        if isinstance(start_color, int) and isinstance(end_color, int):
            start_r = (start_color >> 16) & 0xFF
            start_g = (start_color >> 8) & 0xFF
            start_b = start_color & 0xFF
            
            end_r = (end_color >> 16) & 0xFF
            end_g = (end_color >> 8) & 0xFF
            end_b = end_color & 0xFF
            
            new_r = int(start_r + (end_r - start_r) * ratio)
            new_g = int(start_g + (end_g - start_g) * ratio)
            new_b = int(start_b + (end_b - start_b) * ratio)
            
            return self.RGB(new_r, new_g, new_b)
        
        # Fallback per colori named
        return end_color if ratio > 0.5 else start_color
    
    def ColorToRGB(self, color):
        """Converte colore in valore RGB"""
        if isinstance(color, str):
            return self._rgb_colors.get(color, 0)
        return color
    
    def SetLegacyColorMode(self, enabled):
        """Abilita/disabilita modalità colori legacy"""
        self._legacy_color_mode = enabled
        
    def GetColorName(self, rgb_value):
        """Ottiene nome colore da valore RGB (se esiste)"""
        for name, value in self._rgb_colors.items():
            if value == rgb_value:
                return name
        return f"RGB({(rgb_value >> 16) & 0xFF},{(rgb_value >> 8) & 0xFF},{rgb_value & 0xFF})"
'''
        return code


# Function: compile_quote_fields_system
# Source: easylanguage_ultime_features_claude.py
    def compile_quote_fields_system(self):
        code = '''
    # ============== QUOTE FIELDS ACCESS ==============
    
    def _init_quote_fields(self):
        """Inizializza sistema quote fields"""
        self._quote_cache = {}
        self._last_quote_update = 0
        self._quote_fields = {
            'InsideBid': 0.0, 'InsideAsk': 0.0, 'LastPrice': 0.0,
            'High52Wk': 0.0, 'Low52Wk': 0.0, 'VWAP': 0.0,
            'DayHigh': 0.0, 'DayLow': 0.0, 'DayOpen': 0.0,
            'PrevClose': 0.0, 'NetChange': 0.0, 'PercentChange': 0.0,
            'TotalVolume': 0, 'BidSize': 0, 'AskSize': 0
        }
    
    def _update_quote_fields(self):
        """Aggiorna quote fields con dati attuali"""
        try:
            if hasattr(self, 'data') and len(self.data) > 0:
                current_price = self.dataclose[0]
                
                # Simula bid/ask spread (in produzione verrebbe da feed reale)
                spread = current_price * 0.0001  # 1 basis point
                self._quote_fields['InsideBid'] = current_price - spread
                self._quote_fields['InsideAsk'] = current_price + spread
                self._quote_fields['LastPrice'] = current_price
                
                # High/Low giornalieri
                self._quote_fields['DayHigh'] = self.datahigh[0]
                self._quote_fields['DayLow'] = self.datalow[0]
                self._quote_fields['DayOpen'] = self.dataopen[0]
                
                # Volume
                self._quote_fields['TotalVolume'] = self.datavolume[0]
                
                # 52 week high/low (simulati)
                if len(self.data) > 252:  # ~1 anno di dati
                    highs = [self.datahigh[-i] for i in range(252)]
                    lows = [self.datalow[-i] for i in range(252)]
                    self._quote_fields['High52Wk'] = max(highs)
                    self._quote_fields['Low52Wk'] = min(lows)
                else:
                    self._quote_fields['High52Wk'] = current_price * 1.2
                    self._quote_fields['Low52Wk'] = current_price * 0.8
                
                # Net Change
                if len(self.data) > 1:
                    prev_close = self.dataclose[-1]
                    self._quote_fields['PrevClose'] = prev_close
                    self._quote_fields['NetChange'] = current_price - prev_close
                    self._quote_fields['PercentChange'] = ((current_price - prev_close) / prev_close) * 100 if prev_close != 0 else 0
                
                self._last_quote_update = len(self.data)
        except:
            pass
    
    def InsideBid(self):
        """Current inside bid price"""
        if self.LastBarOnChart():
            self._update_quote_fields()
            return self._quote_fields['InsideBid']
        return 0.0
    
    def InsideAsk(self):
        """Current inside ask price"""
        if self.LastBarOnChart():
            self._update_quote_fields()
            return self._quote_fields['InsideAsk']
        return 0.0
    
    def LastPrice(self):
        """Last traded price"""
        return self.dataclose[0]
    
    def High52Wk(self):
        """52-week high"""
        if self.LastBarOnChart():
            self._update_quote_fields()
            return self._quote_fields['High52Wk']
        return 0.0
    
    def Low52Wk(self):
        """52-week low"""
        if self.LastBarOnChart():
            self._update_quote_fields()
            return self._quote_fields['Low52Wk']
        return 0.0
    
    def VWAP_Quote(self):
        """Volume Weighted Average Price (quote field)"""
        if self.LastBarOnChart():
            return self.VWAP()  # Usa implementazione esistente
        return 0.0
    
    def DayHigh(self):
        """Day's high price"""
        return self.datahigh[0]
    
    def DayLow(self):
        """Day's low price"""
        return self.datalow[0]
    
    def DayOpen(self):
        """Day's opening price"""
        return self.dataopen[0]
    
    def PrevClose(self):
        """Previous day's closing price"""
        if len(self.data) > 1:
            return self.dataclose[-1]
        return self.dataclose[0]
    
    def NetChange(self):
        """Net change from previous close"""
        if self.LastBarOnChart():
            self._update_quote_fields()
            return self._quote_fields['NetChange']
        return 0.0
    
    def PercentChange(self):
        """Percent change from previous close"""
        if self.LastBarOnChart():
            self._update_quote_fields()
            return self._quote_fields['PercentChange']
        return 0.0
    
    def TotalVolume(self):
        """Total day's volume"""
        return self.datavolume[0]
    
    def BidSize(self):
        """Current bid size"""
        return 100  # Simulato - in produzione da feed reale
    
    def AskSize(self):
        """Current ask size"""  
        return 100  # Simulato - in produzione da feed reale
'''
        return code


# Function: compile_fundamental_data_system
# Source: easylanguage_ultime_features_claude.py
    def compile_fundamental_data_system(self):
        code = '''
    # ============== FUNDAMENTAL DATA ACCESS ==============
    
    def _init_fundamental_data(self):
        """Inizializza sistema dati fondamentali"""
        self._fundamental_cache = {}
        self._fundamental_fields = {
            # Snapshot fields
            'BETA': 1.0, 'QNI': 0.0, 'MKTCAP': 0.0, 'PE': 15.0,
            'EPS': 1.0, 'YIELD': 2.5, 'ROE': 12.0, 'ROA': 8.0,
            
            # Historical fields  
            'SCSI': 1000000.0,  # Cash and equivalents
            'ATOT': 5000000.0,  # Total assets
            'LTLL': 2000000.0,  # Total liabilities
            'REV': 10000000.0,  # Revenue
            'NINC': 1000000.0,  # Net income
        }
        
    def GetFundData(self, field_name, period=0):
        """
        Ottiene dato fondamentale
        field_name: Nome del campo (es. "BETA", "PE", "EPS")
        period: Periodo (0=corrente, 1=precedente, ecc.)
        """
        field_upper = field_name.upper()
        
        # Per semplicità, restituisce valore simulato
        # In produzione si collegherebbe a provider dati fondamentali
        base_value = self._fundamental_fields.get(field_upper, 0.0)
        
        # Simula valori storici con variazione
        if period > 0:
            import random
            variation = 1 + (random.random() - 0.5) * 0.1  # +/- 5%
            return base_value * variation
            
        return base_value
    
    def GetFundPostDate(self, field_name, period=0):
        """Data di pubblicazione del dato fondamentale"""
        # Simula date di pubblicazione trimestrali
        import datetime
        base_date = datetime.datetime.now()
        
        # Sottrai trimestri per periodi precedenti
        months_back = period * 3
        if months_back > 0:
            year = base_date.year
            month = base_date.month - months_back
            while month <= 0:
                month += 12
                year -= 1
            base_date = base_date.replace(year=year, month=month)
        
        # Formato EasyLanguage YYYYMMDD
        return int(base_date.strftime('%Y%m%d'))
    
    def GetFundPeriodEndDate(self, field_name, period=0):
        """Data di fine periodo del dato fondamentale"""
        # Simula fine trimestri
        post_date = self.GetFundPostDate(field_name, period)
        
        # Fine trimestre è tipicamente 3 mesi prima della pubblicazione
        year = post_date // 10000
        month = (post_date // 100) % 100
        
        month -= 3
        if month <= 0:
            month += 12
            year -= 1
        
        # Ultimo giorno del trimestre
        if month in [3, 6, 9, 12]:
            day = 31 if month == 3 else 30
        else:
            day = 28  # Semplificato
            
        return int(f"{year}{month:02d}{day:02d}")
    
    def FundValue(self, field_name, period=0):
        """Alias per GetFundData"""
        return self.GetFundData(field_name, period)
    
    def FundDate(self, field_name, period=0):
        """Alias per GetFundPostDate"""
        return self.GetFundPostDate(field_name, period)
    
    def FundPeriodEndDate(self, field_name, period=0):
        """Alias per GetFundPeriodEndDate"""
        return self.GetFundPeriodEndDate(field_name, period)
    
    # Campi fondamentali comuni
    def Beta(self, period=0):
        """Beta coefficient"""
        return self.GetFundData("BETA", period)
    
    def PE_Ratio(self, period=0):
        """Price/Earnings ratio"""
        return self.GetFundData("PE", period)
    
    def EPS(self, period=0):
        """Earnings Per Share"""
        return self.GetFundData("EPS", period)
    
    def MarketCap(self, period=0):
        """Market Capitalization"""
        return self.GetFundData("MKTCAP", period)
    
    def DividendYield(self, period=0):
        """Dividend Yield"""
        return self.GetFundData("YIELD", period)
    
    def ROE(self, period=0):
        """Return on Equity"""
        return self.GetFundData("ROE", period)
    
    def ROA(self, period=0):
        """Return on Assets"""
        return self.GetFundData("ROA", period)
    
    def TotalAssets(self, period=0):
        """Total Assets"""
        return self.GetFundData("ATOT", period)
    
    def TotalLiabilities(self, period=0):
        """Total Liabilities"""
        return self.GetFundData("LTLL", period)
    
    def CashAndEquivalents(self, period=0):
        """Cash and Equivalents"""
        return self.GetFundData("SCSI", period)
    
    def Revenue(self, period=0):
        """Revenue"""
        return self.GetFundData("REV", period)
    
    def NetIncome(self, period=0):
        """Net Income"""
        return self.GetFundData("NINC", period)
'''
        return code


# Function: compile_multimedia_commentary_system
# Source: easylanguage_ultime_features_claude.py
    def compile_multimedia_commentary_system(self):
        code = '''
    # ============== MULTIMEDIA & COMMENTARY SYSTEM ==============
    
    def _init_multimedia_commentary(self):
        """Inizializza sistema multimedia e commentary"""
        self._sound_library = {
            'alert.wav': 'default_alert',
            'bell.wav': 'bell_sound',
            'chime.wav': 'chime_sound'
        }
        self._commentary_buffer = []
        self._commentary_enabled = True
        
    def PlaySound(self, sound_file="alert.wav"):
        """
        Riproduce un suono
        sound_file: Nome del file audio da riprodurre
        """
        if sound_file in self._sound_library:
            sound_name = self._sound_library[sound_file]
            self.log(f"SOUND: Playing {sound_name} ({sound_file})")
            
            # In produzione, qui si implementerebbe riproduzione reale
            # import pygame, winsound, playsound, etc.
            try:
                # Placeholder per riproduzione audio reale
                print(f"🔊 Playing: {sound_file}")
                return True
            except Exception as e:
                self.log(f"Sound error: {e}")
                return False
        else:
            self.log(f"Sound file not found: {sound_file}")
            return False
    
    def PlayMovie(self, movie_file, x_pos=100, y_pos=100, width=320, height=240):
        """
        Riproduce un video (placeholder)
        In EasyLanguage originale supporta AVI, WMV, etc.
        """
        self.log(f"MOVIE: Would play {movie_file} at ({x_pos},{y_pos}) size {width}x{height}")
        
        # Placeholder - in produzione si userebbe cv2, pygame, tkinter, etc.
        try:
            print(f"🎬 Playing movie: {movie_file}")
            return True
        except Exception as e:
            self.log(f"Movie error: {e}")
            return False
    
    def Commentary(self, *args):
        """
        Sistema Commentary per Analysis Commentary window
        Accetta lista di argomenti come Print
        """
        if not self._commentary_enabled:
            return
        
        # Converte argumenti in stringa
        comment_parts = []
        for arg in args:
            if hasattr(arg, '__call__'):
                # Se è una funzione, la chiama
                try:
                    arg_value = arg()
                    comment_parts.append(str(arg_value))
                except:
                    comment_parts.append(str(arg))
            else:
                comment_parts.append(str(arg))
        
        comment_text = ' '.join(comment_parts)
        
        # Aggiunge a buffer commentary
        commentary_entry = {
            'bar': len(self.data),
            'date': self.Date(),
            'time': self.Time(),
            'text': comment_text
        }
        
        self._commentary_buffer.append(commentary_entry)
        
        # Mantieni solo ultime 1000 entries
        if len(self._commentary_buffer) > 1000:
            self._commentary_buffer = self._commentary_buffer[-1000:]
        
        # Log per debug
        self.log(f"COMMENTARY: {comment_text}")
    
    def CommentaryHTML(self, html_content):
        """
        Commentary con contenuto HTML
        EasyLanguage supporta HTML nel commentary
        """
        html_entry = {
            'bar': len(self.data),
            'date': self.Date(),
            'time': self.Time(),
            'html': html_content,
            'type': 'html'
        }
        
        self._commentary_buffer.append(html_entry)
        self.log(f"HTML COMMENTARY: {html_content[:50]}...")
    
    def AtCommentaryBar(self):
        """
        Verifica se Commentary window è attiva per questa barra
        In EasyLanguage originale è True quando si clicca su barra per Commentary
        """
        # Simulazione - in produzione sarebbe controllato da UI
        return False
    
    def CommentaryEnabled(self):
        """Verifica se Commentary è abilitato"""
        return self._commentary_enabled
    
    def EnableCommentary(self, enabled=True):
        """Abilita/disabilita Commentary"""
        self._commentary_enabled = enabled
        return True
    
    def ClearCommentary(self):
        """Pulisce buffer Commentary"""
        self._commentary_buffer.clear()
        return True
    
    def GetCommentaryCount(self):
        """Ottiene numero entry Commentary"""
        return len(self._commentary_buffer)
    
    def GetCommentaryEntry(self, index):
        """Ottiene entry Commentary specifica"""
        if 0 <= index < len(self._commentary_buffer):
            return self._commentary_buffer[index]
        return None
    
    def GetCommentaryForBar(self, bar_number):
        """Ottiene Commentary per barra specifica"""
        entries = []
        for entry in self._commentary_buffer:
            if entry['bar'] == bar_number:
                entries.append(entry)
        return entries
    
    # Funzioni HTML helper per Commentary
    def HTML_Bold(self, text):
        """Testo grassetto HTML"""
        return f"<b>{text}</b>"
    
    def HTML_Italic(self, text):
        """Testo corsivo HTML"""
        return f"<i>{text}</i>"
    
    def HTML_Color(self, text, color):
        """Testo colorato HTML"""
        return f'<font color="{color}">{text}</font>'
    
    def HTML_Table(self, rows):
        """Tabella HTML"""
        table_html = "<table border='1'>"
        for row in rows:
            table_html += "<tr>"
            for cell in row:
                table_html += f"<td>{cell}</td>"
            table_html += "</tr>"
        table_html += "</table>"
        return table_html
    
    def HTML_Link(self, text, url):
        """Link HTML"""
        return f'<a href="{url}">{text}</a>'
    
    def HTML_Image(self, image_path, width=None, height=None):
        """Immagine HTML"""
        size_attr = ""
        if width: size_attr += f' width="{width}"'
        if height: size_attr += f' height="{height}"'
        return f'<img src="{image_path}"{size_attr}>'
    
    def NewLine(self):
        """New line per Commentary e Print"""
        return "\\n"
'''
        return code


# Function: compile_advanced_stop_system
# Source: easylanguage_ultime_features_claude.py
    def compile_advanced_stop_system(self):
        code = '''
    # ============== ADVANCED STOP MANAGEMENT ==============
    
    def _init_advanced_stops(self):
        """Inizializza sistema stop avanzato"""
        self._active_stops = {}
        self._stop_counter = 0
        self._bracket_orders = {}
        self._trailing_stops = {}
        self._conditional_stops = {}
        
    def SetStopContract(self, contracts=None):
        """Imposta numero contratti per stop"""
        if contracts is None:
            contracts = self.CurrentContracts()
        
        self._stop_contracts = contracts
        self.log(f"STOP CONTRACTS: {contracts}")
        return True
    
    def SetStopShare(self, shares=None):
        """Imposta numero azioni per stop"""
        return self.SetStopContract(shares)
    
    def SetExitOnClose(self, enabled=True):
        """Exit all positions on close"""
        self.options['exit_on_close'] = enabled
        self.log(f"EXIT ON CLOSE: {enabled}")
        return True
    
    def SetExitOnClosePosition(self, position_type="All"):
        """Exit specific position type on close"""
        self.options['exit_on_close_position'] = position_type
        self.log(f"EXIT ON CLOSE POSITION: {position_type}")
        return True
    
    def SetStopLossPosition(self, amount, position_type="All"):
        """Stop loss per tipo posizione specifico"""
        stop_id = f"stop_loss_{self._stop_counter}"
        self._stop_counter += 1
        
        self._active_stops[stop_id] = {
            'type': 'stop_loss_position',
            'amount': amount,
            'position_type': position_type,
            'active': True
        }
        
        self.log(f"STOP LOSS POSITION: {amount} for {position_type}")
        return stop_id
    
    def SetProfitTargetPosition(self, amount, position_type="All"):
        """Profit target per tipo posizione specifico"""
        stop_id = f"profit_target_{self._stop_counter}"
        self._stop_counter += 1
        
        self._active_stops[stop_id] = {
            'type': 'profit_target_position', 
            'amount': amount,
            'position_type': position_type,
            'active': True
        }
        
        self.log(f"PROFIT TARGET POSITION: {amount} for {position_type}")
        return stop_id
    
    def SetBreakEvenStop(self, profit_threshold=0):
        """Break even stop dopo profitto soglia"""
        stop_id = f"breakeven_{self._stop_counter}"
        self._stop_counter += 1
        
        self._active_stops[stop_id] = {
            'type': 'break_even',
            'profit_threshold': profit_threshold,
            'active': True,
            'triggered': False
        }
        
        self.log(f"BREAK EVEN STOP: Threshold {profit_threshold}")
        return stop_id
    
    def SetPercentTrailingStop(self, percent, floor_amount=0):
        """Trailing stop percentuale con floor"""
        stop_id = f"trail_pct_{self._stop_counter}"
        self._stop_counter += 1
        
        self._trailing_stops[stop_id] = {
            'type': 'percent_trailing',
            'percent': percent,
            'floor_amount': floor_amount,
            'high_water_mark': None,
            'low_water_mark': None,
            'active': True
        }
        
        self.log(f"PERCENT TRAILING STOP: {percent}% (Floor: {floor_amount})")
        return stop_id
    
    def SetDollarTrailingStop(self, amount, floor_amount=0):
        """Trailing stop in dollari con floor"""
        stop_id = f"trail_dollar_{self._stop_counter}"
        self._stop_counter += 1
        
        self._trailing_stops[stop_id] = {
            'type': 'dollar_trailing',
            'amount': amount,
            'floor_amount': floor_amount,
            'high_water_mark': None,
            'low_water_mark': None,
            'active': True
        }
        
        self.log(f"DOLLAR TRAILING STOP: ${amount} (Floor: ${floor_amount})")
        return stop_id
    
    def SetTimedExit(self, bars_since_entry=10, exit_type="Market"):
        """Exit dopo numero specificato di barre"""
        stop_id = f"timed_exit_{self._stop_counter}"
        self._stop_counter += 1
        
        self._conditional_stops[stop_id] = {
            'type': 'timed_exit',
            'bars_threshold': bars_since_entry,
            'exit_type': exit_type,
            'active': True
        }
        
        self.log(f"TIMED EXIT: After {bars_since_entry} bars ({exit_type})")
        return stop_id
    
    def SetMAEStop(self, mae_threshold):
        """Maximum Adverse Excursion stop"""
        stop_id = f"mae_stop_{self._stop_counter}"
        self._stop_counter += 1
        
        self._conditional_stops[stop_id] = {
            'type': 'mae_stop',
            'threshold': mae_threshold,
            'max_adverse': 0,
            'active': True
        }
        
        self.log(f"MAE STOP: Threshold ${mae_threshold}")
        return stop_id
    
    def SetMFEStop(self, mfe_threshold, retrace_percent=50):
        """Maximum Favorable Excursion stop"""
        stop_id = f"mfe_stop_{self._stop_counter}"
        self._stop_counter += 1
        
        self._conditional_stops[stop_id] = {
            'type': 'mfe_stop',
            'threshold': mfe_threshold,
            'retrace_percent': retrace_percent,
            'max_favorable': 0,
            'active': True
        }
        
        self.log(f"MFE STOP: Threshold ${mfe_threshold}, Retrace {retrace_percent}%")
        return stop_id
    
    def _update_trailing_stops(self):
        """Aggiorna trailing stops attivi"""
        if self.market_position == 0:
            return
        
        current_price = self.dataclose[0]
        position_profit = self.OpenPositionProfit()
        
        for stop_id, stop_data in self._trailing_stops.items():
            if not stop_data['active']:
                continue
            
            if self.market_position > 0:  # Long position
                if stop_data['high_water_mark'] is None:
                    stop_data['high_water_mark'] = current_price
                else:
                    stop_data['high_water_mark'] = max(stop_data['high_water_mark'], current_price)
                
                if stop_data['type'] == 'percent_trailing':
                    trail_amount = stop_data['high_water_mark'] * (stop_data['percent'] / 100)
                    stop_price = stop_data['high_water_mark'] - trail_amount
                else:  # dollar_trailing
                    stop_price = stop_data['high_water_mark'] - stop_data['amount']
                
                # Check floor
                if stop_data['floor_amount'] > 0:
                    entry_price = self.position_entry_price
                    floor_price = entry_price + stop_data['floor_amount']
                    stop_price = max(stop_price, floor_price)
                
                # Trigger stop se prezzo scende sotto stop
                if current_price <= stop_price:
                    self.log(f"TRAILING STOP TRIGGERED: {stop_id} at {stop_price:.4f}")
                    self.close()  # Exit position
                    stop_data['active'] = False
            
            elif self.market_position < 0:  # Short position
                if stop_data['low_water_mark'] is None:
                    stop_data['low_water_mark'] = current_price
                else:
                    stop_data['low_water_mark'] = min(stop_data['low_water_mark'], current_price)
                
                if stop_data['type'] == 'percent_trailing':
                    trail_amount = stop_data['low_water_mark'] * (stop_data['percent'] / 100)
                    stop_price = stop_data['low_water_mark'] + trail_amount
                else:  # dollar_trailing
                    stop_price = stop_data['low_water_mark'] + stop_data['amount']
                
                # Check floor
                if stop_data['floor_amount'] > 0:
                    entry_price = self.position_entry_price
                    floor_price = entry_price - stop_data['floor_amount']
                    stop_price = min(stop_price, floor_price)
                
                # Trigger stop se prezzo sale sopra stop
                if current_price >= stop_price:
                    self.log(f"TRAILING STOP TRIGGERED: {stop_id} at {stop_price:.4f}")
                    self.close()  # Exit position
                    stop_data['active'] = False
    
    def _update_conditional_stops(self):
        """Aggiorna stop condizionali"""
        if self.market_position == 0:
            return
        
        for stop_id, stop_data in self._conditional_stops.items():
            if not stop_data['active']:
                continue
            
            if stop_data['type'] == 'timed_exit':
                if self.bars_since_entry >= stop_data['bars_threshold']:
                    self.log(f"TIMED EXIT TRIGGERED: {stop_id} after {self.bars_since_entry} bars")
                    if stop_data['exit_type'].lower() == 'market':
                        self.close()
                    else:
                        # Limit exit at current price
                        if self.market_position > 0:
                            self.sell(exectype=bt.Order.Limit, price=self.dataclose[0])
                        else:
                            self.buy(exectype=bt.Order.Limit, price=self.dataclose[0])
                    stop_data['active'] = False
            
            elif stop_data['type'] == 'mae_stop':
                adverse_excursion = min(0, self.open_position_profit)  # Negative values
                stop_data['max_adverse'] = min(stop_data['max_adverse'], adverse_excursion)
                
                if abs(stop_data['max_adverse']) >= stop_data['threshold']:
                    self.log(f"MAE STOP TRIGGERED: {stop_id} at MAE ${abs(stop_data['max_adverse']):.2f}")
                    self.close()
                    stop_data['active'] = False
            
            elif stop_data['type'] == 'mfe_stop':
                favorable_excursion = max(0, self.open_position_profit)  # Positive values
                stop_data['max_favorable'] = max(stop_data['max_favorable'], favorable_excursion)
                
                if (stop_data['max_favorable'] >= stop_data['threshold'] and 
                    favorable_excursion <= stop_data['max_favorable'] * (1 - stop_data['retrace_percent']/100)):
                    self.log(f"MFE STOP TRIGGERED: {stop_id} at MFE ${stop_data['max_favorable']:.2f}")
                    self.close()
                    stop_data['active'] = False
    
    def _update_break_even_stops(self):
        """Aggiorna break even stops"""
        if self.market_position == 0:
            return
        
        for stop_id, stop_data in self._active_stops.items():
            if (stop_data.get('type') == 'break_even' and 
                stop_data['active'] and not stop_data['triggered']):
                
                profit = self.open_position_profit
                
                if profit >= stop_data['profit_threshold']:
                    # Attiva break even
                    entry_price = self.position_entry_price
                    
                    if self.market_position > 0:
                        # Long: stop a entry price
                        self.log(f"BREAK EVEN ACTIVATED: {stop_id} at {entry_price:.4f}")
                        # In produzione: piazza stop order a entry_price
                    else:
                        # Short: stop a entry price  
                        self.log(f"BREAK EVEN ACTIVATED: {stop_id} at {entry_price:.4f}")
                        # In produzione: piazza stop order a entry_price
                    
                    stop_data['triggered'] = True
    
    def ClearAllStops(self):
        """Cancella tutti gli stop attivi"""
        self._active_stops.clear()
        self._trailing_stops.clear() 
        self._conditional_stops.clear()
        self.log("ALL STOPS CLEARED")
        return True
    
    def GetActiveStopsCount(self):
        """Ottiene numero stop attivi"""
        total = 0
        total += len([s for s in self._active_stops.values() if s['active']])
        total += len([s for s in self._trailing_stops.values() if s['active']])
        total += len([s for s in self._conditional_stops.values() if s['active']])
        return total
'''
        return code


# Function: compile_realtime_barstate_system
# Source: easylanguage_ultime_features_claude.py
    def compile_realtime_barstate_system(self):
        code = '''
    # ============== REAL-TIME BAR STATE MANAGEMENT ==============
    
    def _init_barstate_system(self):
        """Inizializza sistema stati barra"""
        self._bar_states = {
            'is_new_bar': False,
            'is_last_bar_on_chart': False,
            'is_real_time': False,
            'update_mode': 'bar_close',  # 'tick_by_tick' or 'bar_close'
            'intrabar_persist_vars': {},
            'current_tick_count': 0,
            'bar_tick_count': 0
        }
        
        # IntrabarPersist variables storage
        self._intrabar_storage = {}
        
    def BarStatus(self, status_type=None):
        """
        Ottiene stato della barra corrente
        0 = Historical bar
        1 = Open of new bar (real-time)
        2 = Intrabar tick (real-time)
        3 = Close of bar (real-time)
        """
        if status_type is None:
            # Default: determina automaticamente
            if self.LastBarOnChart():
                if self._bar_states['is_new_bar']:
                    return 1  # Open of new bar
                elif self._bar_states['is_real_time']:
                    return 2  # Intrabar tick
                else:
                    return 3  # Close of bar
            else:
                return 0  # Historical bar
        
        return status_type
    
    def LastBarOnChart(self):
        """Verifica se è l'ultima barra del grafico"""
        try:
            return len(self.data) >= self.data.buflen() - 1
        except:
            return True  # Assume true se non può determinare
    
    def CurrentBar(self):
        """Numero barra corrente"""
        return len(self.data)
    
    def MaxBarsBack(self):
        """Massimo numero barre indietro disponibili"""
        return getattr(self, '_max_bars_back', 50)
    
    def SetMaxBarsBack(self, bars):
        """Imposta max bars back"""
        self._max_bars_back = bars
        return True
    
    def GetRealTimeStatus(self):
        """Ottiene status real-time dettagliato"""
        return {
            'is_realtime': self._bar_states['is_real_time'],
            'is_new_bar': self._bar_states['is_new_bar'],
            'is_last_bar': self.LastBarOnChart(),
            'bar_status': self.BarStatus(),
            'update_mode': self._bar_states['update_mode'],
            'tick_count': self._bar_states['current_tick_count']
        }
    
    def SetUpdateMode(self, mode='bar_close'):
        """
        Imposta modalità aggiornamento
        'bar_close': Aggiorna solo a chiusura barra
        'tick_by_tick': Aggiorna ad ogni tick
        """
        if mode in ['bar_close', 'tick_by_tick']:
            self._bar_states['update_mode'] = mode
            self.log(f"UPDATE MODE: {mode}")
            return True
        return False
    
    def IntrabarOrderGeneration(self, enabled=True):
        """Abilita/disabilita generazione ordini intra-barra"""
        self.options['intrabar_order_generation'] = enabled
        self.log(f"INTRABAR ORDER GENERATION: {enabled}")
        return True
    
    # IntrabarPersist variable system
    def SetIntrabarPersist(self, var_name, value):
        """Imposta variabile IntrabarPersist"""
        self._intrabar_storage[var_name] = value
        return True
    
    def GetIntrabarPersist(self, var_name, default_value=0):
        """Ottiene variabile IntrabarPersist"""
        return self._intrabar_storage.get(var_name, default_value)
    
    def IncrementIntrabarPersist(self, var_name, increment=1):
        """Incrementa variabile IntrabarPersist"""
        current = self.GetIntrabarPersist(var_name, 0)
        new_value = current + increment
        self.SetIntrabarPersist(var_name, new_value)
        return new_value
    
    def ResetIntrabarPersist(self, var_name=None):
        """Reset variabili IntrabarPersist"""
        if var_name:
            if var_name in self._intrabar_storage:
                del self._intrabar_storage[var_name]
        else:
            self._intrabar_storage.clear()
        return True
    
    def OnNewBar(self):
        """Callback chiamata su nuova barra"""
        self._bar_states['is_new_bar'] = True
        self._bar_states['current_tick_count'] = 0
        self._bar_states['bar_tick_count'] = 0
        
        # Reset alcune variabili intrabar se necessario
        # (alcune potrebbero persistere, altre no)
        
        return True
    
    def OnTick(self):
        """Callback chiamata su ogni tick"""
        self._bar_states['is_new_bar'] = False
        self._bar_states['current_tick_count'] += 1
        self._bar_states['bar_tick_count'] += 1
        self._bar_states['is_real_time'] = self.LastBarOnChart()
        
        return True
    
    def OnBarClose(self):
        """Callback chiamata a chiusura barra"""
        self._bar_states['is_new_bar'] = False
        self._bar_states['is_real_time'] = False
        
        # Finalizza calcoli barra
        final_tick_count = self._bar_states['bar_tick_count']
        self.log(f"BAR CLOSED: {final_tick_count} ticks processed")
        
        return True
    
    def TickCount(self):
        """Numero tick nella barra corrente"""
        return self._bar_states['bar_tick_count']
    
    def TotalTickCount(self):
        """Numero totale tick processati"""
        return self._bar_states['current_tick_count']
    
    def IsHistoricalBar(self):
        """Verifica se è barra storica"""
        return not self.LastBarOnChart()
    
    def IsRealTimeBar(self):
        """Verifica se è barra real-time"""
        return self.LastBarOnChart() and self._bar_states['is_real_time']
    
    def IsNewBar(self):
        """Verifica se è appena iniziata nuova barra"""
        return self._bar_states['is_new_bar']
    
    def GetBarAge(self):
        """Ottiene "età" barra corrente in tick"""
        return self._bar_states['bar_tick_count']
    
    def SimulateRealTimeConditions(self):
        """Simula condizioni real-time per test"""
        import random
        
        # Simula nuovo tick
        self.OnTick()
        
        # Simula nuova barra occasionalmente  
        if random.random() < 0.05:  # 5% chance
            self.OnNewBar()
        
        # Simula chiusura barra occasionalmente
        if random.random() < 0.03:  # 3% chance  
            self.OnBarClose()
        
        return self.GetRealTimeStatus()
'''
        return code


# Function: compile_user_functions_system
# Source: easylanguage_ultime_features_claude.py
    def compile_user_functions_system(self):
        code = '''
    # ============== USER DEFINED FUNCTIONS SYSTEM ==============
    
    def _init_user_functions(self):
        """Inizializza sistema funzioni utente"""
        self._user_functions = {}
        self._function_cache = {}
        self._function_call_count = {}
        
    def DefineFunction(self, func_name, func_code, input_params=None, return_type='Numeric'):
        """
        Definisce una funzione personalizzata
        func_name: Nome della funzione
        func_code: Codice della funzione (lambda o function)
        input_params: Lista parametri input
        return_type: Tipo di ritorno ('Numeric', 'TrueFalse', 'String')
        """
        if input_params is None:
            input_params = []
        
        function_def = {
            'name': func_name,
            'code': func_code,
            'input_params': input_params,
            'return_type': return_type,
            'call_count': 0,
            'last_result': None,
            'cached_results': {}
        }
        
        self._user_functions[func_name] = function_def
        self._function_call_count[func_name] = 0
        
        self.log(f"USER FUNCTION DEFINED: {func_name}({', '.join(input_params)}) -> {return_type}")
        return True
    
    def CallUserFunction(self, func_name, *args, **kwargs):
        """
        Chiama una funzione utente definita
        """
        if func_name not in self._user_functions:
            self.log(f"ERROR: User function '{func_name}' not found")
            return 0  # Default return
        
        func_def = self._user_functions[func_name]
        func_def['call_count'] += 1
        self._function_call_count[func_name] += 1
        
        try:
            # Crea cache key per risultati
            cache_key = str(args) + str(sorted(kwargs.items()))
            
            # Check cache se funzione è pure (stesso input -> stesso output)
            if cache_key in func_def['cached_results']:
                return func_def['cached_results'][cache_key]
            
            # Esegui funzione
            if callable(func_def['code']):
                result = func_def['code'](self, *args, **kwargs)
            else:
                # Se è codice stringa, valutalo in contesto safe
                local_vars = {'self': self, 'args': args, 'kwargs': kwargs}
                result = eval(func_def['code'], {'__builtins__': {}}, local_vars)
            
            # Cache risultato
            func_def['cached_results'][cache_key] = result
            func_def['last_result'] = result
            
            return result
            
        except Exception as e:
            self.log(f"ERROR in user function '{func_name}': {e}")
            return 0  # Default return on error
    
    def GetFunctionCallCount(self, func_name):
        """Ottiene numero chiamate funzione"""
        return self._function_call_count.get(func_name, 0)
    
    def ClearFunctionCache(self, func_name=None):
        """Pulisce cache funzioni"""
        if func_name:
            if func_name in self._user_functions:
                self._user_functions[func_name]['cached_results'].clear()
        else:
            for func_def in self._user_functions.values():
                func_def['cached_results'].clear()
        return True
    
    def ListUserFunctions(self):
        """Lista tutte le funzioni utente"""
        return list(self._user_functions.keys())
    
    def GetFunctionInfo(self, func_name):
        """Ottiene informazioni su funzione"""
        if func_name in self._user_functions:
            return self._user_functions[func_name].copy()
        return None
    
    def RemoveUserFunction(self, func_name):
        """Rimuove funzione utente"""
        if func_name in self._user_functions:
            del self._user_functions[func_name]
            if func_name in self._function_call_count:
                del self._function_call_count[func_name]
            self.log(f"USER FUNCTION REMOVED: {func_name}")
            return True
        return False
    
    # Esempi di funzioni utente comuni
    def DefineCommonUserFunctions(self):
        """Definisce funzioni utente comuni"""
        
        # Funzione Range personalizzata
        def custom_range(strategy, bars_back=0):
            high = strategy.datahigh[-bars_back] if bars_back > 0 else strategy.datahigh[0]
            low = strategy.datalow[-bars_back] if bars_back > 0 else strategy.datalow[0]
            return high - low
        
        self.DefineFunction('MyRange', custom_range, ['bars_back'], 'Numeric')
        
        # Funzione True Range personalizzata
        def custom_true_range(strategy, bars_back=0):
            if bars_back > 0:
                high = strategy.datahigh[-bars_back]
                low = strategy.datalow[-bars_back] 
                prev_close = strategy.dataclose[-bars_back-1] if bars_back < len(strategy.data) - 1 else high
            else:
                high = strategy.datahigh[0]
                low = strategy.datalow[0]
                prev_close = strategy.dataclose[-1] if len(strategy.data) > 1 else high
            
            return max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
        
        self.DefineFunction('MyTrueRange', custom_true_range, ['bars_back'], 'Numeric')
        
        # Funzione media personalizzata
        def custom_average(strategy, length=10, price_series='close'):
            total = 0
            for i in range(length):
                if price_series == 'close':
                    total += strategy.dataclose[-i] if i > 0 else strategy.dataclose[0]
                elif price_series == 'high':
                    total += strategy.datahigh[-i] if i > 0 else strategy.datahigh[0]
                elif price_series == 'low':
                    total += strategy.datalow[-i] if i > 0 else strategy.datalow[0]
                else:  # open
                    total += strategy.dataopen[-i] if i > 0 else strategy.dataopen[0]
            
            return total / length
        
        self.DefineFunction('MyAverage', custom_average, ['length', 'price_series'], 'Numeric')
        
        # Funzione pattern recognition
        def is_doji(strategy, tolerance=0.1):
            open_price = strategy.dataopen[0]
            close = strategy.dataclose[0]
            high = strategy.datahigh[0]
            low = strategy.datalow[0]
            
            body_size = abs(close - open_price)
            total_range = high - low
            
            return (body_size / total_range) <= tolerance if total_range > 0 else False
        
        self.DefineFunction('IsDoji', is_doji, ['tolerance'], 'TrueFalse')
        
        # Funzione volatilità personalizzata
        def custom_volatility(strategy, length=20):
            if length <= 1:
                return 0
                
            prices = []
            for i in range(length):
                prices.append(strategy.dataclose[-i] if i > 0 else strategy.dataclose[0])
            
            # Calcola standard deviation
            mean = sum(prices) / len(prices)
            variance = sum((p - mean) ** 2 for p in prices) / len(prices)
            return (variance ** 0.5)
        
        self.DefineFunction('MyVolatility', custom_volatility, ['length'], 'Numeric')
        
        self.log("COMMON USER FUNCTIONS DEFINED")
        return True
    
    # Macro system for function combinations
    def DefineFunctionMacro(self, macro_name, function_sequence):
        """
        Definisce macro che combina più funzioni
        function_sequence: Lista di (func_name, args) da eseguire in sequenza
        """
        def macro_executor(strategy, *args, **kwargs):
            results = []
            for func_name, func_args in function_sequence:
                if isinstance(func_args, (list, tuple)):
                    result = strategy.CallUserFunction(func_name, *func_args)
                else:
                    result = strategy.CallUserFunction(func_name, func_args)
                results.append(result)
            
            # Return last result or combine based on macro logic
            return results[-1] if results else 0
        
        self.DefineFunction(macro_name, macro_executor, [], 'Numeric')
        self.log(f"FUNCTION MACRO DEFINED: {macro_name}")
        return True
'''
        return code


# Function: compile_option_system
# Source: easylanguage_ultime_features_claude.py
    def compile_option_system(self):
        code = '''
    # ============== OPTION DATA & GREEKS SYSTEM ==============
    
    def _init_option_system(self):
        """Inizializza sistema dati opzioni"""
        self._option_data = {
            'implied_volatility': 0.20,  # 20% default IV
            'call_volume': 0,
            'put_volume': 0,
            'call_open_interest': 0,
            'put_open_interest': 0,
            'option_type': 'stock',  # 'call', 'put', 'stock'
            'strike_price': 0,
            'days_to_expiration': 30,
            'risk_free_rate': 0.02  # 2% default
        }
        
        self._greeks = {
            'delta': 0.5,
            'gamma': 0.1,
            'theta': -0.05,
            'vega': 0.2,
            'rho': 0.1
        }
    
    def IVolatility(self, bars_back=0):
        """Implied Volatility"""
        if self.LastBarOnChart():
            return self._option_data['implied_volatility']
        
        # Per dati storici, simula variazione IV
        base_iv = self._option_data['implied_volatility']
        if bars_back > 0:
            import random
            variation = 1 + (random.random() - 0.5) * 0.3  # +/- 15%
            return base_iv * variation
        
        return base_iv
    
    def CallVolume(self, bars_back=0):
        """Volume totale opzioni Call"""
        base_volume = self._option_data['call_volume']
        if bars_back == 0:
            # Simula volume call proporzionale al volume del sottostante
            underlying_vol = self.datavolume[0]
            return int(underlying_vol * 0.1)  # 10% del volume sottostante
        
        return base_volume
    
    def PutVolume(self, bars_back=0):
        """Volume totale opzioni Put"""
        base_volume = self._option_data['put_volume']
        if bars_back == 0:
            # Simula volume put
            underlying_vol = self.datavolume[0]
            return int(underlying_vol * 0.08)  # 8% del volume sottostante
        
        return base_volume
    
    def CallOpenInt(self, bars_back=0):
        """Open Interest totale opzioni Call"""
        if bars_back == 0:
            return self.CallVolume() * 10  # Stima OI come multiplo del volume
        return self._option_data['call_open_interest']
    
    def PutOpenInt(self, bars_back=0):
        """Open Interest totale opzioni Put"""
        if bars_back == 0:
            return self.PutVolume() * 12  # Stima OI
        return self._option_data['put_open_interest']
    
    def PutCallRatio(self):
        """Put/Call Ratio"""
        call_vol = self.CallVolume()
        put_vol = self.PutVolume()
        
        if call_vol == 0:
            return float('inf') if put_vol > 0 else 0
        
        return put_vol / call_vol
    
    def PutCallVolumeRatio(self):
        """Alias per PutCallRatio"""
        return self.PutCallRatio()
    
    def PutCallOIRatio(self):
        """Put/Call Open Interest Ratio"""
        call_oi = self.CallOpenInt()
        put_oi = self.PutOpenInt()
        
        if call_oi == 0:
            return float('inf') if put_oi > 0 else 0
        
        return put_oi / call_oi
    
    # Greeks calculations (simplified Black-Scholes)
    def CalculateGreeks(self, spot_price=None, strike=None, time_to_exp=None, 
                       risk_free_rate=None, volatility=None, option_type='call'):
        """
        Calcola Greeks usando Black-Scholes semplificato
        """
        if spot_price is None:
            spot_price = self.dataclose[0]
        if strike is None:
            strike = self._option_data['strike_price'] or spot_price
        if time_to_exp is None:
            time_to_exp = self._option_data['days_to_expiration'] / 365.0
        if risk_free_rate is None:
            risk_free_rate = self._option_data['risk_free_rate']
        if volatility is None:
            volatility = self.IVolatility()
        
        import math
        
        # Black-Scholes intermediate calculations
        d1 = (math.log(spot_price / strike) + 
              (risk_free_rate + 0.5 * volatility**2) * time_to_exp) / (volatility * math.sqrt(time_to_exp))
        d2 = d1 - volatility * math.sqrt(time_to_exp)
        
        # Standard normal CDF approximation
        def norm_cdf(x):
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))
        
        # Standard normal PDF
        def norm_pdf(x):
            return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)
        
        # Delta
        if option_type.lower() == 'call':
            delta = norm_cdf(d1)
        else:  # put
            delta = norm_cdf(d1) - 1
        
        # Gamma (same for calls and puts)
        gamma = norm_pdf(d1) / (spot_price * volatility * math.sqrt(time_to_exp))
        
        # Theta
        theta_common = -(spot_price * norm_pdf(d1) * volatility) / (2 * math.sqrt(time_to_exp))
        if option_type.lower() == 'call':
            theta = theta_common - risk_free_rate * strike * math.exp(-risk_free_rate * time_to_exp) * norm_cdf(d2)
        else:  # put
            theta = theta_common + risk_free_rate * strike * math.exp(-risk_free_rate * time_to_exp) * norm_cdf(-d2)
        
        theta /= 365  # Convert to daily theta
        
        # Vega (same for calls and puts)
        vega = spot_price * norm_pdf(d1) * math.sqrt(time_to_exp) / 100  # /100 for 1% vol change
        
        # Rho
        if option_type.lower() == 'call':
            rho = strike * time_to_exp * math.exp(-risk_free_rate * time_to_exp) * norm_cdf(d2) / 100
        else:  # put
            rho = -strike * time_to_exp * math.exp(-risk_free_rate * time_to_exp) * norm_cdf(-d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma, 
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    def Delta(self, option_type='call'):
        """Option Delta"""
        greeks = self.CalculateGreeks(option_type=option_type)
        return greeks['delta']
    
    def Gamma(self, option_type='call'):
        """Option Gamma"""
        greeks = self.CalculateGreeks(option_type=option_type)
        return greeks['gamma']
    
    def Theta(self, option_type='call'):
        """Option Theta (time decay)"""
        greeks = self.CalculateGreeks(option_type=option_type)
        return greeks['theta']
    
    def Vega(self, option_type='call'):
        """Option Vega (vol sensitivity)"""
        greeks = self.CalculateGreeks(option_type=option_type)
        return greeks['vega']
    
    def Rho(self, option_type='call'):
        """Option Rho (rate sensitivity)"""
        greeks = self.CalculateGreeks(option_type=option_type)
        return greeks['rho']
    
    def SetOptionParameters(self, strike=None, days_to_exp=None, option_type=None, iv=None):
        """Imposta parametri opzione"""
        if strike is not None:
            self._option_data['strike_price'] = strike
        if days_to_exp is not None:
            self._option_data['days_to_expiration'] = days_to_exp
        if option_type is not None:
            self._option_data['option_type'] = option_type.lower()
        if iv is not None:
            self._option_data['implied_volatility'] = iv
        
        self.log(f"OPTION PARAMS: Strike={self._option_data['strike_price']}, "
                f"DTE={self._option_data['days_to_expiration']}, "
                f"Type={self._option_data['option_type']}, "
                f"IV={self._option_data['implied_volatility']:.3f}")
        return True
    
    def DaysToExpiration(self):
        """Giorni alla scadenza"""
        return self._option_data['days_to_expiration']
    
    def TimeValue(self, option_price, option_type='call'):
        """Time Value (Valore temporale)"""
        spot = self.dataclose[0]
        strike = self._option_data['strike_price'] or spot
        
        # Intrinsic value
        if option_type.lower() == 'call':
            intrinsic = max(0, spot - strike)
        else:  # put
            intrinsic = max(0, strike - spot)
        
        # Time value = Option price - Intrinsic value
        return max(0, option_price - intrinsic)
    
    def IntrinsicValue(self, option_type='call'):
        """Intrinsic Value (Valore intrinseco)"""
        spot = self.dataclose[0]
        strike = self._option_data['strike_price'] or spot
        
        if option_type.lower() == 'call':
            return max(0, spot - strike)
        else:  # put
            return max(0, strike - spot)
    
    def IsITM(self, option_type='call'):
        """In The Money"""
        return self.IntrinsicValue(option_type) > 0
    
    def IsOTM(self, option_type='call'):
        """Out of The Money"""
        return self.IntrinsicValue(option_type) == 0 and not self.IsATM(option_type)
    
    def IsATM(self, option_type='call', tolerance=0.01):
        """At The Money"""
        spot = self.dataclose[0]
        strike = self._option_data['strike_price'] or spot
        return abs(spot - strike) / strike <= tolerance
    
    def MoneynessPct(self, option_type='call'):
        """Moneyness in percentuale"""
        spot = self.dataclose[0]
        strike = self._option_data['strike_price'] or spot
        
        if option_type.lower() == 'call':
            return (spot - strike) / strike * 100
        else:  # put
            return (strike - spot) / strike * 100
'''
        return code


# Function: compile_advanced_math_system
# Source: easylanguage_ultime_features_claude.py
    def compile_advanced_math_system(self):
        code = '''
    # ============== ADVANCED MATH & STATISTICAL FUNCTIONS ==============
    
    def _init_advanced_math(self):
        """Inizializza sistema matematico avanzato"""
        self._math_cache = {}
        self._statistical_cache = {}
        
    # Funzioni trigonometriche avanzate
    def Sine(self, angle_radians):
        """Seno (angolo in radianti)"""
        import math
        return math.sin(angle_radians)
    
    def Cosine(self, angle_radians):
        """Coseno (angolo in radianti)"""
        import math
        return math.cos(angle_radians)
    
    def Tangent(self, angle_radians):
        """Tangente (angolo in radianti)"""
        import math
        return math.tan(angle_radians)
    
    def ArcSine(self, value):
        """Arcoseno"""
        import math
        return math.asin(max(-1, min(1, value)))  # Clamp to valid range
    
    def ArcCosine(self, value):
        """Arcocoseno"""
        import math
        return math.acos(max(-1, min(1, value)))
    
    def ArcTangent(self, value):
        """Arcotangente"""
        import math
        return math.atan(value)
    
    def ArcTan2(self, y, x):
        """Arcotangente a 2 argomenti"""
        import math
        return math.atan2(y, x)
    
    def DegreesToRadians(self, degrees):
        """Converte gradi in radianti"""
        import math
        return math.radians(degrees)
    
    def RadiansToDegrees(self, radians):
        """Converte radianti in gradi"""
        import math
        return math.degrees(radians)
    
    # Funzioni esponenziali e logaritmiche
    def Exp(self, x):
        """e^x"""
        import math
        return math.exp(x)
    
    def Log(self, x, base=None):
        """Logaritmo (naturale o con base specificata)"""
        import math
        if x <= 0:
            return 0  # Evita errori
        if base is None:
            return math.log(x)  # Natural log
        else:
            return math.log(x, base)
    
    def Log10(self, x):
        """Logaritmo base 10"""
        import math
        return math.log10(x) if x > 0 else 0
    
    def Power(self, base, exponent):
        """Elevazione a potenza"""
        return pow(base, exponent)
    
    def SquareRoot(self, x):
        """Radice quadrata"""
        import math
        return math.sqrt(abs(x))
    
    def CubeRoot(self, x):
        """Radice cubica"""
        return pow(abs(x), 1.0/3) * (1 if x >= 0 else -1)
    
    def NthRoot(self, x, n):
        """Radice n-esima"""
        if n == 0:
            return 1
        return pow(abs(x), 1.0/n) * (1 if x >= 0 or n % 2 == 1 else -1)
    
    # Funzioni statistiche avanzate
    def Correlation(self, series1_name, series2_name, length=14):
        """Correlazione tra due serie"""
        # Simula correlazione tra due serie di prezzi
        # In implementazione reale, avrebbe accesso a serie multiple
        
        import random
        import math
        
        cache_key = f"corr_{series1_name}_{series2_name}_{length}"
        if cache_key in self._statistical_cache:
            return self._statistical_cache[cache_key]
        
        # Simula correlazione basata su dati storici
        if length <= 0 or len(self.data) < length:
            correlation = 0.0
        else:
            # Usa variazione prezzi per simulare correlazione
            price_changes1 = []
            price_changes2 = []  # In realtà sarebbe altra serie
            
            for i in range(1, min(length + 1, len(self.data))):
                change1 = self.dataclose[-i] - self.dataclose[-i-1] if i < len(self.data) - 1 else 0
                # Simula serie2 con correlazione parziale a serie1
                change2 = change1 * 0.7 + random.gauss(0, abs(change1) * 0.3)
                
                price_changes1.append(change1)
                price_changes2.append(change2)
            
            if len(price_changes1) < 2:
                correlation = 0.0
            else:
                # Calcola correlazione di Pearson
                n = len(price_changes1)
                sum1 = sum(price_changes1)
                sum2 = sum(price_changes2)
                sum_sq1 = sum(x * x for x in price_changes1)
                sum_sq2 = sum(x * x for x in price_changes2)
                sum_prod = sum(x * y for x, y in zip(price_changes1, price_changes2))
                
                numerator = n * sum_prod - sum1 * sum2
                denominator = math.sqrt((n * sum_sq1 - sum1**2) * (n * sum_sq2 - sum2**2))
                
                correlation = numerator / denominator if denominator != 0 else 0.0
                correlation = max(-1, min(1, correlation))  # Clamp to [-1, 1]
        
        self._statistical_cache[cache_key] = correlation
        return correlation
    
    def StandardDev(self, length=20, price_series='close'):
        """Deviazione standard"""
        if length <= 1:
            return 0
        
        prices = []
        for i in range(length):
            if price_series == 'close':
                price = self.dataclose[-i] if i > 0 else self.dataclose[0]
            elif price_series == 'high':
                price = self.datahigh[-i] if i > 0 else self.datahigh[0]
            elif price_series == 'low':
                price = self.datalow[-i] if i > 0 else self.datalow[0]
            else:  # open
                price = self.dataopen[-i] if i > 0 else self.dataopen[0]
            
            prices.append(price)
        
        if len(prices) < 2:
            return 0
        
        mean = sum(prices) / len(prices)
        variance = sum((p - mean) ** 2 for p in prices) / (len(prices) - 1)  # Sample std dev
        
        import math
        return math.sqrt(variance)
    
    def Variance(self, length=20, price_series='close'):
        """Varianza"""
        std_dev = self.StandardDev(length, price_series)
        return std_dev ** 2
    
    def Skewness(self, length=20, price_series='close'):
        """Skewness (asimmetria)"""
        if length < 3:
            return 0
        
        prices = []
        for i in range(length):
            if price_series == 'close':
                price = self.dataclose[-i] if i > 0 else self.dataclose[0]
            else:
                price = self.dataclose[-i] if i > 0 else self.dataclose[0]  # Simplified
            prices.append(price)
        
        if len(prices) < 3:
            return 0
        
        mean = sum(prices) / len(prices)
        std_dev = self.StandardDev(length, price_series)
        
        if std_dev == 0:
            return 0
        
        # Third moment
        third_moment = sum(((p - mean) / std_dev) ** 3 for p in prices) / len(prices)
        
        return third_moment
    
    def Kurtosis(self, length=20, price_series='close'):
        """Kurtosis (curtosi)"""
        if length < 4:
            return 0
        
        prices = []
        for i in range(length):
            if price_series == 'close':
                price = self.dataclose[-i] if i > 0 else self.dataclose[0]
            else:
                price = self.dataclose[-i] if i > 0 else self.dataclose[0]
            prices.append(price)
        
        if len(prices) < 4:
            return 0
        
        mean = sum(prices) / len(prices)
        std_dev = self.StandardDev(length, price_series)
        
        if std_dev == 0:
            return 0
        
        # Fourth moment
        fourth_moment = sum(((p - mean) / std_dev) ** 4 for p in prices) / len(prices)
        
        return fourth_moment - 3  # Excess kurtosis
    
    def ZScore(self, value, length=20, price_series='close'):
        """Z-Score (standardized value)"""
        mean = self.Average(price_series, length) if hasattr(self, 'Average') else value
        std_dev = self.StandardDev(length, price_series)
        
        if std_dev == 0:
            return 0
        
        return (value - mean) / std_dev
    
    def Percentile(self, percentile, length=20, price_series='close'):
        """Percentile di una serie"""
        prices = []
        for i in range(min(length, len(self.data))):
            if price_series == 'close':
                price = self.dataclose[-i] if i > 0 else self.dataclose[0]
            else:
                price = self.dataclose[-i] if i > 0 else self.dataclose[0]
            prices.append(price)
        
        if not prices:
            return 0
        
        prices.sort()
        n = len(prices)
        k = (percentile / 100.0) * (n - 1)
        
        if k == int(k):
            return prices[int(k)]
        else:
            lower = prices[int(k)]
            upper = prices[int(k) + 1] if int(k) + 1 < n else prices[int(k)]
            return lower + (k - int(k)) * (upper - lower)
    
    def Median(self, length=20, price_series='close'):
        """Mediana"""
        return self.Percentile(50, length, price_series)
    
    def Mode(self, length=20, price_series='close', precision=2):
        """Moda (valore più frequente)"""
        prices = []
        for i in range(min(length, len(self.data))):
            if price_series == 'close':
                price = self.dataclose[-i] if i > 0 else self.dataclose[0]
            else:
                price = self.dataclose[-i] if i > 0 else self.dataclose[0]
            prices.append(round(price, precision))
        
        if not prices:
            return 0
        
        # Count frequencies
        from collections import Counter
        counter = Counter(prices)
        most_common = counter.most_common(1)
        
        return most_common[0][0] if most_common else prices[-1]
    
    def LinearRegression(self, length=14, price_series='close'):
        """Linear regression (slope and intercept)"""
        if length < 2:
            return {'slope': 0, 'intercept': 0, 'r_squared': 0}
        
        prices = []
        for i in range(min(length, len(self.data))):
            if price_series == 'close':
                price = self.dataclose[-i] if i > 0 else self.dataclose[0]
            else:
                price = self.dataclose[-i] if i > 0 else self.dataclose[0]
            prices.append(price)
        
        prices.reverse()  # Chronological order
        n = len(prices)
        x_values = list(range(n))
        
        # Calculate linear regression
        sum_x = sum(x_values)
        sum_y = sum(prices)
        sum_xy = sum(x * y for x, y in zip(x_values, prices))
        sum_x2 = sum(x * x for x in x_values)
        sum_y2 = sum(y * y for y in prices)
        
        # Slope
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return {'slope': 0, 'intercept': prices[-1], 'r_squared': 0}
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        # Intercept
        intercept = (sum_y - slope * sum_x) / n
        
        # R-squared
        y_mean = sum_y / n
        ss_tot = sum((y - y_mean) ** 2 for y in prices)
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(x_values, prices))
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': max(0, min(1, r_squared)),
            'current_value': slope * (n - 1) + intercept,
            'next_value': slope * n + intercept
        }
    
    def RSquare(self, length=14, price_series='close'):
        """R-Squared coefficient"""
        regression = self.LinearRegression(length, price_series)
        return regression['r_squared']
    
    def LinearRegressionSlope(self, length=14, price_series='close'):
        """Linear regression slope"""
        regression = self.LinearRegression(length, price_series)
        return regression['slope']
    
    def LinearRegressionValue(self, length=14, price_series='close'):
        """Current linear regression value"""
        regression = self.LinearRegression(length, price_series)
        return regression['current_value']
    
    def LinearRegressionForecast(self, length=14, bars_forward=1, price_series='close'):
        """Linear regression forecast"""
        regression = self.LinearRegression(length, price_series)
        return regression['slope'] * (length - 1 + bars_forward) + regression['intercept']
    
    # Funzioni di arrotondamento avanzate
    def RoundUp(self, value, precision=0):
        """Arrotondamento per eccesso"""
        import math
        multiplier = 10 ** precision
        return math.ceil(value * multiplier) / multiplier
    
    def RoundDown(self, value, precision=0):
        """Arrotondamento per difetto"""
        import math
        multiplier = 10 ** precision
        return math.floor(value * multiplier) / multiplier
    
    def RoundToNearest(self, value, increment):
        """Arrotonda al multiplo più vicino"""
        if increment == 0:
            return value
        return round(value / increment) * increment
    
    def TruncateToTick(self, price, tick_size=None):
        """Tronca prezzo al tick size più vicino"""
        if tick_size is None:
            tick_size = self.TickSize() if hasattr(self, 'TickSize') else 0.01
        
        if tick_size == 0:
            return price
        
        return int(price / tick_size) * tick_size
    
    # Funzioni di interpolazione
    def LinearInterpolation(self, x1, y1, x2, y2, x):
        """Interpolazione lineare"""
        if x2 == x1:
            return y1
        
        return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    
    def WeightedAverage(self, values, weights=None):
        """Media ponderata"""
        if not values:
            return 0
        
        if weights is None:
            weights = [1] * len(values)
        
        if len(values) != len(weights):
            return sum(values) / len(values)  # Fallback to simple average
        
        total_weight = sum(weights)
        if total_weight == 0:
            return sum(values) / len(values)
        
        weighted_sum = sum(v * w for v, w in zip(values, weights))
        return weighted_sum / total_weight
'''
        return code
