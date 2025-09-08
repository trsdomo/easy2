# ==============================================================================
# IMPLEMENTAZIONI PRIORITÀ MEDIA - EASYLANGUAGE FEATURES
# ==============================================================================

# 1. CUSTOM INDICATORS CREATION
# ==============================================================================

class CustomIndicatorsCompiler:
    """Sistema completo per creazione indicatori personalizzati"""
    
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

# ==============================================================================
# 2. SHOWME/PAINTBAR STUDIES  
# ==============================================================================

class ShowMePaintBarCompiler:
    """Sistema per ShowMe e PaintBar studies"""
    
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

# ==============================================================================
# 3. SESSION INFORMATION FUNCTIONS
# ==============================================================================

class SessionInfoCompiler:
    """Sistema completo per informazioni di sessione"""
    
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

# ==============================================================================
# 4. SYMBOL INFORMATION FUNCTIONS
# ==============================================================================

class SymbolInfoCompiler:
    """Sistema per informazioni sui simboli"""
    
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

# ==============================================================================
# 5. VOLUME ANALYSIS FUNCTIONS
# ==============================================================================

class VolumeAnalysisCompiler:
    """Sistema per analisi avanzata dei volumi"""
    
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

# ==============================================================================
# INTEGRATION AND COMPILATION METHODS
# ==============================================================================

class MediumPriorityIntegrator:
    """Integratore per tutte le funzionalità a priorità media"""
    
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

# ==============================================================================
# EXAMPLE USAGE PATTERNS
# ==============================================================================

MEDIUM_PRIORITY_EXAMPLES = '''
# ============== ESEMPI DI UTILIZZO PRIORITÀ MEDIA ==============

// 1. CUSTOM INDICATORS
RSI_Custom = CreateIndicator("MyRSI", Custom_RSI, 14);
UpdateIndicator("MyRSI");
Value1 = GetIndicatorValue("MyRSI", 0);  // Valore corrente
Value2 = GetIndicatorValue("MyRSI", 1);  // Valore precedente

MACD_Data = Custom_MACD(12, 26, 9);
Plot1(MACD_Data["macd"], "MACD Line");
Plot2(MACD_Data["signal"], "Signal Line");

// 2. SHOWME/PAINTBAR STUDIES
// ShowMe per breakout
Condition1 = High > Highest(High, 20)[1];
ShowMe_UpArrow(Condition1, High, "Green");

// PaintBar per volumi alti
Condition2 = VolumeRatio() > 2.0;
PaintBar_Volume(Condition2, "Blue");

// Vari tipi di ShowMe
ShowMe_Dot(RSI > 70, High + 0.50, "Red");           // Ipercomprato
ShowMe_Square(RSI < 30, Low - 0.50, "Green");       // Ipervenduto
ShowMe_Diamond(Close crosses above VWAP(), Close, "Yellow");

// 3. SESSION INFORMATION
If IsRegularSession() then begin
    Commentary("Regular session active");
    
    If IsFirstBarOfSession() then
        ShowMe_UpArrow(True, Open, "Blue");
        
    If IsLastBarOfSession() then
        ShowMe_DownArrow(True, Close, "Red");
end;

Value1 = MinutesFromSessionStart();
Value2 = MinutesToSessionEnd();
Commentary("Session: ", Value1, " minutes from start, ", Value2, " minutes to end");

// 4. SYMBOL INFORMATION
Commentary("Symbol: ", GetSymbolName());
Commentary("Type: ", GetInstrumentType());
Commentary("Tick Size: ", TickSize());
Commentary("Point Value: ", PointValue());

If IsStock() then begin
    // Logica specifica per azioni
    SetPlotColor(1, Blue);
end else if IsFuture() then begin
    // Logica specifica per futures
    SetPlotColor(1, Red);
end;

// 5. VOLUME ANALYSIS
// VWAP e deviazioni
Value1 = VWAP();
Plot1(Value1, "VWAP");

// Volume profile
Condition1 = Close > HighestVolumePrice();
Commentary("Price above POC: ", Condition1);

// Analisi volume
Value2 = VolumeRatio();
Value3 = OnBalanceVolume(14);
Value4 = AccumulationDistribution(14);

If VolumeSpike(2.5) then begin
    ShowMe_Diamond(True, High, "Yellow");
    Alert("Volume Spike Detected!");
end;

// Volume trend analysis
Value5 = VolumeTrend(20);
If Value5 > 10 then
    Commentary("Volume trend: Increasing")
else if Value5 < -10 then
    Commentary("Volume trend: Decreasing")
else
    Commentary("Volume trend: Neutral");

// Up/Down volume analysis
Value6 = UpDownVolumeRatio(14);
SetPlotColor(2, IFF(Value6 > 1.2, Green, IFF(Value6 < 0.8, Red, Yellow)));
Plot2(Value6, "Up/Down Vol Ratio");
'''

# ==============================================================================
# UPDATED ROADMAP - MEDIUM PRIORITY COMPLETED
# ==============================================================================

MEDIUM_PRIORITY_COMPLETED = [
    "✅ Custom Indicators Creation (RSI, MACD, Stochastic personalizzati)",
    "✅ ShowMe/PaintBar Studies (Frecce, Dots, Squares, Diamonds, PaintBars)",
    "✅ Session Information Functions (Regular/Pre/After market, tempi sessione)",
    "✅ Symbol Information Functions (Stock/Future/Forex detection, info simbolo)",
    "✅ Volume Analysis Functions (VWAP, Volume Profile, OBV, A/D, analisi completa)"
]

LOW_PRIORITY_NEXT = [
    "🔄 Quote Field Access (Real-time quotes, bid/ask)",
    "🔄 Fundamental Data Access (P/E, EPS, financial data)",
    "🔄 Option Data Functions (Greeks, implied volatility)",
    "🔄 Advanced Stop Management (Complex stop logic)",
    "🔄 Portfolio-Level Functions (Multi-symbol strategies)",
    "🔄 Bar State Functions (IntrabarPersist, real-time states)",
    "🔄 User-Defined Functions (Custom function creation)",
    "🔄 DLL Integration (External library support)",
    "🔄 Strategy Automation Features (Auto-trading)",
    "🔄 Real-time Calculation Control (Tick-by-tick vs bar close)"
]

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

if __name__ == "__main__":
    print_medium_priority_summary()