# =============================================================================
# SYMBOL INFORMATION FUNCTIONS - EASYLANGUAGE INTEGRATION
# Complete Symbol and Instrument Information with EasyLanguage Compatibility
# =============================================================================

from dataclasses import dataclass
from typing import Optional, Dict, Any
import warnings

@dataclass
class SymbolInfo:
    """
    Complete symbol information structure
    """
    name: str = "UNKNOWN"
    description: str = ""
    exchange: str = ""
    instrument_type: str = "STOCK"
    currency: str = "USD"
    
    # Price scaling
    tick_size: float = 0.01
    min_move: float = 0.01
    price_scale: float = 1.0
    point_value: float = 1.0
    big_point_value: float = 1.0
    
    # Contract specifications
    contract_size: int = 1
    margin_required: float = 0.0
    
    # Session information
    session_start: str = "09:30"
    session_end: str = "16:00"
    timezone: str = "EST"
    
    # Market data
    last_price: float = 0.0
    bid_price: float = 0.0
    ask_price: float = 0.0
    volume: int = 0
    
    # Status
    is_tradeable: bool = True
    market_status: str = "CLOSED"

class SymbolTracker:
    """
    Tracker per informazioni simboli con integrazione dati di mercato
    """
    def __init__(self):
        # Current symbol info
        self._current_symbol = SymbolInfo()
        
        # Symbol database
        self._symbol_db = {}
        self._setup_default_symbols()
        
        # Performance counters
        self._lookups_count = 0
        self._cache_hits = 0
        
        # Error tracking
        self._error_count = 0
        self._last_error = None
        
    def _setup_default_symbols(self):
        """Setup database with common symbols and their properties"""
        
        # Major Stock Indices
        self._symbol_db["SPY"] = SymbolInfo(
            name="SPY", description="SPDR S&P 500 ETF", exchange="NYSE",
            instrument_type="ETF", tick_size=0.01, min_move=0.01
        )
        
        self._symbol_db["QQQ"] = SymbolInfo(
            name="QQQ", description="Invesco QQQ ETF", exchange="NASDAQ",
            instrument_type="ETF", tick_size=0.01, min_move=0.01
        )
        
        # Major Stocks
        self._symbol_db["AAPL"] = SymbolInfo(
            name="AAPL", description="Apple Inc", exchange="NASDAQ",
            instrument_type="STOCK", tick_size=0.01, min_move=0.01
        )
        
        self._symbol_db["MSFT"] = SymbolInfo(
            name="MSFT", description="Microsoft Corporation", exchange="NASDAQ",
            instrument_type="STOCK", tick_size=0.01, min_move=0.01
        )
        
        # Futures
        self._symbol_db["ES"] = SymbolInfo(
            name="ES", description="E-mini S&P 500", exchange="CME",
            instrument_type="FUTURE", tick_size=0.25, min_move=0.25,
            point_value=12.50, big_point_value=50.0, contract_size=1
        )
        
        self._symbol_db["NQ"] = SymbolInfo(
            name="NQ", description="E-mini NASDAQ", exchange="CME",
            instrument_type="FUTURE", tick_size=0.25, min_move=0.25,
            point_value=5.0, big_point_value=20.0, contract_size=1
        )
        
        # Forex
        self._symbol_db["EURUSD"] = SymbolInfo(
            name="EURUSD", description="Euro/US Dollar", exchange="FOREX",
            instrument_type="FOREX", tick_size=0.00001, min_move=0.0001,
            point_value=10.0, big_point_value=10.0, contract_size=100000
        )
        
        # Crypto
        self._symbol_db["BTCUSD"] = SymbolInfo(
            name="BTCUSD", description="Bitcoin/US Dollar", exchange="CRYPTO",
            instrument_type="CRYPTO", tick_size=0.01, min_move=0.01,
            point_value=1.0, big_point_value=1.0
        )
        
    def get_symbol_info(self, symbol_name: str) -> SymbolInfo:
        """Get symbol information from database or create default"""
        symbol_upper = symbol_name.upper()
        
        if symbol_upper in self._symbol_db:
            return self._symbol_db[symbol_upper]
        else:
            # Create default symbol info
            return SymbolInfo(name=symbol_name, description=f"Unknown symbol: {symbol_name}")

def __init_symbol_info__(self):
    """
    Inizializzazione sistema informazioni simboli
    """
    self.symbol_tracker = SymbolTracker()
    
    # Current symbol (try to detect from data source)
    self._current_symbol_name = "UNKNOWN"
    self._symbol_cache = {}
    
    # Try to extract symbol from Backtrader data
    if hasattr(self, 'data') and hasattr(self.data, '_name'):
        self._current_symbol_name = self.data._name
    
    print("✓ Symbol Information Functions initialized")

def _update_symbol_info(self):
    """Update symbol information from current data"""
    try:
        # Update current symbol info if available
        if hasattr(self, 'data'):
            # Try to get current price data for symbol
            current_symbol = self.symbol_tracker.get_symbol_info(self._current_symbol_name)
            
            if hasattr(self.data, 'close') and len(self.data.close) > 0:
                current_symbol.last_price = float(self.data.close[0])
            
            if hasattr(self.data, 'volume') and len(self.data.volume) > 0:
                current_symbol.volume = int(self.data.volume[0])
                
            self.symbol_tracker._current_symbol = current_symbol
            
    except Exception as e:
        self._handle_symbol_error("_update_symbol_info", e)

# =============================================================================
# SYMBOL IDENTIFICATION FUNCTIONS
# =============================================================================

def GetSymbolName(self):
    """
    Returns the name/symbol of current instrument
    """
    try:
        self._update_symbol_info()
        return self.symbol_tracker._current_symbol.name
        
    except Exception as e:
        self._handle_symbol_error("GetSymbolName", e)
        return "UNKNOWN"

def GetSymbolDescription(self):
    """
    Returns description of current symbol
    """
    try:
        self._update_symbol_info()
        return self.symbol_tracker._current_symbol.description
        
    except Exception as e:
        self._handle_symbol_error("GetSymbolDescription", e)
        return ""

def GetExchange(self):
    """
    Returns exchange where symbol is traded
    """
    try:
        self._update_symbol_info()
        return self.symbol_tracker._current_symbol.exchange
        
    except Exception as e:
        self._handle_symbol_error("GetExchange", e)
        return ""

def GetInstrumentType(self):
    """
    Returns type of instrument: STOCK, FUTURE, FOREX, OPTION, etc.
    """
    try:
        self._update_symbol_info()
        return self.symbol_tracker._current_symbol.instrument_type
        
    except Exception as e:
        self._handle_symbol_error("GetInstrumentType", e)
        return "STOCK"

def GetCurrency(self):
    """
    Returns currency of the instrument
    """
    try:
        self._update_symbol_info()
        return self.symbol_tracker._current_symbol.currency
        
    except Exception as e:
        self._handle_symbol_error("GetCurrency", e)
        return "USD"

# =============================================================================
# PRICE SCALING FUNCTIONS
# =============================================================================

def TickSize(self):
    """
    Returns minimum price movement (tick size) for the symbol
    """
    try:
        self._update_symbol_info()
        return self.symbol_tracker._current_symbol.tick_size
        
    except Exception as e:
        self._handle_symbol_error("TickSize", e)
        return 0.01

def MinMove(self):
    """
    Returns minimum move value for the symbol
    """
    try:
        self._update_symbol_info()
        return self.symbol_tracker._current_symbol.min_move
        
    except Exception as e:
        self._handle_symbol_error("MinMove", e)
        return 0.01

def PriceScale(self):
    """
    Returns price scale factor
    """
    try:
        self._update_symbol_info()
        return self.symbol_tracker._current_symbol.price_scale
        
    except Exception as e:
        self._handle_symbol_error("PriceScale", e)
        return 1.0

def PointValue(self):
    """
    Returns dollar value of one point movement
    """
    try:
        self._update_symbol_info()
        return self.symbol_tracker._current_symbol.point_value
        
    except Exception as e:
        self._handle_symbol_error("PointValue", e)
        return 1.0

def BigPointValue(self):
    """
    Returns dollar value of one big point movement (for futures)
    """
    try:
        self._update_symbol_info()
        return self.symbol_tracker._current_symbol.big_point_value
        
    except Exception as e:
        self._handle_symbol_error("BigPointValue", e)
        return 1.0

# =============================================================================
# CONTRACT SPECIFICATIONS
# =============================================================================

def ContractSize(self):
    """
    Returns contract size (for futures/forex)
    """
    try:
        self._update_symbol_info()
        return self.symbol_tracker._current_symbol.contract_size
        
    except Exception as e:
        self._handle_symbol_error("ContractSize", e)
        return 1

def MarginRequired(self):
    """
    Returns margin required for one contract
    """
    try:
        self._update_symbol_info()
        return self.symbol_tracker._current_symbol.margin_required
        
    except Exception as e:
        self._handle_symbol_error("MarginRequired", e)
        return 0.0

# =============================================================================
# SESSION INFORMATION
# =============================================================================

def SessionStartTime(self):
    """
    Returns session start time as string
    """
    try:
        self._update_symbol_info()
        return self.symbol_tracker._current_symbol.session_start
        
    except Exception as e:
        self._handle_symbol_error("SessionStartTime", e)
        return "09:30"

def SessionEndTime(self):
    """
    Returns session end time as string
    """
    try:
        self._update_symbol_info()
        return self.symbol_tracker._current_symbol.session_end
        
    except Exception as e:
        self._handle_symbol_error("SessionEndTime", e)
        return "16:00"

def GetTimeZone(self):
    """
    Returns timezone of the symbol's primary exchange
    """
    try:
        self._update_symbol_info()
        return self.symbol_tracker._current_symbol.timezone
        
    except Exception as e:
        self._handle_symbol_error("GetTimeZone", e)
        return "EST"

# =============================================================================
# MARKET DATA FUNCTIONS
# =============================================================================

def LastPrice(self):
    """
    Returns last traded price
    """
    try:
        self._update_symbol_info()
        return self.symbol_tracker._current_symbol.last_price
        
    except Exception as e:
        self._handle_symbol_error("LastPrice", e)
        return 0.0

def BidPrice(self):
    """
    Returns current bid price
    """
    try:
        self._update_symbol_info()
        return self.symbol_tracker._current_symbol.bid_price
        
    except Exception as e:
        self._handle_symbol_error("BidPrice", e)
        return 0.0

def AskPrice(self):
    """
    Returns current ask price
    """
    try:
        self._update_symbol_info()
        return self.symbol_tracker._current_symbol.ask_price
        
    except Exception as e:
        self._handle_symbol_error("AskPrice", e)
        return 0.0

def LastVolume(self):
    """
    Returns volume of last trade
    """
    try:
        self._update_symbol_info()
        return self.symbol_tracker._current_symbol.volume
        
    except Exception as e:
        self._handle_symbol_error("LastVolume", e)
        return 0

# =============================================================================
# STATUS FUNCTIONS
# =============================================================================

def IsTradeable(self):
    """
    Returns True if symbol is tradeable
    """
    try:
        self._update_symbol_info()
        return self.symbol_tracker._current_symbol.is_tradeable
        
    except Exception as e:
        self._handle_symbol_error("IsTradeable", e)
        return True

def MarketStatus(self):
    """
    Returns market status: OPEN, CLOSED, PRE_MARKET, AFTER_HOURS
    """
    try:
        self._update_symbol_info()
        return self.symbol_tracker._current_symbol.market_status
        
    except Exception as e:
        self._handle_symbol_error("MarketStatus", e)
        return "CLOSED"

def IsMarketOpen(self):
    """
    Returns True if market is currently open
    """
    try:
        status = self.MarketStatus()
        return status in ["OPEN", "PRE_MARKET", "AFTER_HOURS"]
        
    except Exception as e:
        self._handle_symbol_error("IsMarketOpen", e)
        return False

# =============================================================================
# SYMBOL DATABASE FUNCTIONS
# =============================================================================

def SetSymbol(self, symbol_name):
    """
    Set current symbol for analysis
    """
    try:
        self._current_symbol_name = str(symbol_name).upper()
        self._update_symbol_info()
        print(f"Symbol set to: {self._current_symbol_name}")
        return True
        
    except Exception as e:
        self._handle_symbol_error("SetSymbol", e)
        return False

def AddSymbolInfo(self, symbol_name, **kwargs):
    """
    Add or update symbol information in database
    """
    try:
        symbol_upper = str(symbol_name).upper()
        
        # Get existing info or create new
        if symbol_upper in self.symbol_tracker._symbol_db:
            symbol_info = self.symbol_tracker._symbol_db[symbol_upper]
        else:
            symbol_info = SymbolInfo(name=symbol_upper)
        
        # Update with provided kwargs
        for key, value in kwargs.items():
            if hasattr(symbol_info, key):
                setattr(symbol_info, key, value)
        
        # Store in database
        self.symbol_tracker._symbol_db[symbol_upper] = symbol_info
        
        print(f"Symbol info updated for: {symbol_upper}")
        return True
        
    except Exception as e:
        self._handle_symbol_error("AddSymbolInfo", e)
        return False

def GetSymbolInfo(self, symbol_name=None):
    """
    Get complete symbol information as dictionary
    """
    try:
        if symbol_name:
            symbol_info = self.symbol_tracker.get_symbol_info(symbol_name)
        else:
            self._update_symbol_info()
            symbol_info = self.symbol_tracker._current_symbol
        
        return {
            'name': symbol_info.name,
            'description': symbol_info.description,
            'exchange': symbol_info.exchange,
            'instrument_type': symbol_info.instrument_type,
            'currency': symbol_info.currency,
            'tick_size': symbol_info.tick_size,
            'min_move': symbol_info.min_move,
            'point_value': symbol_info.point_value,
            'big_point_value': symbol_info.big_point_value,
            'contract_size': symbol_info.contract_size,
            'session_start': symbol_info.session_start,
            'session_end': symbol_info.session_end,
            'timezone': symbol_info.timezone,
            'last_price': symbol_info.last_price,
            'is_tradeable': symbol_info.is_tradeable,
            'market_status': symbol_info.market_status
        }
        
    except Exception as e:
        self._handle_symbol_error("GetSymbolInfo", e)
        return {}

def ListAvailableSymbols(self):
    """
    Returns list of available symbols in database
    """
    try:
        return list(self.symbol_tracker._symbol_db.keys())
        
    except Exception as e:
        self._handle_symbol_error("ListAvailableSymbols", e)
        return []

# =============================================================================
# CALCULATION HELPERS
# =============================================================================

def PointsToPrice(self, points):
    """
    Convert points to price units
    """
    try:
        tick_size = self.TickSize()
        return float(points) * tick_size
        
    except Exception as e:
        self._handle_symbol_error("PointsToPrice", e)
        return float(points)

def PriceToPoints(self, price):
    """
    Convert price to points
    """
    try:
        tick_size = self.TickSize()
        if tick_size > 0:
            return float(price) / tick_size
        return float(price)
        
    except Exception as e:
        self._handle_symbol_error("PriceToPoints", e)
        return float(price)

def PointsToDollars(self, points):
    """
    Convert points to dollar value
    """
    try:
        point_value = self.PointValue()
        return float(points) * point_value
        
    except Exception as e:
        self._handle_symbol_error("PointsToDollars", e)
        return float(points)

def DollarsToPoints(self, dollars):
    """
    Convert dollar amount to points
    """
    try:
        point_value = self.PointValue()
        if point_value > 0:
            return float(dollars) / point_value
        return float(dollars)
        
    except Exception as e:
        self._handle_symbol_error("DollarsToPoints", e)
        return float(dollars)

# =============================================================================
# SYMBOL CLASSIFICATION HELPERS
# =============================================================================

def IsStock(self):
    """Returns True if current symbol is a stock"""
    return self.GetInstrumentType().upper() in ["STOCK", "ETF"]

def IsFuture(self):
    """Returns True if current symbol is a future"""
    return self.GetInstrumentType().upper() == "FUTURE"

def IsForex(self):
    """Returns True if current symbol is forex"""
    return self.GetInstrumentType().upper() == "FOREX"

def IsOption(self):
    """Returns True if current symbol is an option"""
    return self.GetInstrumentType().upper() == "OPTION"

def IsCrypto(self):
    """Returns True if current symbol is cryptocurrency"""
    return self.GetInstrumentType().upper() == "CRYPTO"

def IsIndex(self):
    """Returns True if current symbol is an index"""
    return self.GetInstrumentType().upper() == "INDEX"

# =============================================================================
# ERROR HANDLING
# =============================================================================

def _handle_symbol_error(self, function_name, error):
    """Centralized error handling for symbol functions"""
    try:
        self.symbol_tracker._error_count += 1
        self.symbol_tracker._last_error = f"{function_name}: {str(error)}"
        
        # Log error if in debug mode
        if hasattr(self, '_debug_mode') and self._debug_mode:
            print(f"Symbol Error in {function_name}: {error}")
            
    except Exception:
        pass  # Prevent error handling from causing more errors

def GetLastSymbolError(self):
    """Returns last symbol error message"""
    return self.symbol_tracker._last_error

def GetSymbolErrorCount(self):
    """Returns total count of symbol errors"""
    return self.symbol_tracker._error_count

def ResetSymbolErrorCount(self):
    """Reset symbol error counter"""
    self.symbol_tracker._error_count = 0
    self.symbol_tracker._last_error = None

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_symbol_stats(self):
    """Get performance statistics for symbol operations"""
    try:
        cache_hit_rate = 0.0
        if self.symbol_tracker._lookups_count > 0:
            cache_hit_rate = (self.symbol_tracker._cache_hits / self.symbol_tracker._lookups_count) * 100.0
        
        return {
            'lookups_count': self.symbol_tracker._lookups_count,
            'cache_hits': self.symbol_tracker._cache_hits,
            'cache_hit_rate_percent': cache_hit_rate,
            'error_count': self.symbol_tracker._error_count,
            'last_error': self.symbol_tracker._last_error,
            'symbols_in_db': len(self.symbol_tracker._symbol_db),
            'current_symbol': self._current_symbol_name
        }
        
    except Exception as e:
        print(f"Error getting symbol stats: {e}")
        return {}

def reset_symbol_cache(self):
    """Reset symbol lookup counters"""
    try:
        self.symbol_tracker._lookups_count = 0
        self.symbol_tracker._cache_hits = 0
        self._symbol_cache.clear()
        print("✓ Symbol info cache reset completed")
        
    except Exception as e:
        print(f"Error resetting symbol cache: {e}")

# =============================================================================
# TEST SUITE
# =============================================================================

def run_symbol_info_test():
    """Comprehensive test suite for symbol information functions"""
    print("=== SYMBOL INFORMATION FUNCTIONS TEST ===")
    
    class MockStrategy:
        def __init__(self):
            self.__init_symbol_info__()
            self._debug_mode = True
            
            # Mock data object
            class MockData:
                def __init__(self):
                    self._name = "AAPL"
                    self.close = [150.25]
                    self.volume = [1000000]
            
            self.data = MockData()
    
    strategy = MockStrategy()
    
    # Add all symbol methods
    import types
    for name, obj in globals().items():
        if callable(obj) and (name.startswith('__init_symbol') or 
                             name.startswith('_update_symbol') or 
                             name.startswith('_handle_symbol') or
                             name in ['GetSymbolName', 'GetSymbolDescription', 'GetExchange',
                                     'GetInstrumentType', 'GetCurrency', 'TickSize', 'MinMove',
                                     'PriceScale', 'PointValue', 'BigPointValue', 'ContractSize',
                                     'SessionStartTime', 'SessionEndTime', 'LastPrice', 'IsTradeable',
                                     'MarketStatus', 'SetSymbol', 'GetSymbolInfo', 'ListAvailableSymbols',
                                     'PointsToPrice', 'PriceToPoints', 'PointsToDollars',
                                     'IsStock', 'IsFuture', 'IsForex', 'get_symbol_stats']):
            setattr(strategy, name, types.MethodType(obj, strategy))
    
    print("Testing Basic Symbol Information:")
    print(f"  Symbol Name: {strategy.GetSymbolName()}")
    print(f"  Description: {strategy.GetSymbolDescription()}")
    print(f"  Exchange: {strategy.GetExchange()}")
    print(f"  Instrument Type: {strategy.GetInstrumentType()}")
    print(f"  Currency: {strategy.GetCurrency()}")
    
    print("\nTesting Price Scaling:")
    print(f"  Tick Size: {strategy.TickSize()}")
    print(f"  Min Move: {strategy.MinMove()}")
    print(f"  Point Value: ${strategy.PointValue()}")
    print(f"  Big Point Value: ${strategy.BigPointValue()}")
    
    print("\nTesting Contract Information:")
    print(f"  Contract Size: {strategy.ContractSize()}")
    print(f"  Session Start: {strategy.SessionStartTime()}")
    print(f"  Session End: {strategy.SessionEndTime()}")
    
    print("\nTesting Market Data:")
    print(f"  Last Price: {strategy.LastPrice()}")
    print(f"  Is Tradeable: {strategy.IsTradeable()}")
    print(f"  Market Status: {strategy.MarketStatus()}")
    
    print("\nTesting Symbol Classification:")
    print(f"  Is Stock: {strategy.IsStock()}")
    print(f"  Is Future: {strategy.IsFuture()}")
    print(f"  Is Forex: {strategy.IsForex()}")
    
    print("\nTesting Available Symbols:")
    symbols = strategy.ListAvailableSymbols()
    print(f"  Available Symbols ({len(symbols)}): {', '.join(symbols[:10])}...")
    
    print("\nTesting Different Symbols:")
    for symbol in ["SPY", "ES", "EURUSD", "BTCUSD"]:
        strategy.SetSymbol(symbol)
        info = strategy.GetSymbolInfo()
        print(f"  {symbol}: {info['instrument_type']}, Tick: {info['tick_size']}, Point Value: ${info['point_value']}")
    
    print("\nTesting Conversion Functions:")
    strategy.SetSymbol("ES")  # E-mini S&P 500
    print(f"  ES - 10 points to price: {strategy.PointsToPrice(10)}")
    print(f"  ES - 10 points to dollars: ${strategy.PointsToDollars(10)}")
    print(f"  ES - $500 to points: {strategy.DollarsToPoints(500)} points")
    
    print("\nTesting Custom Symbol Addition:")
    custom_success = strategy.AddSymbolInfo("CUSTOM", 
                                           description="Custom Test Symbol",
                                           instrument_type="STOCK",
                                           tick_size=0.05,
                                           point_value=2.0)
    print(f"  Custom symbol added: {custom_success}")
    
    if custom_success:
        strategy.SetSymbol("CUSTOM")
        custom_info = strategy.GetSymbolInfo()
        print(f"  Custom symbol info: {custom_info['description']}, Tick: {custom_info['tick_size']}")
    
    print("\nTesting Complete Symbol Info:")
    strategy.SetSymbol("AAPL")
    complete_info = strategy.GetSymbolInfo()
    print("  AAPL Complete Info:")
    for key, value in complete_info.items():
        print(f"    {key}: {value}")
    
    # Performance statistics
    stats = strategy.get_symbol_stats()
    print(f"\nPerformance Statistics:")
    print(f"  Lookups: {stats['lookups_count']}")
    print(f"  Symbols in DB: {stats['symbols_in_db']}")
    print(f"  Current Symbol: {stats['current_symbol']}")
    print(f"  Error Count: {stats['error_count']}")
    
    print("✓ Symbol Information Functions test completed")

if __name__ == "__main__":
    run_symbol_info_test()