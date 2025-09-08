# =============================================================================
# ORDER MANAGEMENT & POSITION TRACKING - EASYLANGUAGE INTEGRATION
# Compatible with Backtrader Strategy Framework
# =============================================================================

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple
import warnings
from enum import Enum

class PositionSide(Enum):
    FLAT = 0
    LONG = 1
    SHORT = -1

class OrderType(Enum):
    MARKET = "Market"
    LIMIT = "Limit"
    STOP = "Stop"
    STOP_LIMIT = "StopLimit"

@dataclass
class TradeInfo:
    """Informazioni complete su un trade"""
    entry_price: float
    entry_date: int
    entry_time: int
    entry_bar: int
    exit_price: Optional[float] = None
    exit_date: Optional[int] = None
    exit_time: Optional[int] = None
    exit_bar: Optional[int] = None
    contracts: int = 1
    side: PositionSide = PositionSide.FLAT
    signal_name: str = ""
    profit: float = 0.0
    is_open: bool = True

class PositionTracker:
    """
    Tracker ottimizzato per posizioni e ordini con integrazione Backtrader
    """
    def __init__(self, max_history=1000):
        self.max_history = max_history
        
        # Current position state
        self.market_position = 0  # 0=flat, >0=long, <0=short
        self.current_contracts = 0
        self.position_side = PositionSide.FLAT
        
        # Entry tracking
        self.entry_price = 0.0
        self.entry_date = 0
        self.entry_time = 0
        self.entry_bar = 0
        self.bars_since_entry = 0
        
        # Exit tracking
        self.exit_price = 0.0
        self.bars_since_exit = 0
        
        # Trade history
        self.trade_history: List[TradeInfo] = []
        self.open_trades: List[TradeInfo] = []
        
        # Stop management
        self.stop_loss_price = 0.0
        self.profit_target_price = 0.0
        self.trailing_stop_price = 0.0
        self.break_even_price = 0.0
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        self.net_profit = 0.0
        
        # Risk management
        self.max_contracts = 1
        self.max_risk_per_trade = 0.02  # 2% default
        
        # Bar tracking
        self.current_bar = 0
        self.last_update_bar = -1

def __init_order_management__(self):
    """
    Inizializzazione sistema di gestione ordini ottimizzato
    """
    self.position_tracker = PositionTracker()
    
    # Integration with Backtrader
    self._bt_strategy = self if hasattr(self, 'buy') else None
    self._use_backtrader_orders = True
    
    # Order queue per next bar execution
    self._pending_orders = []
    
    # Position size management
    self.default_contracts = 1
    
    # Performance optimization
    self._position_cache = {}
    self._last_position_update = -1
    
    print("✓ Order Management System initialized")

def _update_position_tracking(self):
    """
    Aggiorna il tracking delle posizioni con la barra corrente
    """
    try:
        if hasattr(self, 'data') and len(self.data.close) > 0:
            current_bar = len(self.data.close) - 1
            
            # Skip se già aggiornato
            if current_bar == self._last_position_update:
                return
            
            self.position_tracker.current_bar = current_bar
            
            # Update bars since entry/exit
            if self.position_tracker.market_position != 0:
                self.position_tracker.bars_since_entry = current_bar - self.position_tracker.entry_bar
            else:
                if self.position_tracker.bars_since_exit >= 0:
                    self.position_tracker.bars_since_exit += 1
            
            # Update trailing stops and targets
            self._update_stop_management()
            
            self._last_position_update = current_bar
            
    except Exception as e:
        print(f"Error updating position tracking: {e}")

def _update_stop_management(self):
    """Aggiorna stop loss e profit targets"""
    if self.position_tracker.market_position == 0:
        return
    
    try:
        current_price = float(self.data.close[0])
        
        # Update trailing stop
        if self.position_tracker.trailing_stop_price > 0:
            if self.position_tracker.market_position > 0:  # Long position
                # Trail stop up
                new_stop = current_price - self.position_tracker.trailing_stop_price
                if new_stop > self.position_tracker.stop_loss_price:
                    self.position_tracker.stop_loss_price = new_stop
            else:  # Short position
                # Trail stop down
                new_stop = current_price + self.position_tracker.trailing_stop_price
                if new_stop < self.position_tracker.stop_loss_price or self.position_tracker.stop_loss_price == 0:
                    self.position_tracker.stop_loss_price = new_stop
        
        # Check break-even
        if self.position_tracker.break_even_price > 0:
            if self.position_tracker.market_position > 0:  # Long
                if current_price >= self.position_tracker.break_even_price:
                    self.position_tracker.stop_loss_price = self.position_tracker.entry_price
            else:  # Short
                if current_price <= self.position_tracker.break_even_price:
                    self.position_tracker.stop_loss_price = self.position_tracker.entry_price
                    
    except Exception as e:
        print(f"Error updating stop management: {e}")

# =============================================================================
# POSITION INFORMATION FUNCTIONS
# =============================================================================

def MarketPosition(self):
    """
    Returns current market position: 0=flat, >0=long, <0=short
    """
    try:
        self._update_position_tracking()
        
        # Try Backtrader integration first
        if self._bt_strategy and hasattr(self._bt_strategy, 'position'):
            bt_size = self._bt_strategy.position.size
            if bt_size != 0:
                self.position_tracker.market_position = int(bt_size)
                self.position_tracker.current_contracts = abs(int(bt_size))
                return int(bt_size)
        
        return self.position_tracker.market_position
        
    except Exception as e:
        print(f"MarketPosition error: {e}")
        return 0

def CurrentContracts(self):
    """Returns number of contracts in current position"""
    try:
        self._update_position_tracking()
        
        # Backtrader integration
        if self._bt_strategy and hasattr(self._bt_strategy, 'position'):
            return abs(int(self._bt_strategy.position.size))
        
        return abs(self.position_tracker.current_contracts)
        
    except Exception as e:
        print(f"CurrentContracts error: {e}")
        return 0

def EntryPrice(self, position_number=0):
    """Returns entry price of specified position (0=current position)"""
    try:
        self._update_position_tracking()
        
        if position_number == 0:
            # Current position
            if self._bt_strategy and hasattr(self._bt_strategy, 'position'):
                return float(self._bt_strategy.position.price) if self._bt_strategy.position.size != 0 else 0.0
            return self.position_tracker.entry_price
        else:
            # Historical position
            if position_number <= len(self.position_tracker.trade_history):
                trade_idx = -(position_number)  # Recent trades first
                return self.position_tracker.trade_history[trade_idx].entry_price
            return 0.0
            
    except Exception as e:
        print(f"EntryPrice error: {e}")
        return 0.0

def EntryDate(self, position_number=0):
    """Returns entry date in EasyLanguage format (YYYYMMDD)"""
    try:
        if position_number == 0:
            return self.position_tracker.entry_date
        else:
            if position_number <= len(self.position_tracker.trade_history):
                trade_idx = -(position_number)
                return self.position_tracker.trade_history[trade_idx].entry_date
            return 0
            
    except Exception as e:
        print(f"EntryDate error: {e}")
        return 0

def EntryTime(self, position_number=0):
    """Returns entry time in EasyLanguage format (HHMM)"""
    try:
        if position_number == 0:
            return self.position_tracker.entry_time
        else:
            if position_number <= len(self.position_tracker.trade_history):
                trade_idx = -(position_number)
                return self.position_tracker.trade_history[trade_idx].entry_time
            return 0
            
    except Exception as e:
        print(f"EntryTime error: {e}")
        return 0

def ExitPrice(self, position_number=1):
    """Returns exit price of last closed position"""
    try:
        if position_number <= len(self.position_tracker.trade_history):
            trade_idx = -(position_number)
            trade = self.position_tracker.trade_history[trade_idx]
            return trade.exit_price if not trade.is_open else 0.0
        return 0.0
        
    except Exception as e:
        print(f"ExitPrice error: {e}")
        return 0.0

def PositionProfit(self, position_number=0):
    """Returns profit/loss of specified position"""
    try:
        self._update_position_tracking()
        
        if position_number == 0:
            # Current open position
            if self.position_tracker.market_position != 0:
                current_price = float(self.data.close[0])
                entry_price = self.EntryPrice(0)
                contracts = abs(self.position_tracker.current_contracts)
                
                if self.position_tracker.market_position > 0:  # Long
                    return (current_price - entry_price) * contracts
                else:  # Short
                    return (entry_price - current_price) * contracts
            return 0.0
        else:
            # Historical position
            if position_number <= len(self.position_tracker.trade_history):
                trade_idx = -(position_number)
                return self.position_tracker.trade_history[trade_idx].profit
            return 0.0
            
    except Exception as e:
        print(f"PositionProfit error: {e}")
        return 0.0

def OpenPositionProfit(self):
    """Returns current open position profit/loss"""
    return self.PositionProfit(0)

def BarsSinceEntry(self, position_number=0):
    """Returns bars since entry for specified position"""
    try:
        if position_number == 0:
            return max(0, self.position_tracker.bars_since_entry)
        else:
            if position_number <= len(self.position_tracker.trade_history):
                trade_idx = -(position_number)
                trade = self.position_tracker.trade_history[trade_idx]
                if trade.exit_bar is not None:
                    return trade.exit_bar - trade.entry_bar
                else:
                    return self.position_tracker.current_bar - trade.entry_bar
            return 0
            
    except Exception as e:
        print(f"BarsSinceEntry error: {e}")
        return 0

def BarsSinceExit(self, position_number=1):
    """Returns bars since exit of last closed position"""
    try:
        if len(self.position_tracker.trade_history) > 0:
            last_trade = self.position_tracker.trade_history[-1]
            if not last_trade.is_open and last_trade.exit_bar is not None:
                return self.position_tracker.current_bar - last_trade.exit_bar
        return 0
        
    except Exception as e:
        print(f"BarsSinceExit error: {e}")
        return 0

# =============================================================================
# ORDER ENTRY FUNCTIONS
# =============================================================================

def Buy(self, contracts=None, name="", next_bar=True, order_type="Market", price=0.0):
    """
    Enter long position
    """
    try:
        contracts = contracts if contracts is not None else self.default_contracts
        
        # Backtrader integration
        if self._use_backtrader_orders and self._bt_strategy and hasattr(self._bt_strategy, 'buy'):
            if order_type.lower() == "market":
                self._bt_strategy.buy(size=contracts)
            elif order_type.lower() == "limit" and price > 0:
                self._bt_strategy.buy(size=contracts, price=price, exectype=self._bt_strategy.Order.Limit)
            elif order_type.lower() == "stop" and price > 0:
                self._bt_strategy.buy(size=contracts, price=price, exectype=self._bt_strategy.Order.Stop)
        else:
            # Manual position tracking
            self._execute_manual_order("buy", contracts, name, order_type, price)
        
        # Log the order
        current_price = float(self.data.close[0]) if hasattr(self, 'data') else price
        print(f"BUY Order: {contracts} contracts at {current_price:.2f} ({order_type})")
        
    except Exception as e:
        print(f"Buy order error: {e}")

def Sell(self, contracts=None, name="", next_bar=True, order_type="Market", price=0.0):
    """
    Exit long position or enter short
    """
    try:
        contracts = contracts if contracts is not None else self.default_contracts
        
        # Determine if this is an exit or new short entry
        current_position = self.MarketPosition()
        
        if current_position > 0:
            # Exit long position
            if self._use_backtrader_orders and self._bt_strategy and hasattr(self._bt_strategy, 'sell'):
                if order_type.lower() == "market":
                    self._bt_strategy.sell(size=contracts)
                elif order_type.lower() == "limit" and price > 0:
                    self._bt_strategy.sell(size=contracts, price=price, exectype=self._bt_strategy.Order.Limit)
                elif order_type.lower() == "stop" and price > 0:
                    self._bt_strategy.sell(size=contracts, price=price, exectype=self._bt_strategy.Order.Stop)
            else:
                self._execute_manual_order("sell", contracts, name, order_type, price)
        
        current_price = float(self.data.close[0]) if hasattr(self, 'data') else price
        print(f"SELL Order: {contracts} contracts at {current_price:.2f} ({order_type})")
        
    except Exception as e:
        print(f"Sell order error: {e}")

def BuyToCover(self, contracts=None, name="", next_bar=True, order_type="Market", price=0.0):
    """
    Exit short position
    """
    try:
        contracts = contracts if contracts is not None else abs(self.position_tracker.current_contracts)
        
        if self._use_backtrader_orders and self._bt_strategy and hasattr(self._bt_strategy, 'buy'):
            # In Backtrader, buying to cover a short is just a buy
            if order_type.lower() == "market":
                self._bt_strategy.buy(size=contracts)
            elif order_type.lower() == "limit" and price > 0:
                self._bt_strategy.buy(size=contracts, price=price, exectype=self._bt_strategy.Order.Limit)
            elif order_type.lower() == "stop" and price > 0:
                self._bt_strategy.buy(size=contracts, price=price, exectype=self._bt_strategy.Order.Stop)
        else:
            self._execute_manual_order("buytocover", contracts, name, order_type, price)
        
        current_price = float(self.data.close[0]) if hasattr(self, 'data') else price
        print(f"BUY TO COVER Order: {contracts} contracts at {current_price:.2f} ({order_type})")
        
    except Exception as e:
        print(f"BuyToCover order error: {e}")

def SellShort(self, contracts=None, name="", next_bar=True, order_type="Market", price=0.0):
    """
    Enter short position
    """
    try:
        contracts = contracts if contracts is not None else self.default_contracts
        
        if self._use_backtrader_orders and self._bt_strategy and hasattr(self._bt_strategy, 'sell'):
            if order_type.lower() == "market":
                self._bt_strategy.sell(size=contracts)
            elif order_type.lower() == "limit" and price > 0:
                self._bt_strategy.sell(size=contracts, price=price, exectype=self._bt_strategy.Order.Limit)
            elif order_type.lower() == "stop" and price > 0:
                self._bt_strategy.sell(size=contracts, price=price, exectype=self._bt_strategy.Order.Stop)
        else:
            self._execute_manual_order("sellshort", contracts, name, order_type, price)
        
        current_price = float(self.data.close[0]) if hasattr(self, 'data') else price
        print(f"SELL SHORT Order: {contracts} contracts at {current_price:.2f} ({order_type})")
        
    except Exception as e:
        print(f"SellShort order error: {e}")

def _execute_manual_order(self, action, contracts, name, order_type, price):
    """
    Esegue ordini manuali quando Backtrader non è disponibile
    """
    try:
        current_price = float(self.data.close[0]) if hasattr(self, 'data') else price
        current_date = int(self.Date()) if hasattr(self, 'Date') else 20240101
        current_time = int(self.Time()) if hasattr(self, 'Time') else 1200
        
        if action.lower() in ["buy", "buytocover"]:
            # Enter or exit to long
            if self.position_tracker.market_position <= 0:
                # New long position
                self.position_tracker.market_position = contracts
                self.position_tracker.current_contracts = contracts
                self.position_tracker.entry_price = current_price
                self.position_tracker.entry_date = current_date
                self.position_tracker.entry_time = current_time
                self.position_tracker.entry_bar = self.position_tracker.current_bar
                self.position_tracker.bars_since_entry = 0
                
                # Create trade record
                trade = TradeInfo(
                    entry_price=current_price,
                    entry_date=current_date,
                    entry_time=current_time,
                    entry_bar=self.position_tracker.current_bar,
                    contracts=contracts,
                    side=PositionSide.LONG,
                    signal_name=name,
                    is_open=True
                )
                self.position_tracker.open_trades.append(trade)
            
        elif action.lower() in ["sell", "sellshort"]:
            # Enter short or exit long
            if action.lower() == "sell" and self.position_tracker.market_position > 0:
                # Exit long position
                self._close_position(current_price, current_date, current_time)
            else:
                # New short position
                self.position_tracker.market_position = -contracts
                self.position_tracker.current_contracts = contracts
                self.position_tracker.entry_price = current_price
                self.position_tracker.entry_date = current_date
                self.position_tracker.entry_time = current_time
                self.position_tracker.entry_bar = self.position_tracker.current_bar
                self.position_tracker.bars_since_entry = 0
                
                # Create trade record
                trade = TradeInfo(
                    entry_price=current_price,
                    entry_date=current_date,
                    entry_time=current_time,
                    entry_bar=self.position_tracker.current_bar,
                    contracts=contracts,
                    side=PositionSide.SHORT,
                    signal_name=name,
                    is_open=True
                )
                self.position_tracker.open_trades.append(trade)
                
    except Exception as e:
        print(f"Manual order execution error: {e}")

def _close_position(self, exit_price, exit_date, exit_time):
    """Chiude la posizione corrente e calcola il profit"""
    try:
        if self.position_tracker.open_trades:
            trade = self.position_tracker.open_trades[-1]
            
            # Calculate profit
            if trade.side == PositionSide.LONG:
                profit = (exit_price - trade.entry_price) * trade.contracts
            else:  # SHORT
                profit = (trade.entry_price - exit_price) * trade.contracts
            
            # Update trade record
            trade.exit_price = exit_price
            trade.exit_date = exit_date
            trade.exit_time = exit_time
            trade.exit_bar = self.position_tracker.current_bar
            trade.profit = profit
            trade.is_open = False
            
            # Move to history
            self.position_tracker.trade_history.append(trade)
            self.position_tracker.open_trades.remove(trade)
            
            # Update performance stats
            self.position_tracker.total_trades += 1
            if profit > 0:
                self.position_tracker.winning_trades += 1
                self.position_tracker.gross_profit += profit
            else:
                self.position_tracker.losing_trades += 1
                self.position_tracker.gross_loss += abs(profit)
            
            self.position_tracker.net_profit += profit
        
        # Reset position
        self.position_tracker.market_position = 0
        self.position_tracker.current_contracts = 0
        self.position_tracker.bars_since_exit = 0
        
    except Exception as e:
        print(f"Close position error: {e}")

# =============================================================================
# STOP MANAGEMENT FUNCTIONS
# =============================================================================

def SetStopLoss(self, amount, type="points"):
    """
    Set stop loss for current position
    amount: stop distance in points or dollars
    type: 'points', 'dollars', 'percent'
    """
    try:
        if self.position_tracker.market_position == 0:
            return False
        
        current_price = float(self.data.close[0])
        entry_price = self.position_tracker.entry_price
        
        if type.lower() == "points":
            if self.position_tracker.market_position > 0:  # Long
                self.position_tracker.stop_loss_price = entry_price - amount
            else:  # Short
                self.position_tracker.stop_loss_price = entry_price + amount
                
        elif type.lower() == "dollars":
            contracts = abs(self.position_tracker.current_contracts)
            points_per_dollar = 1.0  # Assume 1 point = $1, adjust per instrument
            stop_points = amount / (contracts * points_per_dollar)
            
            if self.position_tracker.market_position > 0:  # Long
                self.position_tracker.stop_loss_price = entry_price - stop_points
            else:  # Short
                self.position_tracker.stop_loss_price = entry_price + stop_points
                
        elif type.lower() == "percent":
            percent_decimal = amount / 100.0
            if self.position_tracker.market_position > 0:  # Long
                self.position_tracker.stop_loss_price = entry_price * (1 - percent_decimal)
            else:  # Short
                self.position_tracker.stop_loss_price = entry_price * (1 + percent_decimal)
        
        print(f"Stop Loss set at: {self.position_tracker.stop_loss_price:.2f}")
        return True
        
    except Exception as e:
        print(f"SetStopLoss error: {e}")
        return False

def SetProfitTarget(self, amount, type="points"):
    """
    Set profit target for current position
    """
    try:
        if self.position_tracker.market_position == 0:
            return False
        
        entry_price = self.position_tracker.entry_price
        
        if type.lower() == "points":
            if self.position_tracker.market_position > 0:  # Long
                self.position_tracker.profit_target_price = entry_price + amount
            else:  # Short
                self.position_tracker.profit_target_price = entry_price - amount
                
        elif type.lower() == "percent":
            percent_decimal = amount / 100.0
            if self.position_tracker.market_position > 0:  # Long
                self.position_tracker.profit_target_price = entry_price * (1 + percent_decimal)
            else:  # Short
                self.position_tracker.profit_target_price = entry_price * (1 - percent_decimal)
        
        print(f"Profit Target set at: {self.position_tracker.profit_target_price:.2f}")
        return True
        
    except Exception as e:
        print(f"SetProfitTarget error: {e}")
        return False

def SetDollarTrailingStop(self, amount):
    """Set trailing stop in dollar amount"""
    try:
        if self.position_tracker.market_position != 0:
            self.position_tracker.trailing_stop_price = amount
            print(f"Dollar Trailing Stop set: ${amount}")
            return True
        return False
        
    except Exception as e:
        print(f"SetDollarTrailingStop error: {e}")
        return False

def SetPercentTrailingStop(self, percent):
    """Set trailing stop in percentage"""
    try:
        if self.position_tracker.market_position != 0:
            current_price = float(self.data.close[0])
            amount = current_price * (percent / 100.0)
            self.position_tracker.trailing_stop_price = amount
            print(f"Percent Trailing Stop set: {percent}%")
            return True
        return False
        
    except Exception as e:
        print(f"SetPercentTrailingStop error: {e}")
        return False

def SetBreakEven(self):
    """Set break-even stop"""
    try:
        if self.position_tracker.market_position != 0:
            self.position_tracker.break_even_price = self.position_tracker.entry_price
            print("Break-even stop activated")
            return True
        return False
        
    except Exception as e:
        print(f"SetBreakEven error: {e}")
        return False

def ExitPosition(self):
    """Exit current position at market"""
    try:
        if self.position_tracker.market_position > 0:
            self.Sell(self.position_tracker.current_contracts)
        elif self.position_tracker.market_position < 0:
            self.BuyToCover(self.position_tracker.current_contracts)
        
        return True
        
    except Exception as e:
        print(f"ExitPosition error: {e}")
        return False

# =============================================================================
# PERFORMANCE FUNCTIONS
# =============================================================================

def NetProfit(self):
    """Returns total net profit"""
    return self.position_tracker.net_profit

def GrossProfit(self):
    """Returns total gross profit"""
    return self.position_tracker.gross_profit

def GrossLoss(self):
    """Returns total gross loss"""
    return self.position_tracker.gross_loss

def TotalTrades(self):
    """Returns total number of completed trades"""
    return self.position_tracker.total_trades

def WinningTrades(self):
    """Returns number of winning trades"""
    return self.position_tracker.winning_trades

def LosingTrades(self):
    """Returns number of losing trades"""
    return self.position_tracker.losing_trades

def PercentProfitable(self):
    """Returns percentage of profitable trades"""
    try:
        if self.position_tracker.total_trades > 0:
            return (self.position_tracker.winning_trades / self.position_tracker.total_trades) * 100.0
        return 0.0
    except:
        return 0.0

# =============================================================================
# TEST SUITE
# =============================================================================

def run_order_management_test():
    """Test suite per il sistema di gestione ordini"""
    print("=== ORDER MANAGEMENT TEST ===")
    
    class MockStrategy:
        def __init__(self):
            class MockData:
                def __init__(self):
                    self.close = [100.0, 101.0, 102.5, 101.8, 103.2]
                    self.high = [c + 1 for c in self.close]
                    self.low = [c - 1 for c in self.close]
                    self.open = [c + 0.5 for c in self.close]
                    self.current_idx = 0
                
                def __getitem__(self, idx):
                    return self.close[max(0, min(len(self.close) - 1, self.current_idx + idx))]
            
            self.data = MockData()
            self.__init_order_management__()
        
        def Date(self):
            return 20240315
        
        def Time(self):
            return 1430
    
    strategy = MockStrategy()
    
    # Add all methods
    import types
    for name, obj in globals().items():
        if callable(obj) and (name.startswith('__init_order') or 
                             name.startswith('_update_') or 
                             name.startswith('_execute_') or
                             name.startswith('_close_') or
                             name in ['MarketPosition', 'CurrentContracts', 'EntryPrice', 
                                     'Buy', 'Sell', 'SellShort', 'BuyToCover',
                                     'SetStopLoss', 'SetProfitTarget', 'ExitPosition',
                                     'NetProfit', 'TotalTrades', 'PercentProfitable']):
            setattr(strategy, name, types.MethodType(obj, strategy))
    
    # Test sequence
    print(f"Initial Position: {strategy.MarketPosition()}")
    
    # Test buy order
    strategy.Buy(1, "TestBuy")
    print(f"After Buy - Position: {strategy.MarketPosition()}, Contracts: {strategy.CurrentContracts()}")
    print(f"Entry Price: {strategy.EntryPrice():.2f}")
    
    # Move to next bar
    strategy.data.current_idx = 1
    strategy._update_position_tracking()
    print(f"Bars Since Entry: {strategy.BarsSinceEntry()}")
    print(f"Position Profit: {strategy.PositionProfit():.2f}")
    
    # Test stop loss
    strategy.SetStopLoss(2.0, "points")
    strategy.SetProfitTarget(3.0, "points")
    
    # Test exit
    strategy.data.current_idx = 2
    strategy.Sell(1, "TestExit")
    print(f"After Exit - Position: {strategy.MarketPosition()}")
    print(f"Net Profit: {strategy.NetProfit():.2f}")
    print(f"Total Trades: {strategy.TotalTrades()}")
    
    # Test short position
    strategy.SellShort(1, "TestShort")
    print(f"Short Position: {strategy.MarketPosition()}")
    
    strategy.BuyToCover(1, "CoverShort")
    print(f"After Cover: {strategy.MarketPosition()}")
    
    print("✓ Order Management test completed")

if __name__ == "__main__":
    run_order_management_test()