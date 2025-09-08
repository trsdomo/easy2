"""
Enhanced AST Nodes - Sistema Completo
Tutti i nodi AST necessari per il nuovo sistema EasyLanguage enhanced
"""

from dataclasses import dataclass
from typing import List, Optional, Any, Union

# ===================== BASE NODES =====================

@dataclass
class ASTNode:
    """Nodo base per tutti gli elementi AST"""
    node_type: str

@dataclass
class Program(ASTNode):
    """Nodo radice che contiene tutti gli statement del programma"""
    statements: List[Any]

@dataclass
class Block(ASTNode):
    """Blocco di statement (BEGIN...END o {...})"""
    statements: List[Any]

# ===================== LITERALS E PRIMITIVES =====================

@dataclass
class Number(ASTNode):
    """Numero (intero o float)"""
    value: Union[int, float]

@dataclass
class StringLiteral(ASTNode):
    """Stringa letterale"""
    value: str

@dataclass
class BooleanLiteral(ASTNode):
    """Valore booleano"""
    value: bool

@dataclass
class Variable(ASTNode):
    """Riferimento a variabile"""
    name: str

# ===================== EXPRESSIONS =====================

@dataclass
class BinaryOp(ASTNode):
    """Operazione binaria (+, -, *, /, and, or, etc.)"""
    operator: str
    left: Any
    right: Any

@dataclass
class UnaryOp(ASTNode):
    """Operazione unaria (-, not, etc.)"""
    operator: str
    operand: Any

@dataclass
class Assignment(ASTNode):
    """Assegnazione di variabile"""
    variable: str
    expression: Any

@dataclass
class HistoricalAccess(ASTNode):
    """Accesso a dati storici (Close[1], High[2], etc.)"""
    base_name: str
    index: Any

@dataclass
class FunctionCall(ASTNode):
    """Chiamata di funzione"""
    name: str
    args: List[Any]

# ===================== CONTROL FLOW =====================

@dataclass
class IfStatement(ASTNode):
    """Statement If...Then...Else"""
    condition: Any
    then_block: 'Block'
    else_block: Optional['Block'] = None

@dataclass
class ForLoop(ASTNode):
    """Loop For...To/DownTo"""
    variable: str
    start_value: Any
    end_value: Any
    direction: str = 'to'  # 'to' o 'downto'
    body: Optional['Block'] = None

@dataclass  
class WhileLoop(ASTNode):
    """Loop While"""
    condition: Any
    body: Optional['Block'] = None

@dataclass
class RepeatUntilLoop(ASTNode):
    """Loop Repeat...Until"""
    condition: Any
    body: Optional['Block'] = None

@dataclass
class SwitchCase(ASTNode):
    """Statement Switch/Case"""
    expression: Any
    cases: List['CaseClause']

@dataclass
class CaseClause(ASTNode):
    """Clausola Case singola"""
    values: List[Any]
    statements: List[Any]

@dataclass
class DefaultClause(ASTNode):
    """Clausola Default"""
    statements: List[Any]

@dataclass
class OnceStatement(ASTNode):
    """Statement Once (eseguito una sola volta)"""
    statements: List[Any]

# ===================== DECLARATIONS =====================

@dataclass
class InputsDeclaration(ASTNode):
    """Dichiarazione Inputs"""
    declarations: List['VariableDeclaration']

@dataclass
class VariablesDeclaration(ASTNode):
    """Dichiarazione Variables"""
    declarations: List['VariableDeclaration']

@dataclass
class VariableDeclaration(ASTNode):
    """Dichiarazione singola variabile"""
    name: str
    initial_value: Any
    data_type: Optional[str] = None

@dataclass
class ArrayDeclaration(ASTNode):
    """Dichiarazione array"""
    name: str
    dimensions: List[Any]
    initial_value: Any
    data_type: Optional[str] = None

# ===================== TRADING ORDERS =====================

@dataclass
class OrderStatement(ASTNode):
    """Statement per ordini di trading"""
    verb: str  # 'buy', 'sell', 'sellshort', 'buytocover', etc.
    signal_name: Optional[str] = None
    size: Optional['TradeSize'] = None
    order_type: str = 'market'
    price: Optional[Any] = None
    stop_value: Optional[Any] = None

@dataclass
class TradeSize(ASTNode):
    """Specifica dimensione trade"""
    amount: Any = 1
    unit: str = 'contracts'  # 'contracts', 'shares', 'lots'

@dataclass
class OrderSpec(ASTNode):
    """Specifica esecuzione ordine"""
    order_type: str = 'market'  # 'market', 'limit', 'stop'
    price: Optional[Any] = None
    on_close: bool = False

@dataclass
class BracketOrder(ASTNode):
    """Ordine Bracket (entry + stop + target)"""
    verb: str  # 'buy', 'sell'
    shares: Optional[Any] = None
    entry_price: Optional[Any] = None
    stop_loss: Optional[Any] = None
    profit_target: Optional[Any] = None
    signal_name: Optional[str] = None

# ===================== STOP/TARGET MANAGEMENT =====================

@dataclass
class StopStatement(ASTNode):
    """Statement per stop loss e profit target"""
    stop_type: str  # 'SetStopLoss', 'SetProfitTarget', etc.
    value: Optional[Any] = None

@dataclass
class TrailingStopOrder(ASTNode):
    """Ordine Trailing Stop"""
    amount: Any
    trail_type: str = 'dollar'  # 'dollar' o 'percent'

@dataclass
class SetStopPositionStatement(ASTNode):
    """Statement SetStopPosition"""
    arguments: List[Any]

# ===================== ALERT SYSTEM =====================

@dataclass
class AlertStatement(ASTNode):
    """Statement Alert"""
    message: Optional[Any] = None

@dataclass
class PlaySoundStatement(ASTNode):
    """Statement PlaySound"""
    sound_file: Optional[str] = None

# ===================== FILE I/O OPERATIONS =====================

@dataclass
class FileOperation(ASTNode):
    """Operazione su file"""
    operation: str  # 'print', 'append', 'delete', 'read'
    filename: Any
    arguments: Optional[List[Any]] = None
    text: Optional[Any] = None

@dataclass
class FileCall(ASTNode):
    """Chiamata File()"""
    filename: Any

@dataclass
class PrintStatement(ASTNode):
    """Statement Print (con o senza file)"""
    arguments: List[Any]

# ===================== DRAWING OBJECTS =====================

@dataclass
class DrawingObject(ASTNode):
    """Oggetto di disegno"""
    object_type: str  # 'text_new', 'tl_new', etc.
    date: Optional[Any] = None
    time: Optional[Any] = None
    price: Optional[Any] = None
    text: Optional[Any] = None
    date1: Optional[Any] = None
    time1: Optional[Any] = None
    price1: Optional[Any] = None
    date2: Optional[Any] = None
    time2: Optional[Any] = None
    price2: Optional[Any] = None
    color: Optional[Any] = None
    style: Optional[Any] = None

# ===================== ARRAY OPERATIONS =====================

@dataclass
class ArraySetMaxIndex(ASTNode):
    """Array_SetMaxIndex statement"""
    array_name: str
    max_index: Any

@dataclass
class ArrayFunction(ASTNode):
    """Funzioni per array"""
    function_name: str  # 'SummationArray', 'AverageArray', etc.
    array_ref: str
    size: Optional[Any] = None

# ===================== MULTI-DATA ANALYSIS =====================

@dataclass
class MultiDataAccess(ASTNode):
    """Accesso a dati multipli"""
    data_number: int
    series_name: str  # 'close', 'open', 'high', 'low', 'volume'
    bars_back: Optional[Any] = None

# ===================== ADVANCED EXPRESSIONS =====================

@dataclass
class CrossesAbove(ASTNode):
    """Operatore CrossesAbove"""
    left: Any
    right: Any

@dataclass
class CrossesBelow(ASTNode):
    """Operatore CrossesBelow"""
    left: Any
    right: Any

@dataclass
class BarStateFunction(ASTNode):
    """Funzioni stato barra"""
    function_name: str  # 'InsideBar', 'OutsideBar', 'UpBar', 'DownBar'
    bars_back: Optional[Any] = None

# ===================== MATHEMATICAL FUNCTIONS =====================

@dataclass
class MathFunction(ASTNode):
    """Funzioni matematiche avanzate"""
    function_name: str  # 'AbsValue', 'Square', 'SquareRoot', etc.
    arguments: List[Any]

@dataclass
class IIFFunction(ASTNode):
    """Funzione IIF (Immediate IF)"""
    condition: Any
    true_value: Any
    false_value: Any

# ===================== MARKET DATA FUNCTIONS =====================

@dataclass
class MarketDataFunction(ASTNode):
    """Funzioni per dati di mercato"""
    function_name: str  # 'High', 'Low', 'Close', 'Volume', etc.
    bars_back: Optional[Any] = None
    data_stream: Optional[int] = None

@dataclass
class SeriesFunction(ASTNode):
    """Funzioni per serie temporali"""
    function_name: str  # 'Highest', 'Lowest', 'Average', 'Summation'
    series_ref: Any
    length: Any

# ===================== PERFORMANCE TRACKING =====================

@dataclass
class PerformanceFunction(ASTNode):
    """Funzioni performance tracking"""
    function_name: str  # 'TotalTrades', 'WinningTrades', etc.
    position_back: Optional[Any] = None

@dataclass
class PositionFunction(ASTNode):
    """Funzioni posizione"""
    function_name: str  # 'MarketPosition', 'CurrentContracts', etc.
    args: Optional[List[Any]] = None

# ===================== DATETIME FUNCTIONS =====================

@dataclass
class DateTimeFunction(ASTNode):
    """Funzioni data/ora"""
    function_name: str  # 'Date', 'Time', 'BarNumber', etc.
    bars_back: Optional[Any] = None

# ===================== PLOTTING AND OUTPUT =====================

@dataclass
class PlotStatement(ASTNode):
    """Statement Plot"""
    expression: Any
    plot_name: Optional[str] = None
    color: Optional[Any] = None
    style: Optional[Any] = None

# ===================== COMMENTARY SYSTEM =====================

@dataclass
class CommentaryStatement(ASTNode):
    """Statement Commentary"""
    text: Any
    position: Optional[str] = None  # 'top', 'bottom', 'left', 'right'

# ===================== OPTION ATTRIBUTES =====================

@dataclass
class AttributeStatement(ASTNode):
    """Attributi globali della strategia"""
    attribute_name: str
    value: Any

# ===================== ADVANCED CONDITIONAL =====================

@dataclass
class ConditionalCompilation(ASTNode):
    """Compilazione condizionale"""
    condition_type: str
    condition: Optional[Any] = None
    statements: List[Any] = None

# ===================== ERROR HANDLING =====================

@dataclass
class TryExceptStatement(ASTNode):
    """Statement Try...Except (se supportato)"""
    try_block: Block
    except_block: Block
    exception_type: Optional[str] = None

# ===================== CUSTOM USER FUNCTIONS =====================

@dataclass
class UserFunctionDeclaration(ASTNode):
    """Dichiarazione funzione utente"""
    name: str
    parameters: List[str]
    return_type: Optional[str] = None
    body: Block

@dataclass
class UserFunctionCall(ASTNode):
    """Chiamata funzione utente"""
    name: str
    arguments: List[Any]