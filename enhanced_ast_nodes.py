"""
AST Nodes per EasyLanguage completo
Definisce tutti i nodi dell'Abstract Syntax Tree per supportare
tutte le funzionalità EasyLanguage avanzate.
"""

from typing import List, Optional, Union, Any
from dataclasses import dataclass


@dataclass
class ASTNode:
    """Classe base per tutti i nodi AST"""
    node_type: str = "base"
    line_number: Optional[int] = None
    column: Optional[int] = None


# ============== BASIC NODES ==============

@dataclass
class Expression(ASTNode):
    """Classe base per tutte le espressioni"""
    node_type: str = "expression"


@dataclass
class Statement(ASTNode):
    """Classe base per tutti gli statement"""
    node_type: str = "statement"


@dataclass
class Block(ASTNode):
    """Blocco di statements"""
    statements: List[Statement]
    node_type: str = "block"


@dataclass
class Program(ASTNode):
    """Nodo root del programma"""
    statements: List[Statement]
    node_type: str = "program"


# ============== EXPRESSIONS ==============

@dataclass
class Literal(Expression):
    """Letterali (numeri, stringhe, boolean)"""
    value: Union[int, float, str, bool]
    data_type: str = "auto"
    node_type: str = "literal"


@dataclass
class Identifier(Expression):
    """Identificatori (nomi di variabili, funzioni)"""
    name: str
    node_type: str = "identifier"


@dataclass
class BinaryOperation(Expression):
    """Operazioni binarie (+, -, *, /, =, <>, etc.)"""
    left: Expression
    operator: str
    right: Expression
    node_type: str = "binary_operation"


@dataclass
class UnaryOperation(Expression):
    """Operazioni unarie (-, not, etc.)"""
    operator: str
    operand: Expression
    node_type: str = "unary_operation"


@dataclass
class FunctionCall(Expression):
    """Chiamate a funzione"""
    name: str
    arguments: List[Expression]
    node_type: str = "function_call"


@dataclass
class ArrayAccess(Expression):
    """Accesso agli array: Array[index] o Array[index][historical]"""
    array_name: str
    index: Expression
    historical_reference: Optional[Expression] = None
    node_type: str = "array_access"


@dataclass
class DataReference(Expression):
    """Riferimenti a data stream: Data1, Data2, Data(n)"""
    data_number: Union[int, Expression]
    price_type: str = "close"  # close, open, high, low, volume
    bars_back: Optional[Expression] = None
    node_type: str = "data_reference"


@dataclass
class ConditionalExpression(Expression):
    """Espressione condizionale ternaria: condition ? true_expr : false_expr"""
    condition: Expression
    true_expr: Expression
    false_expr: Expression
    node_type: str = "conditional_expression"


@dataclass
class HistoricalReference(Expression):
    """Riferimento storico: Close[1], High[2], etc."""
    expression: Expression
    bars_back: Expression
    node_type: str = "historical_reference"


@dataclass
class MathFunction(Expression):
    """Funzioni matematiche avanzate"""
    function_name: str
    arguments: List[Expression]
    node_type: str = "math_function"


# ============== STATEMENTS ==============

@dataclass
class Assignment(Statement):
    """Assegnazione di variabile"""
    variable: str
    expression: Expression
    node_type: str = "assignment"


@dataclass
class IfStatement(Statement):
    """Statement IF-THEN-ELSE"""
    condition: Expression
    then_block: Block
    else_block: Optional[Block] = None
    node_type: str = "if_statement"


@dataclass
class OrderStatement(Statement):
    """Statement di ordine (Buy, Sell, etc.)"""
    verb: str  # buy, sell, sellshort, etc.
    signal_name: Optional[str] = None
    shares: Optional[Expression] = None
    price_type: str = "market"  # market, limit, stop, close
    price_value: Optional[Expression] = None
    stop_value: Optional[Expression] = None
    from_entry: Optional[str] = None
    node_type: str = "order_statement"


@dataclass
class PrintStatement(Statement):
    """Statement PRINT"""
    arguments: List[Expression]
    output_type: str = "console"  # console, file, printer
    filename: Optional[Expression] = None
    node_type: str = "print_statement"


@dataclass
class StopStatement(Statement):
    """Statement per stop (SetStopLoss, SetProfitTarget, etc.)"""
    stop_type: str
    value: Optional[Expression] = None
    node_type: str = "stop_statement"


@dataclass
class SetStopPositionStatement(Statement):
    """Statement SetStopPosition"""
    node_type: str = "set_stop_position_statement"


# ============== LOOP STATEMENTS ==============

@dataclass
class ForLoop(Statement):
    """Loop FOR con TO/DOWNTO"""
    variable: str
    start_value: Expression
    end_value: Expression
    direction: str = "to"  # "to" o "downto"
    body: List[Statement]
    node_type: str = "for_loop"


@dataclass
class WhileLoop(Statement):
    """Loop WHILE"""
    condition: Expression
    body: List[Statement]
    node_type: str = "while_loop"


@dataclass
class RepeatUntilLoop(Statement):
    """Loop REPEAT-UNTIL"""
    body: List[Statement]
    condition: Expression
    node_type: str = "repeat_until_loop"


# ============== SWITCH/CASE STATEMENTS ==============

@dataclass
class CaseCondition(ASTNode):
    """Condizione per Case statement"""
    value: Optional[Expression] = None
    range: bool = False
    start: Optional[Expression] = None
    end: Optional[Expression] = None
    node_type: str = "case_condition"


@dataclass
class CaseStatement(ASTNode):
    """Statement CASE individuale"""
    conditions: List[CaseCondition] = None
    body: List[Statement] = None
    is_default: bool = False
    has_break: bool = False
    node_type: str = "case_statement"
    
    def __post_init__(self):
        if self.conditions is None:
            self.conditions = []
        if self.body is None:
            self.body = []


@dataclass
class SwitchCase(Statement):
    """Statement SWITCH completo"""
    expression: Expression
    cases: List[CaseStatement]
    node_type: str = "switch_case"


# ============== ONCE STATEMENT ==============

@dataclass
class OnceStatement(Statement):
    """Statement ONCE"""
    condition: Expression
    body: List[Statement]
    node_type: str = "once_statement"


# ============== ALERT SYSTEM ==============

@dataclass
class AlertStatement(Statement):
    """Statement ALERT"""
    message: Optional[Expression] = None
    condition: Optional[Expression] = None
    sound_file: Optional[Expression] = None
    node_type: str = "alert_statement"


# ============== FILE I/O OPERATIONS ==============

@dataclass
class FileOperation(Statement):
    """Operazioni File I/O"""
    operation: str  # print_to_file, file_append, file_delete
    filename: Expression
    arguments: Optional[List[Expression]] = None
    text: Optional[Expression] = None
    node_type: str = "file_operation"


# ============== DRAWING OBJECTS ==============

@dataclass
class DrawingObject(Statement):
    """Oggetti di disegno (Text, Trendline)"""
    object_type: str  # text_new, trendline_new
    # Per Text
    date: Optional[Expression] = None
    time: Optional[Expression] = None
    price: Optional[Expression] = None
    text: Optional[Expression] = None
    color: Optional[Expression] = None
    # Per Trendline
    date1: Optional[Expression] = None
    time1: Optional[Expression] = None
    price1: Optional[Expression] = None
    date2: Optional[Expression] = None
    time2: Optional[Expression] = None
    price2: Optional[Expression] = None
    node_type: str = "drawing_object"


# ============== ARRAY OPERATIONS ==============

@dataclass
class ArraySetMaxIndex(Statement):
    """Array_SetMaxIndex statement"""
    array_name: str
    max_index: Expression
    node_type: str = "array_set_max_index"


@dataclass
class ArrayFunction(Expression):
    """Funzioni array (SummationArray, AverageArray, etc.)"""
    function_name: str
    array_ref: str
    size: Expression
    ascending: Optional[bool] = None  # Per SortArray
    node_type: str = "array_function"


# ============== ENHANCED STATEMENTS ==============

@dataclass
class EnhancedPrintStatement(PrintStatement):
    """Print statement esteso"""
    output_type: str = "console"  # console, file, printer
    node_type: str = "enhanced_print_statement"


@dataclass
class EnhancedOrderStatement(OrderStatement):
    """Order statement esteso con supporto completo"""
    order_type: str = "buy"  # buy, sell, sellshort, etc.
    signal_name: Optional[str] = None
    shares: Optional[int] = None
    price_type: str = "market"
    price_value: Optional[Expression] = None
    from_entry: Optional[str] = None
    node_type: str = "enhanced_order_statement"


# ============== DECLARATION NODES ==============

@dataclass
class InputsDeclaration(Statement):
    """Dichiarazione inputs"""
    inputs: dict  # nome -> valore_default
    node_type: str = "inputs_declaration"


@dataclass
class VariablesDeclaration(Statement):
    """Dichiarazione variabili"""
    variables: dict  # nome -> tipo/valore
    node_type: str = "variables_declaration"


@dataclass
class ArrayDeclaration(Statement):
    """Dichiarazione array"""
    arrays: dict  # nome -> info_array
    node_type: str = "array_declaration"


# ============== SPECIAL NODES ==============

@dataclass
class CommentNode(ASTNode):
    """Nodo per commenti"""
    text: str
    is_line_comment: bool = False
    node_type: str = "comment"


@dataclass
class PlotStatement(Statement):
    """Statement PLOT"""
    plot_number: int
    expression: Expression
    plot_name: Optional[str] = None
    color: Optional[str] = None
    node_type: str = "plot_statement"


@dataclass
class OptionStatement(Statement):
    """Statement di opzione (MaxBarsBack, etc.)"""
    option_name: str
    value: Expression
    node_type: str = "option_statement"


# ============== UTILITY FUNCTIONS ==============

def create_binary_operation(left: Expression, operator: str, right: Expression) -> BinaryOperation:
    """Helper per creare operazioni binarie"""
    return BinaryOperation(left=left, operator=operator, right=right)


def create_function_call(name: str, *args: Expression) -> FunctionCall:
    """Helper per creare chiamate a funzione"""
    return FunctionCall(name=name, arguments=list(args))


def create_if_statement(condition: Expression, then_statements: List[Statement], 
                       else_statements: Optional[List[Statement]] = None) -> IfStatement:
    """Helper per creare statement IF"""
    then_block = Block(statements=then_statements)
    else_block = Block(statements=else_statements) if else_statements else None
    return IfStatement(condition=condition, then_block=then_block, else_block=else_block)


def create_assignment(variable: str, expression: Expression) -> Assignment:
    """Helper per creare assegnazioni"""
    return Assignment(variable=variable, expression=expression)


def create_literal(value: Union[int, float, str, bool]) -> Literal:
    """Helper per creare letterali"""
    if isinstance(value, bool):
        data_type = "boolean"
    elif isinstance(value, int):
        data_type = "integer"
    elif isinstance(value, float):
        data_type = "numeric"
    elif isinstance(value, str):
        data_type = "string"
    else:
        data_type = "auto"
    
    return Literal(value=value, data_type=data_type)


def create_identifier(name: str) -> Identifier:
    """Helper per creare identificatori"""
    return Identifier(name=name)
