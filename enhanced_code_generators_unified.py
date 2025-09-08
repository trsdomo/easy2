"""
Enhanced Code Generators per EasyLanguage - Versione Unificata
Modulo completo per la generazione del codice Python/Backtrader
da AST EasyLanguage con supporto per tutte le funzionalità avanzate.
"""

import math
from typing import List, Dict, Any, Optional
from .enhanced_ast_nodes import *


class EnhancedCodeGenerators:
    """
    Generatore di codice completo e moderno per EasyLanguage
    Supporta tutte le funzionalità: loop, switch/case, once, alert, 
    file I/O, drawing objects, array operations avanzate, multi-data analysis
    """
    
    def __init__(self, compiler_instance):
        self.compiler = compiler_instance
        self.stop_loss_commands = []
        self.profit_target_commands = []
        self.position_tracking = {}
        self.loop_counters = {}
        self.once_variables = set()
        self.drawing_counter = 0
        self.alert_counter = 0
        self.file_counter = 0

    # ===================== INIT CODE GENERATION =====================

    def generate_init_code(self):
        """Genera il codice per il metodo __init__ della strategia."""
        self.compiler._compilation_context = 'init'
        
        # Riferimenti ai dati base
        self._add_data_references()
        
        # Sistema EasyTrader e inizializzazioni
        self._add_easytrader_system()
        
        # Sistemi di tracking e controllo
        self._add_position_tracking_init()
        self._add_stops_init()
        
        # Inizializzazioni specifiche
        self._initialize_user_variables()
        self._initialize_arrays()
        self._initialize_indicators()
        self._setup_plotting()
        self._setup_stops_and_targets()
        self._setup_dynamic_indicators()

    def _add_data_references(self):
        """Aggiunge riferimenti ai dati principali"""
        self.compiler.init_code.extend([
            "# Riferimenti ai dati principali",
            "self.dataclose = self.datas[0].close",
            "self.dataopen = self.datas[0].open", 
            "self.datahigh = self.datas[0].high",
            "self.datalow = self.datas[0].low",
            "self.datavolume = self.datas[0].volume"
        ])

    def _add_easytrader_system(self):
        """Aggiunge istanza EasyTrader e inizializzazione sistemi"""
        self.compiler.init_code.extend([
            "",
            "# Sistema EasyTrader e inizializzazioni",
            "self.et_functions_instance = EasyTraderFunctions(self)",
            "self._init_easylanguage_systems()",
            "",
            "# Variabili di stato per la logica EasyLanguage", 
            "self.entry_bar_len = 0",
            "self.signal_names = {}"
        ])

    def _add_position_tracking_init(self):
        """Aggiunge inizializzazione tracking posizioni"""
        self.compiler.init_code.extend([
            "",
            "# Sistema di tracking delle posizioni EasyLanguage",
            "self.market_position = 0  # 0=Flat, 1=Long, -1=Short",
            "self.position_entry_price = 0.0",
            "self.position_entry_bar = 0", 
            "self.bars_since_entry = 0",
            "self.bars_since_exit = 0",
            "self.current_contracts = 0",
            "self.max_contracts_held = 0",
            "self.position_profit = 0.0",
            "self.open_position_profit = 0.0",
            "self.total_trades = 0",
            "self.winning_trades = 0",
            "self.losing_trades = 0"
        ])

    def _add_stops_init(self):
        """Aggiunge inizializzazione sistema stop"""
        self.compiler.init_code.extend([
            "",
            "# Sistema di Stop Loss e Profit Target",
            "self.set_dollar_trailing_stop = None",
            "self.set_percent_trailing_stop = None", 
            "self.set_stop_loss = None",
            "self.set_profit_target = None",
            "self.set_break_even = None"
        ])

    def _initialize_user_variables(self):
        """Inizializza le variabili utente."""
        self.compiler.init_code.append("\n# Inizializzazione delle variabili utente")
        
        # Variabili definite dall'utente
        for name, var_data in self.compiler.variables.items():
            if hasattr(var_data, 'initial_value'):
                value = self.compiler.expression_compiler.compile_expression(var_data.initial_value)
            else:
                value = self.compiler.expression_compiler.compile_expression(var_data.get('initial_value', 0))
            self.compiler.init_code.append(f"self.{name} = {value}")

        # Variabili standard EasyLanguage
        self._add_standard_variables()

    def _add_standard_variables(self):
        """Aggiunge variabili standard EasyLanguage"""
        standard_vars = {
            'condition': ['Condition1', 'Condition2', 'Condition3', 'Condition4'],
            'value': ['Value1', 'Value2', 'Value3', 'Value4', 'Value5', 
                     'Value6', 'Value7', 'Value8', 'Value9', 'Value10'],
            'common': ['BarsSinceEntry', 'BarsSinceExit', 'EntryBar', 'ExitBar']
        }
        
        for var_type, vars_list in standard_vars.items():
            for var_name in vars_list:
                if var_name not in self.compiler.variables:
                    default_val = "False" if var_type == 'condition' else "0"
                    self.compiler.init_code.append(f"self.{var_name} = {default_val}")

    def _initialize_arrays(self):
        """Inizializza gli array."""
        if not (hasattr(self.compiler, 'arrays') and self.compiler.arrays):
            return
            
        self.compiler.init_code.append("\n# Inizializzazione degli array")
        for name, array_data in self.compiler.arrays.items():
            dims, init_val = self._extract_array_info(array_data)
            
            if isinstance(dims, (list, tuple)) and len(dims) == 1:
                size = self.compiler.expression_compiler.compile_expression(dims[0])
                compiled_init = self.compiler.expression_compiler.compile_expression(init_val)
                self.compiler.init_code.append(f"self.{name} = [{compiled_init}] * {size}")
            else:
                self.compiler.init_code.append(f"self.{name} = 0  # Array multidimensionale non supportato")

    def _extract_array_info(self, array_data):
        """Estrae informazioni dimensione e valore iniziale dell'array"""
        if isinstance(array_data, dict):
            return array_data.get('dimensions', [10]), array_data.get('initial_value', 0)
        return [10], 0  # defaults

    def _initialize_indicators(self):
        """Inizializza gli indicatori."""
        if not (hasattr(self.compiler, 'init_assignments') and self.compiler.init_assignments):
            return
            
        self.compiler.init_code.append("\n# Calcolo degli indicatori")
        for var_name, value_node in self.compiler.init_assignments.items():
            core_node = self._extract_core_indicator_node(value_node)
            indicator_code = self.compiler.expression_compiler.compile_expression(core_node)
            self.compiler.init_code.append(f"self.{var_name} = {indicator_code}")

    def _extract_core_indicator_node(self, value_node):
        """Estrae il nodo core dell'indicatore"""
        if isinstance(value_node, tuple) and value_node[0] == 'historical_access':
            return value_node[1]
        return value_node

    def _setup_plotting(self):
        """Configura le informazioni di plotting."""
        if not (hasattr(self.compiler, 'plot_info') and self.compiler.plot_info):
            return
            
        self.compiler.init_code.append("\n# Assegnazione delle informazioni di plotting")
        for var_name, plot_name in self.compiler.plot_info.items():
            self.compiler.init_code.extend([
                f"if hasattr(self, '{var_name}'):",
                f"    getattr(self, '{var_name}').plotinfo.plotname = '{plot_name}'"
            ])

    def _setup_stops_and_targets(self):
        """Configura stop loss e profit target."""
        if hasattr(self.compiler, 'stop_loss') and self.compiler.stop_loss is not None:
            self.compiler.init_code.extend([
                "\n# Stop Loss globale",
                f"self.stop_loss_val = {self.compiler.expression_compiler.compile_expression(self.compiler.stop_loss)}"
            ])
        
        if hasattr(self.compiler, 'profit_target') and self.compiler.profit_target is not None:
            self.compiler.init_code.extend([
                "\n# Profit Target globale", 
                f"self.profit_target_val = {self.compiler.expression_compiler.compile_expression(self.compiler.profit_target)}"
            ])

    def _setup_dynamic_indicators(self):
        """Configura indicatori dinamici."""
        if not (hasattr(self.compiler, 'dynamic_indicators') and self.compiler.dynamic_indicators):
            return
            
        self.compiler.init_code.append("\n# Indicatori di condizione dinamici (es. Crossovers)")
        for name, code in self.compiler.dynamic_indicators.items():
            self.compiler.init_code.append(f"self.{name} = {code}")

    # ===================== NEXT CODE GENERATION =====================

    def generate_next_code(self):
        """Genera il corpo del metodo next() processando il resto dell'AST."""
        self.compiler._compilation_context = 'next'
        
        # Aggiorna variabili di posizione
        self._add_position_updates()
        
        # Safety checks per indicatori
        self._add_safety_checks()
        
        # Processa statements dell'AST
        self._process_ast_statements()

    def _add_position_updates(self):
        """Aggiunge aggiornamento variabili di posizione"""
        self.compiler.next_code.extend([
            "        # Aggiorna le variabili di posizione EasyLanguage",
            "        self._update_position_variables()",
            ""
        ])

    def _add_safety_checks(self):
        """Aggiunge controlli di sicurezza per indicatori"""
        if hasattr(self.compiler, 'init_assignments') and self.compiler.init_assignments:
            indicator_names = list(self.compiler.init_assignments.keys())
            
            # Check base per dati sufficienti
            self.compiler.next_code.extend([
                "        if len(self.datas[0]) < 2:",
                "            return",
                ""
            ])
            
            # Check esistenza indicatori
            for name in indicator_names:
                self.compiler.next_code.extend([
                    f"        if not hasattr(self, '{name}') or len(self.{name}) == 0:",
                    "            return"
                ])
            self.compiler.next_code.append("")
            
            # Check validità indicatori
            for name in indicator_names:
                self.compiler.next_code.extend([
                    f"        try:",
                    f"            _ = self.{name}[0]",
                    f"        except (IndexError, TypeError):",
                    "            return"
                ])
            self.compiler.next_code.append("")
        else:
            self.compiler.next_code.extend([
                "        if len(self.datas[0]) < 2:",
                "            return",
                ""
            ])

    def _process_ast_statements(self):
        """Processa gli statement dell'AST"""
        ast_statements = self._extract_ast_statements()
        main_block = [node for node in ast_statements if self._should_include_in_main_block(node)]
        
        compiled_statements = self.compile_statement_block(main_block, indent_level=2)
        self.compiler.next_code.extend(compiled_statements)

    def _extract_ast_statements(self):
        """Estrae gli statement dall'AST"""
        if len(self.compiler.ast) == 1 and isinstance(self.compiler.ast[0], Program):
            return self.compiler.ast[0].statements
        return self.compiler.ast

    def _should_include_in_main_block(self, node) -> bool:
        """Determina se un nodo deve essere incluso nel main block"""
        excluded_types = ['inputs_decl', 'vars_decl', 'array_decl', 'assignment', 'option', 'attribute']
        
        if hasattr(node, 'node_type'):
            if node.node_type not in excluded_types:
                return True
            # Include assignments che non sono in init_assignments  
            if node.node_type == 'assignment':
                return hasattr(node, 'variable') and node.variable not in getattr(self.compiler, 'init_assignments', {})
        elif isinstance(node, tuple) and len(node) >= 1:
            return node[0] not in excluded_types
        
        return True

    # ===================== STATEMENT COMPILATION =====================

    def compile_statement_block(self, block, indent_level=2):
        """Compila una lista di nodi statement con supporto completo."""
        code_lines = []
        statements = self._extract_statements(block)

        for node in statements:
            try:
                compiled_lines = self._compile_single_statement(node, indent_level)
                if compiled_lines:
                    code_lines.extend(compiled_lines)
            except Exception as e:
                indent = "    " * indent_level
                code_lines.append(f"{indent}# Errore nella compilazione: {str(e)}")

        return code_lines if code_lines else [f"{'    ' * indent_level}pass"]

    def _extract_statements(self, block):
        """Estrae gli statement da un blocco"""
        if hasattr(block, 'node_type') and block.node_type == 'program':
            return block.statements
        elif hasattr(block, 'statements'):
            return block.statements
        elif isinstance(block, list):
            return block
        else:
            return [block]

    def _compile_single_statement(self, node, indent_level) -> List[str]:
        """Compila un singolo statement"""
        if hasattr(node, 'node_type'):
            return self._compile_node_statement(node, indent_level)
        elif isinstance(node, tuple) and len(node) >= 1:
            return self._compile_tuple_statement(node, indent_level)
        return []

    def _compile_node_statement(self, node, indent_level) -> List[str]:
        """Compila statement da nodi strutturati"""
        node_type = node.node_type
        
        # Statement base
        base_handlers = {
            'assignment': self._compile_assignment,
            'if_statement': self._compile_if_statement_new,
            'order_statement': self._compile_order,
            'print_statement': self._compile_print_statement,
            'stop_statement': self._compile_stop_statement,
            'set_stop_position_statement': self._compile_set_stop_position,
            'block': lambda n, i: self.compile_statement_block(n.statements, i)
        }
        
        # Statement avanzati
        advanced_handlers = {
            'for_loop': self._compile_for_loop,
            'while_loop': self._compile_while_loop,
            'repeat_until_loop': self._compile_repeat_until_loop,
            'switch_case': self._compile_switch_case,
            'once_statement': self._compile_once_statement,
            'alert_statement': self._compile_alert_statement,
            'file_operation': self._compile_file_operation,
            'drawing_object': self._compile_drawing_object,
            'array_set_max_index': self._compile_array_set_max_index
        }
        
        all_handlers = {**base_handlers, **advanced_handlers}
        
        if node_type in all_handlers:
            handler = all_handlers[node_type]
            result = handler(node, indent_level)
            return result if isinstance(result, list) else [result]
        
        return []

    def _compile_tuple_statement(self, node, indent_level) -> List[str]:
        """Compila statement da tuple (formato legacy)"""
        node_type = node[0]
        
        tuple_handlers = {
            'if_statement': self._compile_if_statement_tuple,
            'order': self._compile_order_tuple
        }
        
        if node_type in tuple_handlers:
            return tuple_handlers[node_type](node, indent_level)
        
        return []

    # ===================== BASIC COMPILATION METHODS =====================

    def _compile_assignment(self, node, indent_level):
        """Compila un nodo Assignment"""
        indent = "    " * indent_level
        var_name = str(node.variable)
        expr_code = self.compiler.expression_compiler.compile_expression(node.expression)
        return f"{indent}self.{var_name} = {expr_code}"

    def _compile_if_statement_new(self, node, indent_level):
        """Compila un nodo IfStatement strutturato"""
        indent = "    " * indent_level
        condition_code = self.compiler.expression_compiler.compile_expression(node.condition)

        code_lines = [f"{indent}if {condition_code}:"]

        # Compila il blocco then
        then_statements = self.compile_statement_block(node.then_block.statements, indent_level + 1)
        code_lines.extend(then_statements)

        # Compila il blocco else se presente
        if node.else_block and node.else_block.statements:
            code_lines.append(f"{indent}else:")
            else_statements = self.compile_statement_block(node.else_block.statements, indent_level + 1)
            code_lines.extend(else_statements)

        return code_lines

    def _compile_if_statement_tuple(self, node, indent_level):
        """Compila statement if (formato tuple legacy)"""
        indent = "    " * indent_level
        condition_code = self.compiler.expression_compiler.compile_expression(node[1])
        
        then_block_code = self.compile_statement_block(node[2][1], indent_level + 1)
        
        code_lines = [f"{indent}if {condition_code}:"]
        code_lines.extend(then_block_code)
        
        if len(node) > 3 and node[3] is not None:
            else_block_code = self.compile_statement_block(node[3][1], indent_level + 1)
            code_lines.append(f"{indent}else:")
            code_lines.extend(else_block_code)
            
        return code_lines

    def _compile_order(self, node, indent_level):
        """Compila tutti i tipi di ordini supportati da EasyLanguage."""
        indent = "    " * indent_level
        
        # Estrai parametri ordine
        order_params = self._extract_order_parameters(node)
        
        # Genera codice in base al tipo di ordine
        code_lines = []
        comment = f"  # Signal: {order_params['signal_name']}"
        
        if order_params['verb'] in ['buy', 'buy_to_open']:
            code_lines.extend(self._generate_buy_order_code(order_params, indent, comment))
        elif order_params['verb'] in ['sellshort', 'sell_short', 'sell_to_open']:
            code_lines.extend(self._generate_sell_short_order_code(order_params, indent, comment))
        elif order_params['verb'] in ['sell', 'sell_to_close']:
            code_lines.extend(self._generate_sell_order_code(order_params, indent, comment))
        elif order_params['verb'] in ['buytocover', 'buy_to_cover', 'cover']:
            code_lines.extend(self._generate_cover_order_code(order_params, indent, comment))
        elif order_params['verb'] in ['exitlong', 'exitshort', 'exitposition']:
            code_lines.extend(self._generate_exit_order_code(order_params, indent, comment))
        else:
            code_lines.append(f"{indent}# Verbo dell'ordine sconosciuto: {order_params['verb']}")

        return code_lines

    def _extract_order_parameters(self, node):
        """Estrae parametri da un nodo ordine"""
        params = {
            'verb': None,
            'signal_name': None,
            'shares': "100",
            'price_type': "market",
            'price_value': None,
            'stop_value': None
        }
        
        if hasattr(node, 'node_type') and node.node_type == 'order_statement':
            params['verb'] = str(node.verb).lower() if hasattr(node, 'verb') else None
            
            if hasattr(node, 'signal_name'):
                if hasattr(node.signal_name, 'value'):
                    params['signal_name'] = node.signal_name.value
                else:
                    params['signal_name'] = str(node.signal_name)
            
            if hasattr(node, 'shares') and node.shares:
                params['shares'] = self.compiler.expression_compiler.compile_expression(node.shares)
            
            if hasattr(node, 'price_type'):
                params['price_type'] = str(node.price_type).lower()
                if hasattr(node, 'price_value') and node.price_value:
                    params['price_value'] = self.compiler.expression_compiler.compile_expression(node.price_value)
            
            if hasattr(node, 'stop_value') and node.stop_value:
                params['stop_value'] = self.compiler.expression_compiler.compile_expression(node.stop_value)
        
        if not params['signal_name']:
            params['signal_name'] = f"Signal_{getattr(self.compiler, 'order_counter', 1)}"
            if hasattr(self.compiler, 'order_counter'):
                self.compiler.order_counter += 1
        
        return params

    def _generate_buy_order_code(self, params, indent, comment):
        """Genera codice per ordini di acquisto Long"""
        code_lines = [f"{indent}if self.market_position <= 0:"]
        
        if params['price_type'] == "market":
            code_lines.append(f"{indent}    self.buy(size={params['shares']}){comment}")
        elif params['price_type'] in ["limit", "close"]:
            price = params['price_value'] if params['price_value'] else "self.dataclose[0]"
            code_lines.append(f"{indent}    self.buy(size={params['shares']}, exectype=bt.Order.Limit, price={price}){comment}")
        elif params['price_type'] in ["stop", "stop_market"]:
            price = params['price_value'] if params['price_value'] else "self.datahigh[0]"
            code_lines.append(f"{indent}    self.buy(size={params['shares']}, exectype=bt.Order.Stop, price={price}){comment}")
        
        return code_lines

    def _generate_sell_short_order_code(self, params, indent, comment):
        """Genera codice per ordini di vendita allo scoperto"""
        code_lines = [f"{indent}if self.market_position >= 0:"]
        
        if params['price_type'] == "market":
            code_lines.append(f"{indent}    self.sell(size={params['shares']}){comment}")
        elif params['price_type'] in ["limit", "close"]:
            price = params['price_value'] if params['price_value'] else "self.dataclose[0]"
            code_lines.append(f"{indent}    self.sell(size={params['shares']}, exectype=bt.Order.Limit, price={price}){comment}")
        elif params['price_type'] in ["stop", "stop_market"]:
            price = params['price_value'] if params['price_value'] else "self.datalow[0]"
            code_lines.append(f"{indent}    self.sell(size={params['shares']}, exectype=bt.Order.Stop, price={price}){comment}")
        
        return code_lines

    def _generate_sell_order_code(self, params, indent, comment):
        """Genera codice per chiusura posizioni Long"""
        code_lines = [f"{indent}if self.market_position > 0:"]
        
        if params['price_type'] == "market":
            code_lines.append(f"{indent}    self.close(){comment}")
        elif params['price_type'] in ["limit", "close"]:
            price = params['price_value'] if params['price_value'] else "self.dataclose[0]"
            code_lines.append(f"{indent}    self.sell(size=self.position.size, exectype=bt.Order.Limit, price={price}){comment}")
        elif params['price_type'] in ["stop", "stop_market"]:
            price = params['price_value'] if params['price_value'] else "self.datalow[0]"
            code_lines.append(f"{indent}    self.sell(size=self.position.size, exectype=bt.Order.Stop, price={price}){comment}")
        
        return code_lines

    def _generate_cover_order_code(self, params, indent, comment):
        """Genera codice per copertura posizioni Short"""
        code_lines = [f"{indent}if self.market_position < 0:"]
        
        if params['price_type'] == "market":
            code_lines.append(f"{indent}    self.close(){comment}")
        elif params['price_type'] in ["limit", "close"]:
            price = params['price_value'] if params['price_value'] else "self.dataclose[0]"
            code_lines.append(f"{indent}    self.buy(size=abs(self.position.size), exectype=bt.Order.Limit, price={price}){comment}")
        elif params['price_type'] in ["stop", "stop_market"]:
            price = params['price_value'] if params['price_value'] else "self.datahigh[0]"
            code_lines.append(f"{indent}    self.buy(size=abs(self.position.size), exectype=bt.Order.Stop, price={price}){comment}")
        
        return code_lines

    def _generate_exit_order_code(self, params, indent, comment):
        """Genera codice per ordini di uscita speciali"""
        if params['verb'] == 'exitlong':
            return [
                f"{indent}if self.market_position > 0:",
                f"{indent}    self.close(){comment}"
            ]
        elif params['verb'] == 'exitshort':
            return [
                f"{indent}if self.market_position < 0:",
                f"{indent}    self.close(){comment}"
            ]
        else:  # exitposition
            return [
                f"{indent}if self.market_position != 0:",
                f"{indent}    self.close(){comment}"
            ]

    def _compile_order_tuple(self, node, indent_level):
        """Compila ordini in formato tuple (legacy)"""
        # Implementazione per compatibilità con formato tuple
        return [f"{'    ' * indent_level}# Order tuple compilation not implemented"]

    def _compile_print_statement(self, node, indent_level):
        """Compila un nodo PrintStatement, ignorando la sintassi File()."""
        indent = "    " * indent_level
        
        if not hasattr(node, 'arguments') or not node.arguments:
            return [f"{indent}print()"]
        
        args_to_compile = node.arguments
        if len(node.arguments) == 1 and isinstance(node.arguments[0], list):
            args_to_compile = node.arguments[0]
        
        compiled_args = []
        for arg in args_to_compile:
            # IGNORA la chiamata a File() ma processa gli altri argomenti
            if isinstance(arg, FunctionCall) and arg.name.lower() == 'file':
                continue
                
            compiled_arg = self.compiler.expression_compiler.compile_expression(arg)
            compiled_args.append(compiled_arg)
        
        if not compiled_args:
            return [f"{indent}pass  # Print statement to file was ignored"]

        args_str = ', '.join(compiled_args)
        return [f"{indent}print({args_str})"]

    def _compile_stop_statement(self, node, indent_level):
        """Compila le istruzioni di stop (SetStopLoss, SetProfitTarget, etc.)"""
        indent = "    " * indent_level
        code_lines = []
        
        if not hasattr(node, 'stop_type'):
            return [f"{indent}# Stop statement senza tipo"]
            
        stop_type = node.stop_type.lower()
        
        stop_handlers = {
            'setstoploss': lambda: self._compile_set_stop_loss(node, indent),
            'setprofittarget': lambda: self._compile_set_profit_target(node, indent),
            'setdollartrailingstop': lambda: self._compile_set_dollar_trailing_stop(node, indent),
            'setpercenttrailingstop': lambda: self._compile_set_percent_trailing_stop(node, indent),
            'setbreakeven': lambda: self._compile_set_break_even(node, indent)
        }
        
        if stop_type in stop_handlers:
            return stop_handlers[stop_type]()
        
        return [f"{indent}# Stop type sconosciuto: {stop_type}"]

    def _compile_set_stop_loss(self, node, indent):
        """Compila SetStopLoss"""
        if hasattr(node, 'value'):
            value = self.compiler.expression_compiler.compile_expression(node.value)
            return [
                f"{indent}self.set_stop_loss = {value}",
                f"{indent}# TODO: Implementare SetStopLoss con valore {value}"
            ]
        return [f"{indent}# SetStopLoss senza valore"]

    def _compile_set_profit_target(self, node, indent):
        """Compila SetProfitTarget"""
        if hasattr(node, 'value'):
            value = self.compiler.expression_compiler.compile_expression(node.value)
            return [
                f"{indent}self.set_profit_target = {value}",
                f"{indent}# TODO: Implementare SetProfitTarget con valore {value}"
            ]
        return [f"{indent}# SetProfitTarget senza valore"]

    def _compile_set_dollar_trailing_stop(self, node, indent):
        """Compila SetDollarTrailingStop"""
        if hasattr(node, 'value'):
            value = self.compiler.expression_compiler.compile_expression(node.value)
            return [
                f"{indent}self.set_dollar_trailing_stop = {value}",
                f"{indent}# TODO: Implementare SetDollarTrailingStop con valore {value}"
            ]
        return [f"{indent}# SetDollarTrailingStop senza valore"]

    def _compile_set_percent_trailing_stop(self, node, indent):
        """Compila SetPercentTrailingStop"""
        if hasattr(node, 'value'):
            value = self.compiler.expression_compiler.compile_expression(node.value)
            return [
                f"{indent}self.set_percent_trailing_stop = {value}",
                f"{indent}# TODO: Implementare SetPercentTrailingStop con valore {value}"
            ]
        return [f"{indent}# SetPercentTrailingStop senza valore"]

    def _compile_set_break_even(self, node, indent):
        """Compila SetBreakEven"""
        return [
            f"{indent}self.set_break_even = True",
            f"{indent}# TODO: Implementare SetBreakEven"
        ]

    def _compile_set_stop_position(self, node, indent_level):
        """Compila un'istruzione SetStopPosition in self.close()."""
        indent = "    " * indent_level
        return [f"{indent}self.close()  # Compiled from SetStopPosition"]

    # ===================== ADVANCED COMPILATION METHODS =====================

    def _compile_for_loop(self, node, indent_level):
        """Compila loop For con To e DownTo"""
        indent = "    " * indent_level
        code_lines = []
        
        var_name = node.variable
        start_val = self.compiler.expression_compiler.compile_expression(node.start_value)
        end_val = self.compiler.expression_compiler.compile_expression(node.end_value)
        
        if hasattr(node, 'direction') and node.direction.lower() == 'downto':
            # For x = 10 downto 1
            code_lines.append(f"{indent}for self.{var_name} in range({start_val}, {end_val} - 1, -1):")
        else:
            # For x = 1 to 10
            code_lines.append(f"{indent}for self.{var_name} in range({start_val}, {end_val} + 1):")
            
        # Compila il blocco del loop
        loop_body = self.compile_statement_block(node.body, indent_level + 1)
        code_lines.extend(loop_body)
        
        return code_lines

    def _compile_while_loop(self, node, indent_level):
        """Compila loop While"""
        indent = "    " * indent_level
        condition = self.compiler.expression_compiler.compile_expression(node.condition)
        
        code_lines = [f"{indent}while {condition}:"]
        
        # Compila il blocco del loop
        loop_body = self.compile_statement_block(node.body, indent_level + 1)
        code_lines.extend(loop_body)
        
        return code_lines

    def _compile_repeat_until_loop(self, node, indent_level):
        """Compila loop Repeat-Until"""
        indent = "    " * indent_level
        condition = self.compiler.expression_compiler.compile_expression(node.condition)
        
        # Repeat-Until viene convertito in while not condition
        code_lines = [
            f"{indent}while True:",
            *self.compile_statement_block(node.body, indent_level + 1),
            f"{indent}    if {condition}:",
            f"{indent}        break"
        ]
        
        return code_lines

    def _compile_switch_case(self, node, indent_level):
        """Compila statement Switch/Case"""
        indent = "    " * indent_level
        switch_var = self.compiler.expression_compiler.compile_expression(node.expression)
        
        code_lines = []
        first_case = True
        
        for case in node.cases:
            if hasattr(case, 'values'):
                # Case normale
                case_values = [self.compiler.expression_compiler.compile_expression(v) for v in case.values]
                condition = f"{switch_var} in [{', '.join(case_values)}]"
                
                if first_case:
                    code_lines.append(f"{indent}if {condition}:")
                    first_case = False
                else:
                    code_lines.append(f"{indent}elif {condition}:")
                    
                case_body = self.compile_statement_block(case.statements, indent_level + 1)
                code_lines.extend(case_body)
            else:
                # Default case
                code_lines.append(f"{indent}else:")
                default_body = self.compile_statement_block(case.statements, indent_level + 1)
                code_lines.extend(default_body)
        
        return code_lines

    def _compile_once_statement(self, node, indent_level):
        """Compila statement Once"""
        indent = "    " * indent_level
        
        # Genera nome variabile unica per questo Once
        once_var = f"_once_{len(self.once_variables)}"
        self.once_variables.add(once_var)
        
        # Aggiunge inizializzazione della variabile once nel __init__
        self.compiler.init_code.append(f"self.{once_var} = False")
        
        code_lines = [
            f"{indent}if not self.{once_var}:",
            f"{indent}    self.{once_var} = True",
            *self.compile_statement_block(node.statements, indent_level + 1)
        ]
        
        return code_lines

    def _compile_alert_statement(self, node, indent_level):
        """Compila statement Alert"""
        indent = "    " * indent_level
        
        if hasattr(node, 'message'):
            message = self.compiler.expression_compiler.compile_expression(node.message)
            return [f"{indent}self.Alert({message})"]
        else:
            return [f"{indent}self.Alert()"]

    def _compile_file_operation(self, node, indent_level):
        """Compila operazioni file"""
        indent = "    " * indent_level
        
        if not hasattr(node, 'operation'):
            return [f"{indent}# File operation senza tipo"]
            
        op_type = node.operation.lower()
        
        if op_type == 'print':
            filename = self.compiler.expression_compiler.compile_expression(node.filename)
            if hasattr(node, 'arguments'):
                args = [self.compiler.expression_compiler.compile_expression(arg) for arg in node.arguments]
                args_str = ', '.join(args)
                return [f"{indent}self.Print_File({filename}, {args_str})"]
            else:
                return [f"{indent}self.Print_File({filename})"]
        elif op_type == 'append':
            filename = self.compiler.expression_compiler.compile_expression(node.filename)
            text = self.compiler.expression_compiler.compile_expression(node.text)
            return [f"{indent}self.FileAppend({filename}, {text})"]
        elif op_type == 'delete':
            filename = self.compiler.expression_compiler.compile_expression(node.filename)
            return [f"{indent}self.FileDelete({filename})"]
        
        return [f"{indent}# File operation sconosciuta: {op_type}"]

    def _compile_drawing_object(self, node, indent_level):
        """Compila oggetti di disegno"""
        indent = "    " * indent_level
        
        if not hasattr(node, 'object_type'):
            return [f"{indent}# Drawing object senza tipo"]
            
        obj_type = node.object_type.lower()
        
        if obj_type == 'text_new':
            params = self._extract_drawing_params(node, ['date', 'time', 'price', 'text'])
            return [f"{indent}self.Text_New({', '.join(params)})"]
        elif obj_type == 'tl_new':
            params = self._extract_drawing_params(node, ['date1', 'time1', 'price1', 'date2', 'time2', 'price2'])
            return [f"{indent}self.TL_New({', '.join(params)})"]
        
        return [f"{indent}# Drawing object sconosciuto: {obj_type}"]

    def _extract_drawing_params(self, node, param_names):
        """Estrae parametri per oggetti di disegno"""
        params = []
        for param_name in param_names:
            if hasattr(node, param_name):
                param_value = self.compiler.expression_compiler.compile_expression(getattr(node, param_name))
                params.append(param_value)
            else:
                params.append("0")  # default
        return params

    def _compile_array_set_max_index(self, node, indent_level):
        """Compila Array_SetMaxIndex"""
        indent = "    " * indent_level
        
        if hasattr(node, 'array_name') and hasattr(node, 'max_index'):
            array_name = str(node.array_name)
            max_index = self.compiler.expression_compiler.compile_expression(node.max_index)
            return [f"{indent}self.Array_SetMaxIndex('{array_name}', {max_index})"]
        
        return [f"{indent}# Array_SetMaxIndex con parametri mancanti"]

    # ===================== STRATEGY CLASS BUILDING =====================

    def build_enhanced_strategy_class(self):
        """Costruisce la classe Strategy completa con tutte le funzionalità EasyLanguage."""

        params_str = '\n'.join([f"        {k}={v}," for k, v in self.compiler.params.items()])
        init_code_str = '\n'.join([f"        {line}" for line in self.compiler.init_code])
        next_code_str = '\n'.join(self.compiler.next_code)

        # Template della classe completa
        template = f'''import backtrader as bt
import math
import numpy as np
from collections import deque
from easytrader.functions.runtime import EasyTraderFunctions

class CompiledStrategy(bt.Strategy):
    """
    Strategia EasyLanguage compilata in Python/Backtrader
    Supporta tutte le funzionalità EasyLanguage avanzate:
    - Loop statements (For/While/Repeat-Until)
    - Switch/Case statements  
    - Once statements
    - Alert system completo
    - File I/O operations
    - Drawing objects
    - Array operations avanzate
    - Multi-data analysis
    - Advanced order types
    """
    
    params = dict(
{params_str}
    )

    def log(self, txt, dt=None):
        """Logging function for this strategy"""
        pass  # Logging disabilitato per output pulito

    def __init__(self):
        self.log("STRATEGY: __init__ called")
        
{init_code_str}
        
        self.order = None

    def _init_easylanguage_systems(self):
        """Inizializza tutti i sistemi EasyLanguage"""
        self._init_loop_protection()
        self._init_alert_system()
        self._init_drawing_objects()
        self._init_multi_data()
        self._init_advanced_orders()
        self._init_performance_tracking()
        self._init_commentary_system()
        self._init_plot_system()
        self._init_file_system()
        self._init_array_system()

    # ============== LOOP PROTECTION SYSTEM ==============
    
    def _init_loop_protection(self):
        """Inizializza sistema di protezione loop"""
        self._loop_counters = {{}}
        self._max_loop_iterations = 10000
    
    def _reset_loop_counter(self, loop_id):
        """Reset counter per un loop specifico"""
        if loop_id in self._loop_counters:
            self._loop_counters[loop_id] = 0
    
    def _increment_loop_counter(self, loop_id):
        """Incrementa e controlla counter loop"""
        if loop_id not in self._loop_counters:
            self._loop_counters[loop_id] = 0
        self._loop_counters[loop_id] += 1
        return self._loop_counters[loop_id] > self._max_loop_iterations

    # ============== ALERT SYSTEM COMPLETO ==============
    
    def _init_alert_system(self):
        """Inizializza il sistema di alert completo"""
        self._alerts_enabled = True
        self._alert_count = 0
        self._last_alert_bar = -1
        self._alert_log = []
        self._alert_sounds = {{"default": "alert.wav"}}
    
    def Alert(self, message="Alert triggered"):
        """Genera un alert con messaggio"""
        current_bar = len(self.data)
        
        # Evita alert multipli sulla stessa barra
        if current_bar == self._last_alert_bar:
            return
            
        self._last_alert_bar = current_bar
        self._alert_count += 1
        
        alert_info = {{
            'bar': current_bar,
            'time': self.data.datetime.datetime(0),
            'message': message,
            'symbol': getattr(self.data, '_name', 'Unknown'),
            'price': self.dataclose[0]
        }}
        
        self._alert_log.append(alert_info)
        
        # Log dell'alert
        self.log(f"ALERT #{{self._alert_count}}: {{message}} at {{alert_info['price']:.2f}}")
        
        return True
    
    def CheckAlert(self):
        """Verifica se gli alert sono abilitati"""
        return self._alerts_enabled
    
    def PlaySound(self, sound_file="default"):
        """Riproduce un suono (placeholder)"""
        self.log(f"SOUND: Would play {{sound_file}}")
        return True
    
    def GetAlertCount(self):
        """Restituisce il numero di alert generati"""
        return self._alert_count

    # ============== DRAWING OBJECTS SYSTEM ==============
    
    def _init_drawing_objects(self):
        """Inizializza il sistema di disegno"""
        self._text_objects = {{}}
        self._trendlines = {{}}
        self._object_counter = 0
    
    def Text_New(self, date, time, price, text_string):
        """Crea un nuovo oggetto testo"""
        self._object_counter += 1
        object_id = self._object_counter
        
        text_obj = {{
            'id': object_id,
            'date': date,
            'time': time,
            'price': price,
            'text': text_string,
            'color': 'Black',
            'size': 12,
            'style': 'Normal'
        }}
        
        self._text_objects[object_id] = text_obj
        self.log(f"TEXT OBJECT {{object_id}}: '{{text_string}}' at {{price}}")
        return object_id
    
    def Text_SetColor(self, text_id, color):
        """Imposta il colore di un oggetto testo"""
        if text_id in self._text_objects:
            self._text_objects[text_id]['color'] = color
            return True
        return False
    
    def TL_New(self, date1, time1, price1, date2, time2, price2):
        """Crea una nuova trendline"""
        self._object_counter += 1
        tl_id = self._object_counter
        
        trendline = {{
            'id': tl_id,
            'date1': date1, 'time1': time1, 'price1': price1,
            'date2': date2, 'time2': time2, 'price2': price2,
            'color': 'Blue',
            'thickness': 1,
            'extend_left': False,
            'extend_right': False
        }}
        
        self._trendlines[tl_id] = trendline
        self.log(f"TRENDLINE {{tl_id}}: ({{price1:.2f}}, {{price2:.2f}})")
        return tl_id
    
    def TL_SetColor(self, tl_id, color):
        """Imposta il colore di una trendline"""
        if tl_id in self._trendlines:
            self._trendlines[tl_id]['color'] = color
            return True
        return False

    # ============== MULTI-DATA ANALYSIS SUPPORT ==============
    
    def _init_multi_data(self):
        """Inizializza supporto per dati multipli"""
        self._data_streams = {{}}
        for i, data in enumerate(self.datas):
            self._data_streams[i] = {{
                'open': data.open,
                'high': data.high, 
                'low': data.low,
                'close': data.close,
                'volume': data.volume,
                'datetime': data.datetime
            }}
    
    def Data(self, data_num):
        """Accesso a stream di dati specifico (Data(1), Data(2), etc.)"""
        if data_num <= len(self.datas):
            return self.datas[data_num - 1]  # EL usa base 1, Python base 0
        return None
    
    def Close_Data(self, data_num, bars_back=0):
        """Close di un data stream specifico"""
        data = self.Data(data_num)
        if data:
            return data.close[-bars_back] if bars_back > 0 else data.close[0]
        return 0
    
    def Open_Data(self, data_num, bars_back=0):
        """Open di un data stream specifico"""
        data = self.Data(data_num)
        if data:
            return data.open[-bars_back] if bars_back > 0 else data.open[0]
        return 0
    
    def High_Data(self, data_num, bars_back=0):
        """High di un data stream specifico"""
        data = self.Data(data_num)
        if data:
            return data.high[-bars_back] if bars_back > 0 else data.high[0]
        return 0
    
    def Low_Data(self, data_num, bars_back=0):
        """Low di un data stream specifico"""
        data = self.Data(data_num)
        if data:
            return data.low[-bars_back] if bars_back > 0 else data.low[0]
        return 0
    
    def Volume_Data(self, data_num, bars_back=0):
        """Volume di un data stream specifico"""
        data = self.Data(data_num)
        if data:
            return data.volume[-bars_back] if bars_back > 0 else data.volume[0]
        return 0

    # ============== ADVANCED ORDER TYPES ==============
    
    def _init_advanced_orders(self):
        """Inizializza sistema ordini avanzati"""
        self._active_brackets = {{}}
        self._oco_orders = {{}}
        self._trailing_stops = {{}}
        self._order_id_counter = 0
    
    def Buy_Bracket(self, shares, entry_price=None, stop_loss=None, profit_target=None, signal_name="BracketBuy"):
        """Ordine Bracket per entrata Long con stop e target automatici"""
        self._order_id_counter += 1
        bracket_id = f"bracket_{{self._order_id_counter}}"
        
        # Piazza ordine di entrata
        if entry_price is None:
            order = self.buy(size=shares)
        else:
            order = self.buy(size=shares, exectype=bt.Order.Limit, price=entry_price)
        
        # Salva informazioni bracket
        self._active_brackets[bracket_id] = {{
            'entry_order': order,
            'shares': shares,
            'stop_loss': stop_loss,
            'profit_target': profit_target,
            'signal_name': signal_name,
            'status': 'pending'
        }}
        
        self.log(f"BRACKET ORDER: {{signal_name}} - Entry: {{entry_price}}, Stop: {{stop_loss}}, Target: {{profit_target}}")
        return bracket_id
    
    def SetTrailingStop(self, amount, trail_type="dollar"):
        """Imposta trailing stop per posizione corrente"""
        if self.market_position != 0:
            trail_id = f"trail_{{self._order_id_counter}}"
            self._order_id_counter += 1
            
            self._trailing_stops[trail_id] = {{
                'amount': amount,
                'type': trail_type,
                'position_size': self.current_contracts,
                'entry_price': self.position_entry_price,
                'high_water_mark': self.position_entry_price if self.market_position > 0 else None,
                'low_water_mark': self.position_entry_price if self.market_position < 0 else None
            }}
            
            self.log(f"TRAILING STOP: {{trail_type}} {{amount}} for {{self.current_contracts}} contracts")
            return trail_id
        return None

    # ============== FILE I/O SYSTEM ==============
    
    def _init_file_system(self):
        """Inizializza il sistema File I/O"""
        self._open_files = {{}}
        self._file_counter = 0
    
    def Print_File(self, filename, *args):
        """Stampa su file (equivalente di Print(File(...), ...)"""
        try:
            text = ' '.join(str(arg) for arg in args) + '\\n'
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(text)
            return True
        except Exception as e:
            self.log(f"Error writing to file {{filename}}: {{e}}")
            return False
    
    def FileAppend(self, filename, text):
        """Aggiunge testo a un file"""
        try:
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(str(text))
            return True
        except Exception as e:
            self.log(f"Error appending to file {{filename}}: {{e}}")
            return False
    
    def FileDelete(self, filename):
        """Elimina un file"""
        try:
            import os
            os.remove(filename)
            return True
        except Exception as e:
            self.log(f"Error deleting file {{filename}}: {{e}}")
            return False

    # ============== ARRAY SYSTEM AVANZATO ==============
    
    def _init_array_system(self):
        """Inizializza il sistema array avanzato"""
        self._dynamic_arrays = {{}}
    
    def Array_SetMaxIndex(self, array_name, max_index):
        """Imposta la dimensione massima di un array dinamico"""
        array_attr = f"_array_{{array_name}}"
        if not hasattr(self, array_attr):
            setattr(self, array_attr, [0] * (max_index + 1))
        else:
            current_array = getattr(self, array_attr)
            if len(current_array) <= max_index:
                current_array.extend([0] * (max_index + 1 - len(current_array)))
        return True
    
    def SummationArray(self, array_ref, size):
        """Somma i primi size elementi di un array"""
        if hasattr(self, array_ref):
            array = getattr(self, array_ref)
            return sum(array[1:min(size+1, len(array))])  # Skip elemento 0
        return 0
    
    def AverageArray(self, array_ref, size):
        """Media dei primi size elementi di un array"""
        total = self.SummationArray(array_ref, size)
        return total / size if size > 0 else 0

    # ============== PERFORMANCE TRACKING ==============
    
    def _init_performance_tracking(self):
        """Inizializza tracking delle performance"""
        self._performance_metrics = {{
            'max_drawdown': 0.0,
            'max_profit': 0.0,
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0
        }}
    
    def _init_commentary_system(self):
        """Inizializza sistema di commentary"""
        self._commentary_enabled = True
        self._commentary_log = []
    
    def _init_plot_system(self):
        """Inizializza sistema di plot"""
        self._plot_objects = {{}}
        self._plot_counter = 0

    def _update_position_variables(self):
        """Aggiorna le variabili che tracciano lo stato della posizione"""
        if self.position:
            self.market_position = 1 if self.position.size > 0 else (-1 if self.position.size < 0 else 0)
            self.current_contracts = abs(self.position.size)
            
            # Calcola profit della posizione aperta
            if self.position.size != 0:
                current_price = self.dataclose[0]
                if self.position.size > 0:  # Long
                    self.open_position_profit = (current_price - self.position.price) * self.position.size
                else:  # Short
                    self.open_position_profit = (self.position.price - current_price) * abs(self.position.size)
            else:
                self.open_position_profit = 0.0
        else:
            self.market_position = 0
            self.current_contracts = 0
            self.open_position_profit = 0.0

    def notify_order(self, order):
        """Gestione nativa degli ordini Backtrader"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED, Price: {{order.executed.price:.2f}}")
                self.position_entry_price = order.executed.price
                self.position_entry_bar = len(self.data)
                self.bars_since_entry = 0
            elif order.issell():
                self.log(f"SELL EXECUTED, Price: {{order.executed.price:.2f}}")
                self.bars_since_exit = 0

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"Order {{order.status}}")

        self.order = None

    def notify_trade(self, trade):
        """Notifica native dei trade completati"""
        if not trade.isclosed:
            return
            
        self.log(f"TRADE PROFIT: {{trade.pnlcomm:.2f}}")
        self.total_trades += 1
        
        if trade.pnlcomm > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
            
        # Aggiorna bars_since_exit
        self.bars_since_exit = 0

    def next(self):
        self.log(f"STRATEGY: next called - Close: {{self.dataclose[0]:.2f}}")
        
        # Aggiorna contatori barre
        if self.market_position != 0:
            self.bars_since_entry += 1
        else:
            self.bars_since_exit += 1
        
        if self.order:
            return
            
{next_code_str}

    # ============== FUNZIONI HELPER EASYLANGUAGE ==============
    
    def MarketPosition(self):
        """Restituisce la posizione corrente: 1=Long, 0=Flat, -1=Short"""
        return self.market_position
    
    def CurrentContracts(self):
        """Restituisce il numero di contratti attualmente detenuti"""
        return self.current_contracts
    
    def EntryPrice(self, pos_back=0):
        """Restituisce il prezzo di entrata della posizione"""
        return self.position_entry_price if pos_back == 0 else self.position_entry_price
    
    def BarsSinceEntry(self):
        """Restituisce il numero di barre dall'ultima entrata"""
        return self.bars_since_entry
    
    def BarsSinceExit(self):
        """Restituisce il numero di barre dall'ultima uscita"""
        return self.bars_since_exit
    
    def OpenPositionProfit(self):
        """Restituisce il profitto della posizione aperta"""
        return self.open_position_profit
    
    def TotalTrades(self):
        """Restituisce il numero totale di trades"""
        return self.total_trades
    
    def WinningTrades(self):
        """Restituisce il numero di trades vincenti"""
        return self.winning_trades
    
    def LosingTrades(self):
        """Restituisce il numero di trades perdenti"""
        return self.losing_trades

    # ============== FUNZIONI STOP/TARGET ==============
    
    def SetStopLoss(self, amount):
        """Imposta uno stop loss"""
        self.set_stop_loss = amount
        self.log(f"SetStopLoss: {{amount}}")
        
    def SetProfitTarget(self, amount):
        """Imposta un profit target"""
        self.set_profit_target = amount
        self.log(f"SetProfitTarget: {{amount}}")
        
    def SetDollarTrailingStop(self, amount):
        """Imposta un trailing stop in dollari"""
        self.set_dollar_trailing_stop = amount
        self.log(f"SetDollarTrailingStop: {{amount}}")
        
    def SetPercentTrailingStop(self, percent):
        """Imposta un trailing stop percentuale"""
        self.set_percent_trailing_stop = percent
        self.log(f"SetPercentTrailingStop: {{percent}}%")
        
    def SetBreakEven(self):
        """Imposta il break even"""
        self.set_break_even = True
        self.log("SetBreakEven activated")

    # ============== FUNZIONI MATEMATICHE AVANZATE ==============
    
    def AbsValue(self, value):
        """Valore assoluto"""
        return abs(value)
    
    def Square(self, value):
        """Quadrato di un numero"""
        return value * value
    
    def SquareRoot(self, value):
        """Radice quadrata"""
        return math.sqrt(abs(value))
    
    def Power(self, base, exponent):
        """Elevazione a potenza"""
        return pow(base, exponent)
    
    def Log(self, value):
        """Logaritmo naturale"""
        return math.log(abs(value)) if value != 0 else 0
    
    def Sine(self, angle):
        """Seno (angolo in radianti)"""
        return math.sin(angle)
    
    def Cosine(self, angle):
        """Coseno (angolo in radianti)"""
        return math.cos(angle)
    
    def Round(self, value, decimals=0):
        """Arrotondamento normale"""
        return round(value, decimals)
    
    def MinList(self, *values):
        """Minimo di una lista di valori"""
        return min(values)
    
    def MaxList(self, *values):
        """Massimo di una lista di valori"""
        return max(values)

    # ============== FUNZIONI DI STATO MERCATO ==============
    
    def LastBarOnChart(self):
        """Restituisce True se questa è l'ultima barra del grafico"""
        return len(self.data) == self.data.buflen()
    
    def BarNumber(self):
        """Restituisce il numero della barra corrente"""
        return len(self.data)
    
    def CurrentBar(self):
        """Restituisce il numero della barra corrente (alias di BarNumber)"""
        return len(self.data)
    
    def Date(self, bars_back=0):
        """Restituisce la data in formato EasyLanguage YYYYMMDD"""
        dt = self.data.datetime.date(-bars_back) if bars_back > 0 else self.data.datetime.date(0)
        return int(dt.strftime('%Y%m%d'))
    
    def Time(self, bars_back=0):
        """Restituisce l'ora in formato EasyLanguage HHMM"""
        dt = self.data.datetime.time(-bars_back) if bars_back > 0 else self.data.datetime.time(0)
        return int(dt.strftime('%H%M'))
    
    def Volume(self, bars_back=0):
        """Restituisce il volume"""
        return self.data.volume[-bars_back] if bars_back > 0 else self.data.volume[0]
    
    def Open(self, bars_back=0):
        """Restituisce il prezzo di apertura"""
        return self.data.open[-bars_back] if bars_back > 0 else self.data.open[0]
    
    def High(self, bars_back=0):
        """Restituisce il prezzo massimo"""
        return self.data.high[-bars_back] if bars_back > 0 else self.data.high[0]
    
    def Low(self, bars_back=0):
        """Restituisce il prezzo minimo"""
        return self.data.low[-bars_back] if bars_back > 0 else self.data.low[0]
    
    def Close(self, bars_back=0):
        """Restituisce il prezzo di chiusura"""
        return self.data.close[-bars_back] if bars_back > 0 else self.data.close[0]

    # ============== FUNZIONI DI ANALISI AVANZATE ==============
    
    def Highest(self, series_ref, length):
        """Restituisce il valore più alto in una serie per un numero di barre"""
        if hasattr(self, series_ref):
            series = getattr(self, series_ref)
            values = [series[-i] for i in range(min(length, len(series)))]
            return max(values) if values else 0
        return 0
    
    def Lowest(self, series_ref, length):
        """Restituisce il valore più basso in una serie per un numero di barre"""
        if hasattr(self, series_ref):
            series = getattr(self, series_ref)
            values = [series[-i] for i in range(min(length, len(series)))]
            return min(values) if values else 0
        return 0
    
    def Average(self, series_ref, length):
        """Calcola la media di una serie per un numero di barre"""
        if hasattr(self, series_ref):
            series = getattr(self, series_ref)
            values = [series[-i] for i in range(min(length, len(series)))]
            return sum(values) / len(values) if values else 0
        return 0
    
    def Summation(self, series_ref, length):
        """Calcola la somma di una serie per un numero di barre"""
        if hasattr(self, series_ref):
            series = getattr(self, series_ref)
            values = [series[-i] for i in range(min(length, len(series)))]
            return sum(values)
        return 0

    # ============== FUNZIONI PER PATTERN RECOGNITION ==============
    
    def InsideBar(self, bars_back=0):
        """Verifica se la barra è un inside bar"""
        if bars_back == 0:
            return (self.datahigh[0] <= self.datahigh[1] and 
                   self.datalow[0] >= self.datalow[1])
        else:
            return (self.datahigh[-bars_back] <= self.datahigh[-bars_back-1] and 
                   self.datalow[-bars_back] >= self.datalow[-bars_back-1])
    
    def OutsideBar(self, bars_back=0):
        """Verifica se la barra è un outside bar"""
        if bars_back == 0:
            return (self.datahigh[0] > self.datahigh[1] and 
                   self.datalow[0] < self.datalow[1])
        else:
            return (self.datahigh[-bars_back] > self.datahigh[-bars_back-1] and 
                   self.datalow[-bars_back] < self.datalow[-bars_back-1])
    
    def UpBar(self, bars_back=0):
        """Verifica se la barra è rialzista"""
        if bars_back == 0:
            return self.dataclose[0] > self.dataopen[0]
        else:
            return self.dataclose[-bars_back] > self.dataopen[-bars_back]
    
    def DownBar(self, bars_back=0):
        """Verifica se la barra è ribassista"""
        if bars_back == 0:
            return self.dataclose[0] < self.dataopen[0]
        else:
            return self.dataclose[-bars_back] < self.dataopen[-bars_back]

    # ============== FUNZIONI PER GESTIONE ORDINI AVANZATI ==============
    
    def CancelAllOrders(self):
        """Cancella tutti gli ordini pendenti"""
        # Placeholder per implementazione futura
        self.log("CancelAllOrders called")
        return True
    
    def ModifyOrder(self, order_id, new_price):
        """Modifica il prezzo di un ordine esistente"""
        # Placeholder per implementazione futura
        self.log(f"ModifyOrder: {{order_id}} to {{new_price}}")
        return True
    
    def GetOrderStatus(self, order_id):
        """Ottiene lo status di un ordine"""
        # Placeholder per implementazione futura
        return "Unknown"

    # ============== UTILITY FUNCTIONS ==============
    
    def IIF(self, condition, true_value, false_value):
        """Funzione IIF (Immediate IF) di EasyLanguage"""
        return true_value if condition else false_value
    
    def Mod(self, dividend, divisor):
        """Operazione modulo"""
        return dividend % divisor if divisor != 0 else 0
    
    def IntPortion(self, value):
        """Restituisce la parte intera di un numero"""
        return int(value)
    
    def FracPortion(self, value):
        """Restituisce la parte frazionaria di un numero"""
        return value - int(value)
    
    def Sign(self, value):
        """Restituisce il segno di un numero (-1, 0, 1)"""
        return 1 if value > 0 else (-1 if value < 0 else 0)

    # ============== FUNZIONI PER SUPPORTO MODULAR SYSTEM ==============
    
    def build_strategy_with_runtime_manager(self, runtime_manager=None):
        """Supporto per integrazione con RuntimeManager"""
        if runtime_manager is None:
            return self.build_enhanced_strategy_class()
        
        base_strategy = self.build_enhanced_strategy_class()
        enhanced_strategy = runtime_manager.inject_into_strategy_class(base_strategy)
        
        return enhanced_strategy
'''
        return template

    # ===================== COMPATIBILITY AND UTILITY METHODS =====================

    def build_strategy_class(self):
        """Wrapper per compatibilità - usa la versione enhanced"""
        return self.build_enhanced_strategy_class()

    def get_compilation_stats(self):
        """Restituisce statistiche sulla compilazione"""
        return {
            'total_variables': len(self.compiler.variables) if hasattr(self.compiler, 'variables') else 0,
            'total_arrays': len(self.compiler.arrays) if hasattr(self.compiler, 'arrays') else 0,
            'total_indicators': len(self.compiler.init_assignments) if hasattr(self.compiler, 'init_assignments') else 0,
            'once_statements': len(self.once_variables),
            'drawing_objects': self.drawing_counter,
            'alerts_generated': self.alert_counter,
            'file_operations': self.file_counter
        }

    def reset_counters(self):
        """Reset dei contatori per nuova compilazione"""
        self.loop_counters = {}
        self.once_variables = set()
        self.drawing_counter = 0
        self.alert_counter = 0
        self.file_counter = 0