import backtrader as bt
import math
import numpy as np
from collections import deque

from lark import Lark
from .grammar import easytrader_grammar
from .transformer import EasyTraderTransformer
from ..functions.runtime import EasyTraderFunctions
from .ast_nodes import *
from .function_mappings import FunctionMappings
from .expression_compiler import ExpressionCompiler
from .enhanced_code_generators_unified import EnhancedCodeGenerators

class EasyTraderCompiler:
    def __init__(self, ast_tree=None):
        self.ast = ast_tree if isinstance(ast_tree, list) else [ast_tree] if ast_tree else []
        self.params = {}
        self.variables = {}
        self.arrays = {}
        self.init_code = []
        self.next_code = []
        self.parser = Lark(easytrader_grammar, parser='lalr', start='start', propagate_positions=False)
        
        # Tracciamento degli indicatori definiti in __init__
        self._compilation_context = 'init' 
        self.init_assignments = {}

        self.dynamic_indicators = {}
        self.dynamic_indicator_count = 0
        self.plot_info = {}

        # Informazioni globali sulla strategia
        self.stop_loss = None
        self.profit_target = None
        self.options = {}
        self.order_counter = 0

        # Inizializza i moduli helper con la versione enhanced
        self.function_mappings = FunctionMappings()
        self.expression_compiler = ExpressionCompiler(self)
        self.code_generators = EnhancedCodeGenerators(self)

    def compile(self):
        """Metodo principale che orchestra il processo di compilazione."""
        try:
            self._pre_scan_variables()
            self._process_declarations()
            self._process_assignments()
            self._process_global_statements()
            self._process_plots()
            self._generate_next_code()
            self._generate_init_code()
            return self._build_strategy_class()
        except Exception as e:
            raise Exception(f"Errore durante la compilazione: {e}")

    def compile_from_text(self, easylanguage_code):
        """Compila direttamente da testo EasyLanguage grezzo senza AST intermedi."""
        try:
            tree = self._parse_raw_easylanguage(easylanguage_code)
            transformer = EasyTraderTransformer()
            ast_list = transformer.transform(tree)
            self.ast = ast_list if isinstance(ast_list, list) else [ast_list]
            return self.compile()
        except Exception as e:
            raise Exception(f"Errore durante la compilazione da testo: {e}")

    def _parse_raw_easylanguage(self, code):
        """Parsa testo EasyLanguage grezzo usando il parser Lark e crea un AST."""
        return self.parser.parse(code.lower())

    def _pre_scan_variables(self):
        """Pre-scansione dell'AST per identificare variabili non dichiarate."""
        def scan_node(node):
            if not isinstance(node, tuple):
                return

            node_type = node[0]
            if node_type == 'assignment' and len(node) >= 3:
                var_name = node[1]
                if var_name.startswith('Value') and var_name[5:].isdigit():
                    if var_name not in self.variables:
                        self.variables[var_name] = {'initial_value': 0, 'data_type': 'Numeric'}
                scan_node(node[2])
            elif node_type in ['binary_op', 'unary_op'] and len(node) >= 2:
                for sub_node in node[1:]:
                    if isinstance(sub_node, (tuple, list)):
                        scan_node(sub_node)
            elif node_type == 'func_call' and len(node) >= 3:
                for arg in node[2]:
                    if isinstance(arg, (tuple, list)):
                        scan_node(arg)
            elif node_type == 'if_statement' and len(node) >= 2:
                scan_node(node[1])
                for sub_node in node[2:]:
                    if isinstance(sub_node, (tuple, list)):
                        scan_node(sub_node)
            elif len(node) > 1:
                for sub_node in node[1:]:
                    if isinstance(sub_node, (tuple, list)):
                        scan_node(sub_node)

        for node in self.ast:
            scan_node(node)

    def _process_plots(self):
        """Scansiona l'AST per i comandi Plot e popola self.plot_info."""
        for node in self.ast:
            if isinstance(node, tuple) and node[0] == 'plot':
                if len(node) >= 3 and len(node[2]) >= 2:
                    var_name = node[2][0]
                    plot_name = node[2][1]
                    self.plot_info[var_name] = plot_name

    def _process_declarations(self):
        ast_statements = []
        if self.ast and isinstance(self.ast[0], Program):
            ast_statements = self.ast[0].statements
        else:
            ast_statements = self.ast

        for node in ast_statements:
            if hasattr(node, 'node_type'):
                node_type = node.node_type
                
                if node_type == 'inputs_decl':
                    for decl in node.declarations:
                        value = decl.initial_value.value if hasattr(decl.initial_value, 'value') else decl.initial_value
                        self.params[decl.name] = value
                        
                elif node_type == 'vars_decl':
                    for decl in node.declarations:
                        self.variables[decl.name] = {
                            'initial_value': decl.initial_value, 
                            'data_type': decl.data_type
                        }
            
            elif isinstance(node, tuple) and len(node) >= 2:
                node_type = node[0]
                
                if node_type == 'inputs_decl':
                    for name, initial_value, data_type in node[1]:
                        self.params[name] = initial_value
                        
                elif node_type == 'vars_decl':
                    for name, initial_value, data_type in node[1]:
                        self.variables[name] = {'initial_value': initial_value, 'data_type': data_type}
                        
                elif node_type == 'array_decl':
                    for name, dimensions, initial_value, data_type in node[1]:
                        self.arrays[name] = {'dimensions': dimensions, 'initial_value': initial_value, 'data_type': data_type}
                        
                elif node_type == 'attribute':
                    self.options[node[1]] = node[2]

    def _process_assignments(self):
        """Scansiona le assegnazioni per identificare gli indicatori che vanno in __init__."""
        ast_statements = []
        if len(self.ast) == 1 and isinstance(self.ast[0], Program):
            ast_statements = self.ast[0].statements
        else:
            ast_statements = self.ast

        for node in ast_statements:
            if isinstance(node, Assignment):
                var_name = node.variable
                value_node = node.expression

                def find_func_call_dataclass(n):
                    if isinstance(n, FunctionCall):
                        return n
                    elif isinstance(n, BinaryOp):
                        found = find_func_call_dataclass(n.left)
                        if found: return found
                        found = find_func_call_dataclass(n.right)
                        if found: return found
                    elif isinstance(n, UnaryOp):
                        found = find_func_call_dataclass(n.operand)
                        if found: return found
                    elif isinstance(n, HistoricalAccess):
                        found = find_func_call_dataclass(n.base_name)
                        if found: return found
                    return None

                func_call_node = find_func_call_dataclass(value_node)

                if func_call_node:
                    func_name = func_call_node.name.lower()
                    if func_name in self.function_mappings.function_map:
                        self.init_assignments[var_name] = value_node

    def _process_global_statements(self):
        """Scansiona tutti gli statements per informazioni globali come SetStopLoss."""
        for node in self.ast:
            if isinstance(node, tuple) and node[0] == 'stop' and len(node) >= 2:
                kind = node[1].lower()
                args = node[2] if len(node) > 2 else []
                value = args[0] if args else None
                
                if kind == 'setstoploss':
                    self.stop_loss = value
                elif kind == 'setprofittarget':
                    self.profit_target = value
                elif kind == 'setexitonclose':
                    self.options['set_exit_on_close'] = True

    def _generate_init_code(self):
        """Delega la generazione del codice __init__ al modulo Enhanced CodeGenerators."""
        self.code_generators.generate_init_code()

    def _generate_next_code(self):
        """Delega la generazione del codice next() al modulo Enhanced CodeGenerators."""
        self.code_generators.generate_next_code()

    def _build_strategy_class(self):
        """Delega la costruzione della classe al modulo Enhanced CodeGenerators."""
        return self.code_generators.build_enhanced_strategy_class()

    def _compile_statement_block(self, block, indent_level=2):
        """Delega al modulo Enhanced CodeGenerators."""
        return self.code_generators.compile_statement_block(block, indent_level)

    # ===================== NUOVE FUNZIONALITÀ ENHANCED =====================
    
    def get_compilation_stats(self):
        """Restituisce statistiche complete di compilazione"""
        return self.code_generators.get_compilation_stats()
    
    def reset_compilation_counters(self):
        """Reset dei contatori per nuova compilazione"""
        self.code_generators.reset_counters()
        
    def supports_advanced_features(self):
        """Verifica il supporto per funzionalità avanzate"""
        return True
    
    def get_supported_features(self):
        """Lista delle funzionalità supportate dalla versione enhanced"""
        return [
            'for_loops',
            'while_loops', 
            'repeat_until_loops',
            'switch_case_statements',
            'once_statements',
            'alert_system',
            'file_io_operations',
            'drawing_objects',
            'multi_data_analysis',
            'advanced_order_types',
            'array_operations',
            'trailing_stops',
            'bracket_orders',
            'performance_tracking',
            'loop_protection',
            'pattern_recognition',
            'mathematical_functions',
            'utility_functions'
        ]
    
    def compile_with_enhanced_features(self, easylanguage_code, enable_advanced=True):
        """
        Compila con funzionalità enhanced abilitate
        
        Args:
            easylanguage_code: Il codice EasyLanguage da compilare
            enable_advanced: Se abilitare le funzionalità avanzate
        
        Returns:
            Tuple: (compiled_strategy_class, compilation_stats)
        """
        try:
            # Reset contatori
            self.reset_compilation_counters()
            
            # Compila il codice
            strategy_class = self.compile_from_text(easylanguage_code)
            
            # Ottieni statistiche
            stats = self.get_compilation_stats()
            
            return strategy_class, stats
            
        except Exception as e:
            raise Exception(f"Errore durante la compilazione enhanced: {e}")
    
    def validate_enhanced_syntax(self, easylanguage_code):
        """
        Valida la sintassi per le funzionalità enhanced
        
        Returns:
            Dict: Risultato della validazione con eventuali warning/errori
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'advanced_features_detected': []
        }
        
        # Lista di funzionalità avanzate da rilevare nel codice
        advanced_patterns = {
            'for_loop': ['for ', ' to ', ' downto '],
            'while_loop': ['while ', ' begin'],
            'switch_case': ['switch ', 'case ', 'default:'],
            'once_statement': ['once begin'],
            'alert': ['alert(', 'alert '],
            'file_io': ['print(file(', 'fileappend(', 'filedelete('],
            'drawing': ['text_new(', 'tl_new('],
            'array_ops': ['array_setmaxindex(', 'summationarray('],
        }
        
        code_lower = easylanguage_code.lower()
        
        for feature, patterns in advanced_patterns.items():
            for pattern in patterns:
                if pattern in code_lower:
                    validation_result['advanced_features_detected'].append(feature)
                    break
        
        return validation_result