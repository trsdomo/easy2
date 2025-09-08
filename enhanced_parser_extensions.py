"""
Enhanced Parser Extensions per EasyLanguage
Modulo completo per il parsing di tutte le funzionalità EasyLanguage avanzate
"""

import re
from typing import List, Optional, Union, Tuple, Dict, Any
from .enhanced_ast_nodes import *


class EnhancedEasyLanguageParser:
    """
    Parser esteso per tutte le funzionalità EasyLanguage avanzate:
    - Loop Statements (For/While/Repeat-Until)
    - Switch/Case Statements
    - Once Statements  
    - Alert System
    - File I/O Operations
    - Drawing Objects
    - Array Operations avanzate
    - Multi-data analysis
    - Enhanced expressions
    """
    
    def __init__(self, base_parser):
        self.base_parser = base_parser
        self.tokens = []
        self.pos = 0
        
        # Keywords per le nuove funzionalità 
        self.advanced_keywords = {
            # Loop keywords
            'for', 'to', 'downto', 'while', 'repeat', 'until', 'break', 'continue',
            # Switch keywords  
            'switch', 'case', 'default',
            # Once keyword
            'once',
            # Alert keywords
            'alert', 'checkalert', 'playsound',
            # File I/O keywords
            'file', 'fileappend', 'filedelete', 'printer', 'newline',
            # Drawing keywords
            'text_new', 'text_setcolor', 'text_delete', 'text_setstring',
            'tl_new', 'tl_setcolor', 'tl_delete', 'tl_setextend',
            # Array keywords
            'array_setmaxindex', 'summationarray', 'averagearray', 
            'highestarray', 'lowestarray', 'sortarray',
            # Math functions
            'absvalue', 'square', 'squareroot', 'power', 'log', 'log10',
            'sine', 'cosine', 'tangent', 'arcsine', 'arccosine', 'arctangent',
            'intportion', 'fracportion', 'ceiling', 'floor', 'round', 'sign',
            'minlist', 'maxlist', 'random'
        }
        
        # Patterns regex per riconoscimento avanzato
        self.patterns = {
            'for_loop': r'for\s+(\w+)\s*=\s*(.+?)\s+(to|downto)\s+(.+?)\s+begin',
            'while_loop': r'while\s+(.+?)\s+begin',
            'repeat_until': r'repeat\s+(.+?)\s+until\s+(.+?)\s*;',
            'switch_case': r'switch\s*\(\s*(.+?)\s*\)\s*begin',
            'once_statement': r'once\s+(.+?)\s+begin',
            'alert_statement': r'alert\s*(?:\(\s*(.+?)\s*\))?',
            'file_operation': r'(print\s*\(\s*file\s*\(|fileappend\s*\(|filedelete\s*\()',
            'drawing_object': r'(text_new|tl_new)\s*\(',
            'array_operation': r'(array_setmaxindex|summationarray|averagearray)\s*\(',
        }

    def parse_enhanced_statement(self, statement_text: str) -> Optional[Statement]:
        """Parser principale per statement avanzati"""
        statement_text = statement_text.strip()
        if not statement_text:
            return None
        
        # Prova a parsare ogni tipo di statement avanzato
        parsers = [
            self.parse_for_loop,
            self.parse_while_loop, 
            self.parse_repeat_until_loop,
            self.parse_switch_case,
            self.parse_once_statement,
            self.parse_alert_statement,
            self.parse_file_operation,
            self.parse_drawing_object,
            self.parse_array_operation,
            self.parse_enhanced_print,
            self.parse_enhanced_order,
        ]
        
        for parser in parsers:
            try:
                result = parser(statement_text)
                if result:
                    return result
            except Exception as e:
                continue
        
        return None

    # ===================== FOR LOOP PARSING =====================
    
    def parse_for_loop(self, text: str) -> Optional[ForLoop]:
        """Parse For loop: For x = 1 to 10 / For x = 10 downto 1"""
        # Pattern più flessibile per For loop
        patterns = [
            r'for\s+(\w+)\s*=\s*(.+?)\s+(to|downto)\s+(.+?)\s+begin\s*(.*?)\s*end',
            r'for\s+(\w+)\s*=\s*(.+?)\s+(to|downto)\s+(.+?)\s*do\s*(.*?)(?=\s*(?:end|$))',
            r'for\s+(\w+)\s*=\s*(.+?)\s+(to|downto)\s+(.+?)\s*{(.*?)}'
        ]
        
        for pattern in patterns:
            match = re.match(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                variable = match.group(1)
                start_expr = self._safe_parse_expression(match.group(2).strip())
                direction = match.group(3).lower()
                end_expr = self._safe_parse_expression(match.group(4).strip())
                body_text = match.group(5).strip() if match.group(5) else ""
                
                body = self.parse_statement_block(body_text) if body_text else []
                
                return WhileLoop(condition=condition, body=body)
        
        return None

    # ===================== REPEAT-UNTIL LOOP PARSING =====================
    
    def parse_repeat_until_loop(self, text: str) -> Optional[RepeatUntilLoop]:
        """Parse Repeat-Until loop"""
        patterns = [
            r'repeat\s+(.*?)\s+until\s+(.+?)\s*;',
            r'repeat\s*(.*?)\s+until\s+(.+?)(?=\s*$)',
            r'repeat\s*{(.*?)}\s*until\s+(.+?)\s*;'
        ]
        
        for pattern in patterns:
            match = re.match(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                body_text = match.group(1).strip()
                condition = self._safe_parse_expression(match.group(2).strip())
                
                body = self.parse_statement_block(body_text) if body_text else []
                
                return RepeatUntilLoop(body=body, condition=condition)
        
        return None

    # ===================== SWITCH/CASE PARSING =====================
    
    def parse_switch_case(self, text: str) -> Optional[SwitchCase]:
        """Parse Switch/Case statement"""
        patterns = [
            r'switch\s*\(\s*(.+?)\s*\)\s*begin\s*(.*?)\s*end',
            r'switch\s*\(\s*(.+?)\s*\)\s*{(.*?)}'
        ]
        
        for pattern in patterns:
            match = re.match(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                switch_expr = self._safe_parse_expression(match.group(1).strip())
                cases_text = match.group(2).strip()
                
                cases = self.parse_case_statements(cases_text)
                
                return SwitchCase(expression=switch_expr, cases=cases)
        
        return None
    
    def parse_case_statements(self, text: str) -> List[CaseStatement]:
        """Parse individual case statements"""
        cases = []
        
        # Split sui case e default usando regex più robusta
        case_pattern = r'\b(case|default)\b'
        parts = re.split(case_pattern, text, flags=re.IGNORECASE)
        
        current_case = None
        i = 0
        
        while i < len(parts):
            part = parts[i].strip()
            if not part:
                i += 1
                continue
                
            if part.lower() == 'case':
                if current_case:
                    cases.append(current_case)
                current_case = CaseStatement()
                i += 1
                
            elif part.lower() == 'default':
                if current_case:
                    cases.append(current_case)
                current_case = CaseStatement(is_default=True)
                i += 1
                
            else:
                if current_case:
                    # Parse case condition e body
                    if ':' in part:
                        condition_part, body_part = part.split(':', 1)
                        
                        # Parse condition
                        if not current_case.is_default:
                            conditions = self.parse_case_conditions(condition_part.strip())
                            current_case.conditions = conditions
                        
                        # Parse body
                        body_statements = self.parse_statement_block(body_part.strip())
                        current_case.body = body_statements
                        
                        # Check for break
                        if 'break' in body_part.lower():
                            current_case.has_break = True
                i += 1
        
        if current_case:
            cases.append(current_case)
            
        return cases
    
    def parse_case_conditions(self, text: str) -> List[CaseCondition]:
        """Parse case conditions (single values, ranges, lists)"""
        conditions = []
        
        # Split su virgole per multiple conditions
        parts = [part.strip() for part in text.split(',')]
        
        for part in parts:
            if ' to ' in part.lower():
                # Range condition: 10 to 20
                start_str, end_str = part.lower().split(' to ')
                start_expr = self._safe_parse_expression(start_str.strip())
                end_expr = self._safe_parse_expression(end_str.strip())
                conditions.append(CaseCondition(range=True, start=start_expr, end=end_expr))
            else:
                # Single value condition
                value_expr = self._safe_parse_expression(part)
                conditions.append(CaseCondition(value=value_expr))
        
        return conditions

    # ===================== ONCE STATEMENT PARSING =====================
    
    def parse_once_statement(self, text: str) -> Optional[OnceStatement]:
        """Parse Once statement"""
        patterns = [
            r'once\s+(.+?)\s+begin\s*(.*?)\s*end',
            r'once\s+(.+?)\s*{(.*?)}'
        ]
        
        for pattern in patterns:
            match = re.match(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                condition = self._safe_parse_expression(match.group(1).strip())
                body_text = match.group(2).strip() if match.group(2) else ""
                
                body = self.parse_statement_block(body_text) if body_text else []
                
                return OnceStatement(condition=condition, body=body)
        
        return None

    # ===================== ALERT PARSING =====================
    
    def parse_alert_statement(self, text: str) -> Optional[AlertStatement]:
        """Parse Alert statement"""
        # Simple alert
        if re.match(r'alert\s*;', text, re.IGNORECASE):
            return AlertStatement()
        
        # Alert with message
        match = re.match(r'alert\s*\(\s*(.+?)\s*\)\s*;', text, re.IGNORECASE)
        if match:
            message = self._safe_parse_expression(match.group(1).strip())
            return AlertStatement(message=message)
        
        # Alert with condition
        match = re.match(r'if\s+(.+?)\s+then\s+alert(?:\s*\(\s*(.+?)\s*\))?\s*;', text, re.IGNORECASE)
        if match:
            condition = self._safe_parse_expression(match.group(1).strip())
            message = None
            if match.group(2):
                message = self._safe_parse_expression(match.group(2).strip())
            return AlertStatement(condition=condition, message=message)
        
        return None

    # ===================== FILE I/O PARSING =====================
    
    def parse_file_operation(self, text: str) -> Optional[FileOperation]:
        """Parse File I/O operations"""
        
        # Print to file: Print(File("filename"), args...)
        match = re.match(r'print\s*\(\s*file\s*\(\s*(.+?)\s*\)\s*(?:,\s*(.+?))?\s*\)\s*;', text, re.IGNORECASE)
        if match:
            filename = self._safe_parse_expression(match.group(1).strip())
            args = []
            if match.group(2):
                args_text = match.group(2).strip()
                # Parsing più robusto degli argomenti
                args = self._parse_argument_list(args_text)
            
            return FileOperation(
                operation='print_to_file',
                filename=filename,
                arguments=args
            )
        
        # FileAppend
        match = re.match(r'fileappend\s*\(\s*(.+?)\s*,\s*(.+?)\s*\)\s*;', text, re.IGNORECASE)
        if match:
            filename = self._safe_parse_expression(match.group(1).strip())
            text_expr = self._safe_parse_expression(match.group(2).strip())
            
            return FileOperation(
                operation='file_append',
                filename=filename,
                text=text_expr
            )
        
        # FileDelete
        match = re.match(r'filedelete\s*\(\s*(.+?)\s*\)\s*;', text, re.IGNORECASE)
        if match:
            filename = self._safe_parse_expression(match.group(1).strip())
            
            return FileOperation(
                operation='file_delete',
                filename=filename
            )
        
        return None

    # ===================== DRAWING OBJECTS PARSING =====================
    
    def parse_drawing_object(self, text: str) -> Optional[DrawingObject]:
        """Parse Drawing objects (Text, Trendlines)"""
        
        # Text_New
        match = re.match(r'text_new\s*\(\s*(.+?)\s*,\s*(.+?)\s*,\s*(.+?)\s*,\s*(.+?)\s*\)\s*;', text, re.IGNORECASE)
        if match:
            date = self._safe_parse_expression(match.group(1).strip())
            time = self._safe_parse_expression(match.group(2).strip())
            price = self._safe_parse_expression(match.group(3).strip())
            text_expr = self._safe_parse_expression(match.group(4).strip())
            
            return DrawingObject(
                object_type='text_new',
                date=date,
                time=time,
                price=price,
                text=text_expr
            )
        
        # TL_New (TrendLine) - parsing più flessibile
        tl_pattern = r'tl_new\s*\(\s*(.+?)\s*,\s*(.+?)\s*,\s*(.+?)\s*,\s*(.+?)\s*,\s*(.+?)\s*,\s*(.+?)\s*\)\s*;'
        match = re.match(tl_pattern, text, re.IGNORECASE)
        if match:
            date1 = self._safe_parse_expression(match.group(1).strip())
            time1 = self._safe_parse_expression(match.group(2).strip())
            price1 = self._safe_parse_expression(match.group(3).strip())
            date2 = self._safe_parse_expression(match.group(4).strip())
            time2 = self._safe_parse_expression(match.group(5).strip())
            price2 = self._safe_parse_expression(match.group(6).strip())
            
            return DrawingObject(
                object_type='trendline_new',
                date1=date1, time1=time1, price1=price1,
                date2=date2, time2=time2, price2=price2
            )
        
        return None

    # ===================== ARRAY OPERATIONS PARSING =====================
    
    def parse_array_operation(self, text: str) -> Optional[Statement]:
        """Parse Array operations"""
        
        # Array_SetMaxIndex
        match = re.match(r'array_setmaxindex\s*\(\s*(\w+)\s*,\s*(.+?)\s*\)\s*;', text, re.IGNORECASE)
        if match:
            array_name = match.group(1)
            max_index = self._safe_parse_expression(match.group(2).strip())
            return ArraySetMaxIndex(array_name=array_name, max_index=max_index)
        
        # Array functions (SummationArray, AverageArray, etc.)
        array_functions = ['summationarray', 'averagearray', 'highestarray', 'lowestarray', 'sortarray']
        for func_name in array_functions:
            pattern = rf'{func_name}\s*\(\s*(\w+)\s*,\s*(.+?)\s*\)'
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                array_ref = match.group(1)
                size = self._safe_parse_expression(match.group(2).strip())
                return ArrayFunction(
                    function_name=func_name,
                    array_ref=array_ref,
                    size=size
                )
        
        return None

    # ===================== ENHANCED PRINT PARSING =====================
    
    def parse_enhanced_print(self, text: str) -> Optional[EnhancedPrintStatement]:
        """Parse enhanced Print statements"""
        
        # Print to printer
        match = re.match(r'print\s*\(\s*printer\s*(?:,\s*(.+?))?\s*\)\s*;', text, re.IGNORECASE)
        if match:
            args = []
            if match.group(1):
                args_text = match.group(1).strip()
                args = self._parse_argument_list(args_text)
            
            return EnhancedPrintStatement(
                output_type='printer',
                arguments=args
            )
        
        return None

    # ===================== ENHANCED ORDER PARSING =====================
    
    def parse_enhanced_order(self, text: str) -> Optional[EnhancedOrderStatement]:
        """Parse enhanced order statements con supporto completo"""
        
        # Pattern più avanzato per ordini
        order_verbs = 'buy|sell|sellshort|buytocover|exitlong|exitshort|exitposition'
        
        # Pattern complesso per ordini EasyLanguage
        patterns = [
            rf'({order_verbs})(?:\s*\(\s*"(.+?)"\s*\))?\s*(?:(\d+)\s+shares?\s+)?(?:next\s+bar\s+at\s+)?(?:(market|limit|stop|close)(?:\s+(.+?))?)?\s*(?:from\s+entry\s+"(.+?)")?\s*;',
            rf'({order_verbs})\s*(?:"(.+?)")?\s*(?:(\d+)\s+contracts?)?\s*(?:at\s+(market|limit|stop|close)(?:\s+(.+?))?)?\s*;'
        ]
        
        for pattern in patterns:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                order_type = match.group(1).lower()
                signal_name = match.group(2) if len(match.groups()) > 1 and match.group(2) else None
                shares = match.group(3) if len(match.groups()) > 2 and match.group(3) else None
                price_type = match.group(4) if len(match.groups()) > 3 and match.group(4) else 'market'
                price_value = match.group(5) if len(match.groups()) > 4 and match.group(5) else None
                from_entry = match.group(6) if len(match.groups()) > 5 and match.group(6) else None
                
                return EnhancedOrderStatement(
                    order_type=order_type,
                    signal_name=signal_name,
                    shares=int(shares) if shares else None,
                    price_type=price_type.lower() if price_type else 'market',
                    price_value=self._safe_parse_expression(price_value) if price_value else None,
                    from_entry=from_entry
                )
        
        return None

    # ===================== UTILITY METHODS =====================
    
    def parse_statement_block(self, text: str) -> List[Statement]:
        """Parse un blocco di statement"""
        if not text or not text.strip():
            return []
            
        statements = []
        statement_texts = self.split_statements(text)
        
        for stmt_text in statement_texts:
            stmt_text = stmt_text.strip()
            if not stmt_text or stmt_text.startswith('//') or stmt_text.startswith('{'):
                continue
                
            # Prova prima i parser avanzati
            advanced_stmt = self.parse_enhanced_statement(stmt_text)
            if advanced_stmt:
                statements.append(advanced_stmt)
            else:
                # Fallback al parser base
                try:
                    base_stmt = self.base_parser.parse_statement(stmt_text)
                    if base_stmt:
                        statements.append(base_stmt)
                except:
                    # Se anche il parser base fallisce, crea un commento
                    continue
        
        return statements
    
    def split_statements(self, text: str) -> List[str]:
        """Split statement su ; ma ignora ; dentro stringhe e commenti"""
        statements = []
        current = ""
        in_string = False
        in_comment = False
        string_char = None
        i = 0
        
        while i < len(text):
            char = text[i]
            
            # Gestione stringhe
            if char in ['"', "'"] and not in_comment:
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None
                current += char
                
            # Gestione commenti di blocco
            elif char == '{' and not in_string:
                in_comment = True
                current += char
            elif char == '}' and not in_string and in_comment:
                in_comment = False
                current += char
                
            # Gestione commenti di linea
            elif char == '/' and i + 1 < len(text) and text[i + 1] == '/' and not in_string and not in_comment:
                # Salta fino alla fine della linea
                while i < len(text) and text[i] != '\n':
                    current += text[i]
                    i += 1
                continue
                
            # Split sui semicolon
            elif char == ';' and not in_string and not in_comment:
                if current.strip():
                    statements.append(current.strip())
                current = ""
            else:
                current += char
            
            i += 1
        
        if current.strip():
            statements.append(current.strip())
        
        return statements

    def _safe_parse_expression(self, expr_text: str):
        """Parse sicuro delle espressioni con fallback"""
        if not expr_text or not expr_text.strip():
            return create_literal(0)
            
        try:
            if self.base_parser and hasattr(self.base_parser, 'expression_compiler'):
                return self.base_parser.expression_compiler.parse_expression(expr_text.strip())
            elif self.base_parser and hasattr(self.base_parser, 'parse_expression'):
                return self.base_parser.parse_expression(expr_text.strip())
            else:
                # Fallback: crea un identificatore o letterale
                expr_text = expr_text.strip()
                if expr_text.isdigit():
                    return create_literal(int(expr_text))
                elif expr_text.replace('.', '').isdigit():
                    return create_literal(float(expr_text))
                elif expr_text.startswith('"') and expr_text.endswith('"'):
                    return create_literal(expr_text[1:-1])
                else:
                    return create_identifier(expr_text)
        except:
            # Ultimate fallback
            return create_literal(0)

    def _parse_argument_list(self, args_text: str) -> List[Expression]:
        """Parse robusta di una lista di argomenti"""
        if not args_text or not args_text.strip():
            return []
            
        args = []
        current_arg = ""
        paren_count = 0
        in_string = False
        string_char = None
        
        for char in args_text:
            if char in ['"', "'"] and not in_string:
                in_string = True
                string_char = char
                current_arg += char
            elif char == string_char and in_string:
                in_string = False
                string_char = None
                current_arg += char
            elif char == '(' and not in_string:
                paren_count += 1
                current_arg += char
            elif char == ')' and not in_string:
                paren_count -= 1
                current_arg += char
            elif char == ',' and not in_string and paren_count == 0:
                if current_arg.strip():
                    args.append(self._safe_parse_expression(current_arg.strip()))
                current_arg = ""
            else:
                current_arg += char
        
        if current_arg.strip():
            args.append(self._safe_parse_expression(current_arg.strip()))
        
        return args

    # ===================== INTEGRATION WITH BASE PARSER =====================
    
    def integrate_with_base_parser(self, base_parser):
        """Integra i parser avanzati con il parser base"""
        
        # Salva il metodo originale parse_statement
        if hasattr(base_parser, 'parse_statement'):
            original_parse_statement = base_parser.parse_statement
            
            def enhanced_parse_statement(statement_text):
                # Prova prima i parser avanzati
                enhanced_result = self.parse_enhanced_statement(statement_text)
                if enhanced_result:
                    return enhanced_result
                
                # Fallback al parser originale
                try:
                    return original_parse_statement(statement_text)
                except:
                    return None
            
            # Sostituisci il metodo
            base_parser.parse_statement = enhanced_parse_statement
        
        base_parser.enhanced_parser = self
        self.base_parser = base_parser


# ===================== EXPRESSION PARSER EXTENSIONS =====================

class EnhancedExpressionParser:
    """Parser esteso per espressioni EasyLanguage avanzate"""
    
    def __init__(self, base_expression_parser):
        self.base_parser = base_expression_parser
        
    def parse_enhanced_expression(self, expr_text: str) -> Optional[Expression]:
        """Parser principale per espressioni avanzate"""
        expr_text = expr_text.strip()
        if not expr_text:
            return None
        
        try:
            # Array access: MyArray[5] o MyArray[5][1]
            array_match = re.match(r'(\w+)\[(.+?)\](?:\[(.+?)\])?', expr_text)
            if array_match:
                array_name = array_match.group(1)
                index = self._safe_parse_expression(array_match.group(2))
                historical_ref = None
                if array_match.group(3):
                    historical_ref = self._safe_parse_expression(array_match.group(3))
                
                return ArrayAccess(
                    array_name=array_name,
                    index=index,
                    historical_reference=historical_ref
                )
            
            # Data references: Data2, Data(n)
            data_match = re.match(r'data(\d+)|data\s*\(\s*(.+?)\s*\)', expr_text, re.IGNORECASE)
            if data_match:
                if data_match.group(1):
                    data_number = int(data_match.group(1))
                else:
                    data_number = self._safe_parse_expression(data_match.group(2))
                return DataReference(data_number=data_number)
            
            # Math functions
            math_functions = ['absvalue', 'square', 'squareroot', 'power', 'log', 'log10', 
                             'sine', 'cosine', 'tangent', 'arcsine', 'arccosine', 'arctangent',
                             'intportion', 'fracportion', 'ceiling', 'floor', 'round', 'sign',
                             'minlist', 'maxlist', 'random']
            
            for func_name in math_functions:
                pattern = rf'{func_name}\s*\(\s*(.+?)\s*\)'
                match = re.match(pattern, expr_text, re.IGNORECASE)
                if match:
                    args_text = match.group(1)
                    args = self._parse_argument_list(args_text)
                    return MathFunction(function_name=func_name, arguments=args)
            
            # Conditional expression (ternary): condition ? true_expr : false_expr
            ternary_match = re.match(r'(.+?)\?\s*(.+?)\s*:\s*(.+)', expr_text)
            if ternary_match:
                condition = self._safe_parse_expression(ternary_match.group(1).strip())
                true_expr = self._safe_parse_expression(ternary_match.group(2).strip())
                false_expr = self._safe_parse_expression(ternary_match.group(3).strip())
                
                return ConditionalExpression(
                    condition=condition,
                    true_expr=true_expr,
                    false_expr=false_expr
                )
            
        except Exception as e:
            pass
        
        return None

    def _safe_parse_expression(self, expr_text: str):
        """Parse sicuro delle espressioni"""
        if self.base_parser and hasattr(self.base_parser, 'parse_expression'):
            try:
                return self.base_parser.parse_expression(expr_text.strip())
            except:
                return create_literal(0)
        else:
            return create_literal(0)

    def _parse_argument_list(self, args_text: str) -> List[Expression]:
        """Parse di una lista di argomenti"""
        if not args_text.strip():
            return []
        
        args = []
        for arg in args_text.split(','):
            arg = arg.strip()
            if arg:
                args.append(self._safe_parse_expression(arg))
        
        return args


# ===================== COMPREHENSIVE INTEGRATION =====================

class ComprehensiveEasyLanguageParser:
    """Parser completo che integra tutte le funzionalità"""
    
    def __init__(self):
        self.base_parser = None
        self.statement_parser = None
        self.expression_parser = None
        
    def setup_parsers(self, base_parser):
        """Setup di tutti i parser"""
        self.base_parser = base_parser
        self.statement_parser = EnhancedEasyLanguageParser(base_parser)
        
        if hasattr(base_parser, 'expression_compiler'):
            self.expression_parser = EnhancedExpressionParser(base_parser.expression_compiler)
        
        # Integra con il parser base
        self.statement_parser.integrate_with_base_parser(base_parser)
    
    def parse_complete_easylanguage_file(self, source_code: str) -> List[Statement]:
        """Parse completo di un file EasyLanguage"""
        try:
            # Divide in sezioni logiche
            sections = self.divide_into_sections(source_code)
            
            statements = []
            
            # Parse main code
            main_code = sections.get('main', '')
            if main_code:
                main_statements = self.statement_parser.parse_statement_block(main_code)
                statements.extend(main_statements)
            
            return statements
        except Exception as e:
            return []
    
    def divide_into_sections(self, source_code: str) -> Dict[str, str]:
        """Divide il codice in sezioni logiche"""
        sections = {
            'inputs': '',
            'variables': '',
            'arrays': '',
            'main': ''
        }
        
        try:
            # Regex per identificare le sezioni
            inputs_match = re.search(r'inputs?\s*:(.*?)(?=(?:vars?|variables?|arrays?|\w+\s*=|\w+\s*\(|$))', 
                                   source_code, re.IGNORECASE | re.DOTALL)
            vars_match = re.search(r'(?:vars?|variables?)\s*:(.*?)(?=(?:arrays?|\w+\s*=|\w+\s*\(|$))', 
                                 source_code, re.IGNORECASE | re.DOTALL)
            arrays_match = re.search(r'arrays?\s*:(.*?)(?=(?:\w+\s*=|\w+\s*\(|$))', 
                                   source_code, re.IGNORECASE | re.DOTALL)
            
            if inputs_match:
                sections['inputs'] = inputs_match.group(1).strip()
            if vars_match:
                sections['variables'] = vars_match.group(1).strip()
            if arrays_match:
                sections['arrays'] = arrays_match.group(1).strip()
            
            # Il resto è main code
            main_start = 0
            for match in [inputs_match, vars_match, arrays_match]:
                if match:
                    main_start = max(main_start, match.end())
            
            sections['main'] = source_code[main_start:].strip()
            
        except Exception as e:
            sections['main'] = source_code
        
        return sections if body_text else []
                
                return ForLoop(
                    variable=variable,
                    start_value=start_expr,
                    end_value=end_expr,
                    direction=direction,
                    body=body
                )
        
        return None

    # ===================== WHILE LOOP PARSING =====================
    
    def parse_while_loop(self, text: str) -> Optional[WhileLoop]:
        """Parse While loop"""
        patterns = [
            r'while\s+(.+?)\s+begin\s*(.*?)\s*end',
            r'while\s+(.+?)\s+do\s*(.*?)(?=\s*(?:end|$))',
            r'while\s+(.+?)\s*{(.*?)}'
        ]
        
        for pattern in patterns:
            match = re.match(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                condition = self._safe_parse_expression(match.group(1).strip())
                body_text = match.group(2).strip() if match.group(2) else ""
                
                body = self.parse_statement_block(body_text)