"""
Enhanced Runtime Manager per EasyLanguage
Gestisce l'integrazione di moduli specializzati come market data, 
indicators, analysis functions, etc.
"""

import importlib
import inspect
from typing import Dict, List, Any, Callable
from pathlib import Path


class EasyLanguageRuntimeManager:
    """
    Manager che coordina tutti i moduli EasyLanguage e li integra
    nella strategia compilata
    """
    
    def __init__(self):
        self.registered_modules = {}
        self.function_registry = {}
        self.init_functions = []
        self.update_functions = []
        self.cleanup_functions = []
        
    def register_module(self, module_name: str, module_instance=None):
        """Registra un modulo EasyLanguage"""
        if module_instance is None:
            try:
                module_instance = importlib.import_module(module_name)
            except ImportError:
                print(f"Warning: Could not import module {module_name}")
                return False
        
        self.registered_modules[module_name] = module_instance
        self._extract_functions_from_module(module_name, module_instance)
        return True
    
    def _extract_functions_from_module(self, module_name: str, module_instance):
        """Estrae funzioni dal modulo e le categorizza"""
        for name, obj in inspect.getmembers(module_instance):
            if inspect.isfunction(obj):
                # Categorizza le funzioni per tipo
                if name.startswith('__init_'):
                    self.init_functions.append((module_name, name, obj))
                elif name.startswith('update_') and name.endswith('_optimized'):
                    self.update_functions.append((module_name, name, obj))
                elif name.startswith('cleanup_') or name.startswith('reset_'):
                    self.cleanup_functions.append((module_name, name, obj))
                elif not name.startswith('_') and callable(obj):
                    # Funzione EasyLanguage pubblica
                    self.function_registry[name] = (module_name, obj)
    
    def inject_into_strategy_class(self, strategy_class_code: str) -> str:
        """Inietta le funzioni dei moduli nella classe strategia"""
        
        # Trova dove inserire le inizializzazioni
        init_insertion_point = strategy_class_code.find("self._init_easylanguage_systems()")
        if init_insertion_point == -1:
            init_insertion_point = strategy_class_code.find("def __init__(self):")
            init_insertion_point = strategy_class_code.find("\n", init_insertion_point) + 1
        
        # Genera codice di inizializzazione moduli
        init_code = self._generate_init_code()
        
        # Genera codice di update
        update_code = self._generate_update_code()
        
        # Genera funzioni EasyLanguage
        functions_code = self._generate_functions_code()
        
        # Inserisci nel codice della strategia
        modified_code = self._insert_code_sections(
            strategy_class_code, 
            init_code, 
            update_code, 
            functions_code
        )
        
        return modified_code
    
    def _generate_init_code(self) -> str:
        """Genera codice di inizializzazione per tutti i moduli"""
        lines = ["\n        # Initialize specialized modules"]
        
        for module_name, func_name, func_obj in self.init_functions:
            lines.append(f"        # Initialize {module_name}")
            lines.append(f"        self.{func_name}()")
        
        return '\n'.join(lines)
    
    def _generate_update_code(self) -> str:
        """Genera codice di update per il metodo next()"""
        lines = ["\n        # Update specialized modules"]
        
        for module_name, func_name, func_obj in self.update_functions:
            lines.append(f"        # Update {module_name}")
            lines.append(f"        self.{func_name}()")
        
        return '\n'.join(lines)
    
    def _generate_functions_code(self) -> str:
        """Genera il codice delle funzioni EasyLanguage"""
        lines = ["\n    # ============== SPECIALIZED MODULE FUNCTIONS =============="]
        
        for func_name, (module_name, func_obj) in self.function_registry.items():
            # Ottieni la signature della funzione
            sig = inspect.signature(func_obj)
            params = list(sig.parameters.keys())[1:]  # Skip 'self'
            params_str = ', '.join(params)
            
            # Ottieni docstring se presente
            docstring = func_obj.__doc__ or f"{func_name} from {module_name}"
            
            lines.append(f"\n    def {func_name}(self{', ' + params_str if params_str else ''}):")
            lines.append(f'        """{docstring}"""')
            lines.append(f"        return {module_name}.{func_name}(self{', ' + params_str if params_str else ''})")
        
        return '\n'.join(lines)
    
    def _insert_code_sections(self, original_code: str, init_code: str, 
                             update_code: str, functions_code: str) -> str:
        """Inserisce le sezioni di codice nella strategia"""
        
        # Inserisci init code
        if "self._init_easylanguage_systems()" in original_code:
            original_code = original_code.replace(
                "self._init_easylanguage_systems()",
                f"self._init_easylanguage_systems(){init_code}"
            )
        
        # Inserisci update code nel metodo next()
        next_method_start = original_code.find("def next(self):")
        if next_method_start != -1:
            # Trova la fine dell'header del metodo next
            next_body_start = original_code.find("\n", next_method_start)
            next_body_start = original_code.find("\n", next_body_start + 1)  # Skip docstring se presente
            
            # Inserisci update code all'inizio del body
            original_code = (
                original_code[:next_body_start] + 
                update_code + 
                original_code[next_body_start:]
            )
        
        # Aggiungi le funzioni alla fine della classe (prima dell'ultimo ''')
        class_end = original_code.rfind("'''")
        if class_end != -1:
            original_code = (
                original_code[:class_end] + 
                functions_code + 
                "\n" + 
                original_code[class_end:]
            )
        
        return original_code
    
    def auto_discover_modules(self, modules_directory: str = "./"):
        """Scopre automaticamente moduli EasyLanguage nella directory"""
        modules_path = Path(modules_directory)
        
        for py_file in modules_path.glob("easylanguage_*.py"):
            module_name = py_file.stem
            try:
                self.register_module(module_name)
                print(f"Auto-discovered and registered: {module_name}")
            except Exception as e:
                print(f"Failed to register {module_name}: {e}")
    
    def get_available_functions(self) -> Dict[str, str]:
        """Restituisce lista delle funzioni disponibili per modulo"""
        functions_by_module = {}
        
        for func_name, (module_name, func_obj) in self.function_registry.items():
            if module_name not in functions_by_module:
                functions_by_module[module_name] = []
            
            # Aggiungi info sulla funzione
            sig = inspect.signature(func_obj)
            docstring = func_obj.__doc__ or "No description"
            
            functions_by_module[module_name].append({
                'name': func_name,
                'signature': str(sig),
                'description': docstring.split('\n')[0]  # Prima riga
            })
        
        return functions_by_module
    
    def validate_module_compatibility(self, module_name: str) -> bool:
        """Valida che un modulo sia compatibile con il runtime"""
        if module_name not in self.registered_modules:
            return False
        
        module = self.registered_modules[module_name]
        
        # Controlla che abbia almeno una funzione di init o update
        has_init = any(name.startswith('__init_') for name, _ in inspect.getmembers(module, inspect.isfunction))
        has_functions = any(not name.startswith('_') for name, obj in inspect.getmembers(module, inspect.isfunction))
        
        return has_init or has_functions


# =============================================================================
# ENHANCED CODE GENERATORS INTEGRATION
# =============================================================================

class ModularCodeGenerators:
    """
    Estensione di EnhancedCodeGenerators che supporta moduli esterni
    """
    
    def __init__(self, base_generators, runtime_manager: EasyLanguageRuntimeManager):
        self.base = base_generators
        self.runtime_manager = runtime_manager
    
    def build_enhanced_strategy_class_with_modules(self):
        """Costruisce la strategia con tutti i moduli integrati"""
        
        # Genera la classe base
        base_class_code = self.base.build_enhanced_strategy_class()
        
        # Inietta i moduli
        enhanced_class_code = self.runtime_manager.inject_into_strategy_class(base_class_code)
        
        # Aggiungi imports necessari
        enhanced_class_code = self._add_required_imports(enhanced_class_code)
        
        return enhanced_class_code
    
    def _add_required_imports(self, class_code: str) -> str:
        """Aggiunge imports necessari per i moduli"""
        import_lines = []
        
        for module_name in self.runtime_manager.registered_modules:
            if not module_name.startswith('__'):
                import_lines.append(f"from . import {module_name}")
        
        if import_lines:
            imports_section = '\n'.join(import_lines) + '\n\n'
            # Inserisci dopo gli imports esistenti
            existing_imports_end = class_code.find("class CompiledStrategy")
            class_code = class_code[:existing_imports_end] + imports_section + class_code[existing_imports_end:]
        
        return class_code


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def setup_enhanced_easylanguage_compiler(base_compiler):
    """
    Setup completo del compilatore con supporto modulare
    """
    
    # Crea runtime manager
    runtime_manager = EasyLanguageRuntimeManager()
    
    # Auto-discover moduli
    runtime_manager.auto_discover_modules("./")
    
    # Registra manualmente moduli specifici se necessario
    runtime_manager.register_module("easylanguage_market_data")
    
    # Crea generators modulari
    modular_generators = ModularCodeGenerators(
        base_compiler.code_generators,
        runtime_manager
    )
    
    # Sostituisci il generatore nel compiler
    base_compiler.modular_generators = modular_generators
    
    return base_compiler, runtime_manager

def print_available_functions(runtime_manager: EasyLanguageRuntimeManager):
    """Stampa tutte le funzioni disponibili organizzate per modulo"""
    
    functions_by_module = runtime_manager.get_available_functions()
    
    print("=== AVAILABLE EASYLANGUAGE FUNCTIONS ===\n")
    
    for module_name, functions in functions_by_module.items():
        print(f"📦 {module_name.upper()}")
        print("-" * (len(module_name) + 4))
        
        for func_info in functions:
            print(f"  • {func_info['name']}{func_info['signature']}")
            print(f"    {func_info['description']}")
        
        print()


# =============================================================================
# INTEGRATION INSTRUCTIONS
# =============================================================================

"""
ISTRUZIONI PER INTEGRARE I TUOI MODULI:

1. NAMING CONVENTION:
   - I tuoi file devono iniziare con 'easylanguage_'
   - Esempio: easylanguage_market_data.py, easylanguage_indicators.py

2. STRUCTURE REQUIREMENTS:
   - Funzioni di init: __init_module_name__()  
   - Funzioni di update: update_module_optimized()
   - Funzioni EasyLanguage: nomi normali (VWAP, InsideBid, etc.)

3. SETUP AUTOMATICO:
   ```python
   # Nel tuo main compiler
   compiler, runtime_manager = setup_enhanced_easylanguage_compiler(base_compiler)
   
   # Genera strategia con tutti i moduli
   strategy_code = compiler.modular_generators.build_enhanced_strategy_class_with_modules()
   ```

4. VERIFICA INTEGRAZIONE:
   ```python
   # Vedi quali funzioni sono disponibili
   print_available_functions(runtime_manager)
   
   # Controlla compatibilità
   is_compatible = runtime_manager.validate_module_compatibility("easylanguage_market_data")
   ```

5. USO NELLA STRATEGIA:
   - Tutte le funzioni dei tuoi moduli saranno automaticamente disponibili
   - Esempio: self.VWAP(), self.InsideBid(), self.RelativeVolume()
   - Init e update automatici gestiti dal runtime manager
"""