# =============================================================================
# STRING FUNCTIONS - EASYLANGUAGE INTEGRATION
# Complete String Manipulation with EasyLanguage Compatibility
# =============================================================================

import re
import warnings
from functools import lru_cache

class StringTracker:
    """
    Tracker per operazioni su stringhe con caching e ottimizzazioni
    """
    def __init__(self):
        # Performance counters
        self._operations_count = 0
        self._cache_hits = 0
        
        # String operation cache
        self._string_cache = {}
        
        # Error tracking
        self._error_count = 0
        self._last_error = None
        
        # Formatting templates
        self._format_templates = {
            'currency': '${:,.2f}',
            'percent': '{:.2%}',
            'scientific': '{:.2e}',
            'fixed': '{:.{}f}'
        }

def __init_string_functions__(self):
    """
    Inizializzazione sistema string functions ottimizzato
    """
    self.string_tracker = StringTracker()
    
    # Performance flags
    self._use_string_cache = True
    
    print("✓ String Functions initialized")

# =============================================================================
# NUMERIC TO STRING CONVERSION
# =============================================================================

def NumToStr(self, number, decimals=2):
    """
    Convert number to string with specified decimal places
    EasyLanguage compatible format
    """
    try:
        self.string_tracker._operations_count += 1
        
        # Handle special values
        if isinstance(number, str):
            return str(number)
        
        num_val = float(number)
        dec_val = int(max(0, decimals))
        
        # Use format string
        if dec_val == 0:
            return f"{num_val:.0f}"
        else:
            return f"{num_val:.{dec_val}f}"
            
    except Exception as e:
        self._handle_string_error("NumToStr", e)
        return "0"

def StrToNum(self, string_val):
    """
    Convert string to number
    Returns 0 if conversion fails
    """
    try:
        self.string_tracker._operations_count += 1
        
        if not string_val or string_val.strip() == "":
            return 0.0
        
        # Clean the string
        clean_str = str(string_val).strip()
        
        # Remove common formatting characters
        clean_str = clean_str.replace(',', '')
        clean_str = clean_str.replace('$', '')
        clean_str = clean_str.replace('%', '')
        
        # Handle percentage
        is_percent = '%' in str(string_val)
        
        # Convert
        result = float(clean_str)
        
        if is_percent:
            result = result / 100.0
            
        return result
        
    except Exception as e:
        self._handle_string_error("StrToNum", e)
        return 0.0

# =============================================================================
# STRING EXTRACTION FUNCTIONS
# =============================================================================

def LeftStr(self, string_val, count):
    """
    Returns leftmost count characters from string
    EasyLanguage compatible
    """
    try:
        self.string_tracker._operations_count += 1
        
        if not string_val:
            return ""
        
        str_val = str(string_val)
        count_val = max(0, int(count))
        
        return str_val[:count_val]
        
    except Exception as e:
        self._handle_string_error("LeftStr", e)
        return ""

def RightStr(self, string_val, count):
    """
    Returns rightmost count characters from string
    EasyLanguage compatible
    """
    try:
        self.string_tracker._operations_count += 1
        
        if not string_val:
            return ""
        
        str_val = str(string_val)
        count_val = max(0, int(count))
        
        if count_val >= len(str_val):
            return str_val
        
        return str_val[-count_val:] if count_val > 0 else ""
        
    except Exception as e:
        self._handle_string_error("RightStr", e)
        return ""

def MidStr(self, string_val, start, count):
    """
    Returns substring starting at position start with length count
    EasyLanguage uses 1-based indexing
    """
    try:
        self.string_tracker._operations_count += 1
        
        if not string_val:
            return ""
        
        str_val = str(string_val)
        start_val = max(1, int(start))  # EasyLanguage is 1-based
        count_val = max(0, int(count))
        
        # Convert to 0-based indexing
        start_idx = start_val - 1
        end_idx = start_idx + count_val
        
        if start_idx >= len(str_val):
            return ""
        
        return str_val[start_idx:end_idx]
        
    except Exception as e:
        self._handle_string_error("MidStr", e)
        return ""

# =============================================================================
# STRING PROPERTIES
# =============================================================================

def StrLen(self, string_val):
    """
    Returns length of string
    """
    try:
        self.string_tracker._operations_count += 1
        
        if string_val is None:
            return 0
        
        return len(str(string_val))
        
    except Exception as e:
        self._handle_string_error("StrLen", e)
        return 0

# =============================================================================
# STRING CASE CONVERSION
# =============================================================================

def UpperStr(self, string_val):
    """
    Convert string to uppercase
    """
    try:
        self.string_tracker._operations_count += 1
        
        if not string_val:
            return ""
        
        return str(string_val).upper()
        
    except Exception as e:
        self._handle_string_error("UpperStr", e)
        return str(string_val) if string_val else ""

def LowerStr(self, string_val):
    """
    Convert string to lowercase
    """
    try:
        self.string_tracker._operations_count += 1
        
        if not string_val:
            return ""
        
        return str(string_val).lower()
        
    except Exception as e:
        self._handle_string_error("LowerStr", e)
        return str(string_val) if string_val else ""

def ProperStr(self, string_val):
    """
    Convert string to proper case (Title Case)
    """
    try:
        self.string_tracker._operations_count += 1
        
        if not string_val:
            return ""
        
        return str(string_val).title()
        
    except Exception as e:
        self._handle_string_error("ProperStr", e)
        return str(string_val) if string_val else ""

# =============================================================================
# STRING SEARCHING FUNCTIONS
# =============================================================================

def InStr(self, search_in, search_for, start_pos=1):
    """
    Find position of substring within string
    Returns 0 if not found (EasyLanguage convention)
    Uses 1-based indexing like EasyLanguage
    """
    try:
        self.string_tracker._operations_count += 1
        
        if not search_in or not search_for:
            return 0
        
        str_in = str(search_in)
        str_for = str(search_for)
        start_val = max(1, int(start_pos))
        
        # Convert to 0-based for Python
        start_idx = start_val - 1
        
        if start_idx >= len(str_in):
            return 0
        
        # Find the substring
        found_pos = str_in.find(str_for, start_idx)
        
        # Convert back to 1-based, return 0 if not found
        return found_pos + 1 if found_pos >= 0 else 0
        
    except Exception as e:
        self._handle_string_error("InStr", e)
        return 0

def InStrReverse(self, search_in, search_for):
    """
    Find last occurrence of substring within string
    Returns 0 if not found
    """
    try:
        self.string_tracker._operations_count += 1
        
        if not search_in or not search_for:
            return 0
        
        str_in = str(search_in)
        str_for = str(search_for)
        
        # Find last occurrence
        found_pos = str_in.rfind(str_for)
        
        # Convert to 1-based, return 0 if not found
        return found_pos + 1 if found_pos >= 0 else 0
        
    except Exception as e:
        self._handle_string_error("InStrReverse", e)
        return 0

# =============================================================================
# STRING MANIPULATION FUNCTIONS
# =============================================================================

def StrReplace(self, source_str, find_str, replace_str, count=-1):
    """
    Replace occurrences of find_str with replace_str
    count: max number of replacements (-1 for all)
    """
    try:
        self.string_tracker._operations_count += 1
        
        if not source_str:
            return ""
        
        src = str(source_str)
        find = str(find_str)
        replace = str(replace_str)
        count_val = int(count)
        
        if count_val == -1:
            return src.replace(find, replace)
        else:
            return src.replace(find, replace, count_val)
        
    except Exception as e:
        self._handle_string_error("StrReplace", e)
        return str(source_str) if source_str else ""

def TrimStr(self, string_val):
    """
    Remove leading and trailing whitespace
    """
    try:
        self.string_tracker._operations_count += 1
        
        if not string_val:
            return ""
        
        return str(string_val).strip()
        
    except Exception as e:
        self._handle_string_error("TrimStr", e)
        return str(string_val) if string_val else ""

def TrimLeft(self, string_val):
    """
    Remove leading whitespace
    """
    try:
        self.string_tracker._operations_count += 1
        
        if not string_val:
            return ""
        
        return str(string_val).lstrip()
        
    except Exception as e:
        self._handle_string_error("TrimLeft", e)
        return str(string_val) if string_val else ""

def TrimRight(self, string_val):
    """
    Remove trailing whitespace
    """
    try:
        self.string_tracker._operations_count += 1
        
        if not string_val:
            return ""
        
        return str(string_val).rstrip()
        
    except Exception as e:
        self._handle_string_error("TrimRight", e)
        return str(string_val) if string_val else ""

def PadLeft(self, string_val, total_length, pad_char=" "):
    """
    Pad string on left to specified total length
    """
    try:
        self.string_tracker._operations_count += 1
        
        if not string_val:
            string_val = ""
        
        str_val = str(string_val)
        length = int(total_length)
        pad = str(pad_char)[0] if pad_char else " "
        
        return str_val.rjust(length, pad)
        
    except Exception as e:
        self._handle_string_error("PadLeft", e)
        return str(string_val) if string_val else ""

def PadRight(self, string_val, total_length, pad_char=" "):
    """
    Pad string on right to specified total length
    """
    try:
        self.string_tracker._operations_count += 1
        
        if not string_val:
            string_val = ""
        
        str_val = str(string_val)
        length = int(total_length)
        pad = str(pad_char)[0] if pad_char else " "
        
        return str_val.ljust(length, pad)
        
    except Exception as e:
        self._handle_string_error("PadRight", e)
        return str(string_val) if string_val else ""

def ReverseStr(self, string_val):
    """
    Reverse the string
    """
    try:
        self.string_tracker._operations_count += 1
        
        if not string_val:
            return ""
        
        return str(string_val)[::-1]
        
    except Exception as e:
        self._handle_string_error("ReverseStr", e)
        return str(string_val) if string_val else ""

# =============================================================================
# STRING FORMATTING FUNCTIONS
# =============================================================================

def FormatNumber(self, number, format_type="fixed", decimals=2):
    """
    Format number according to specified format type
    Types: fixed, currency, percent, scientific
    """
    try:
        self.string_tracker._operations_count += 1
        
        num_val = float(number)
        fmt_type = str(format_type).lower()
        dec_val = int(decimals)
        
        if fmt_type == "currency":
            return f"${num_val:,.{dec_val}f}"
        elif fmt_type == "percent":
            return f"{num_val:.{dec_val}%}"
        elif fmt_type == "scientific":
            return f"{num_val:.{dec_val}e}"
        else:  # fixed
            return f"{num_val:.{dec_val}f}"
            
    except Exception as e:
        self._handle_string_error("FormatNumber", e)
        return str(number)

def FormatDate(self, date_val, format_str="MM/DD/YYYY"):
    """
    Format date string (basic implementation)
    """
    try:
        self.string_tracker._operations_count += 1
        
        # This is a simplified version - full implementation would parse EL dates
        return str(date_val)
        
    except Exception as e:
        self._handle_string_error("FormatDate", e)
        return str(date_val)

# =============================================================================
# STRING COMPARISON FUNCTIONS
# =============================================================================

def StrCompare(self, str1, str2, case_sensitive=True):
    """
    Compare two strings
    Returns: -1 if str1 < str2, 0 if equal, 1 if str1 > str2
    """
    try:
        self.string_tracker._operations_count += 1
        
        s1 = str(str1) if str1 is not None else ""
        s2 = str(str2) if str2 is not None else ""
        
        if not case_sensitive:
            s1 = s1.lower()
            s2 = s2.lower()
        
        if s1 < s2:
            return -1
        elif s1 > s2:
            return 1
        else:
            return 0
            
    except Exception as e:
        self._handle_string_error("StrCompare", e)
        return 0

def StrEqual(self, str1, str2, case_sensitive=True):
    """
    Test if two strings are equal
    """
    try:
        return self.StrCompare(str1, str2, case_sensitive) == 0
        
    except Exception as e:
        self._handle_string_error("StrEqual", e)
        return False

def StartsWith(self, string_val, prefix):
    """
    Test if string starts with prefix
    """
    try:
        self.string_tracker._operations_count += 1
        
        if not string_val or not prefix:
            return False
        
        return str(string_val).startswith(str(prefix))
        
    except Exception as e:
        self._handle_string_error("StartsWith", e)
        return False

def EndsWith(self, string_val, suffix):
    """
    Test if string ends with suffix
    """
    try:
        self.string_tracker._operations_count += 1
        
        if not string_val or not suffix:
            return False
        
        return str(string_val).endswith(str(suffix))
        
    except Exception as e:
        self._handle_string_error("EndsWith", e)
        return False

def Contains(self, string_val, substring):
    """
    Test if string contains substring
    """
    try:
        self.string_tracker._operations_count += 1
        
        if not string_val or substring is None:
            return False
        
        return str(substring) in str(string_val)
        
    except Exception as e:
        self._handle_string_error("Contains", e)
        return False

# =============================================================================
# STRING SPLITTING AND JOINING
# =============================================================================

def StrSplit(self, string_val, delimiter=","):
    """
    Split string by delimiter - returns list as string representation
    """
    try:
        self.string_tracker._operations_count += 1
        
        if not string_val:
            return ""
        
        str_val = str(string_val)
        delim = str(delimiter)
        
        parts = str_val.split(delim)
        # Return as comma-separated for EasyLanguage compatibility
        return "|".join(parts)  # Use | to avoid confusion with comma delimiter
        
    except Exception as e:
        self._handle_string_error("StrSplit", e)
        return str(string_val) if string_val else ""

def StrJoin(self, parts_str, delimiter=","):
    """
    Join string parts with delimiter
    """
    try:
        self.string_tracker._operations_count += 1
        
        if not parts_str:
            return ""
        
        parts = str(parts_str).split("|")  # Assuming StrSplit format
        delim = str(delimiter)
        
        return delim.join(parts)
        
    except Exception as e:
        self._handle_string_error("StrJoin", e)
        return str(parts_str) if parts_str else ""

# =============================================================================
# REGULAR EXPRESSIONS (BASIC)
# =============================================================================

def RegexMatch(self, string_val, pattern):
    """
    Test if string matches regex pattern
    """
    try:
        self.string_tracker._operations_count += 1
        
        if not string_val or not pattern:
            return False
        
        return bool(re.search(str(pattern), str(string_val)))
        
    except Exception as e:
        self._handle_string_error("RegexMatch", e)
        return False

def RegexReplace(self, string_val, pattern, replacement):
    """
    Replace text using regex pattern
    """
    try:
        self.string_tracker._operations_count += 1
        
        if not string_val:
            return ""
        
        str_val = str(string_val)
        pat = str(pattern)
        repl = str(replacement)
        
        return re.sub(pat, repl, str_val)
        
    except Exception as e:
        self._handle_string_error("RegexReplace", e)
        return str(string_val) if string_val else ""

# =============================================================================
# CHARACTER FUNCTIONS
# =============================================================================

def CharAt(self, string_val, position):
    """
    Get character at position (1-based indexing like EasyLanguage)
    """
    try:
        self.string_tracker._operations_count += 1
        
        if not string_val:
            return ""
        
        str_val = str(string_val)
        pos = int(position)
        
        if pos < 1 or pos > len(str_val):
            return ""
        
        return str_val[pos - 1]  # Convert to 0-based
        
    except Exception as e:
        self._handle_string_error("CharAt", e)
        return ""

def AsciiCode(self, char):
    """
    Get ASCII code of character
    """
    try:
        self.string_tracker._operations_count += 1
        
        if not char:
            return 0
        
        char_str = str(char)
        if len(char_str) == 0:
            return 0
        
        return ord(char_str[0])
        
    except Exception as e:
        self._handle_string_error("AsciiCode", e)
        return 0

def CharFromAscii(self, ascii_code):
    """
    Get character from ASCII code
    """
    try:
        self.string_tracker._operations_count += 1
        
        code = int(ascii_code)
        
        if code < 0 or code > 127:  # Standard ASCII range
            return ""
        
        return chr(code)
        
    except Exception as e:
        self._handle_string_error("CharFromAscii", e)
        return ""

def IsDigit(self, char):
    """
    Test if character is a digit
    """
    try:
        if not char:
            return False
        
        char_str = str(char)
        return len(char_str) > 0 and char_str[0].isdigit()
        
    except Exception:
        return False

def IsAlpha(self, char):
    """
    Test if character is alphabetic
    """
    try:
        if not char:
            return False
        
        char_str = str(char)
        return len(char_str) > 0 and char_str[0].isalpha()
        
    except Exception:
        return False

def IsAlphaNum(self, char):
    """
    Test if character is alphanumeric
    """
    try:
        if not char:
            return False
        
        char_str = str(char)
        return len(char_str) > 0 and char_str[0].isalnum()
        
    except Exception:
        return False

# =============================================================================
# STRING VALIDATION FUNCTIONS
# =============================================================================

def IsNumeric(self, string_val):
    """
    Test if string represents a valid number
    """
    try:
        if not string_val:
            return False
        
        str_val = str(string_val).strip()
        
        # Handle empty string
        if not str_val:
            return False
        
        # Try to convert to float
        float(str_val)
        return True
        
    except ValueError:
        return False
    except Exception:
        return False

def IsEmpty(self, string_val):
    """
    Test if string is empty or None
    """
    return not string_val or str(string_val).strip() == ""

def IsWhitespace(self, string_val):
    """
    Test if string contains only whitespace
    """
    try:
        if not string_val:
            return True
        
        return str(string_val).isspace()
        
    except Exception:
        return False

# =============================================================================
# ADVANCED STRING FUNCTIONS
# =============================================================================

def Repeat(self, string_val, count):
    """
    Repeat string count times
    """
    try:
        self.string_tracker._operations_count += 1
        
        if not string_val:
            return ""
        
        str_val = str(string_val)
        count_val = max(0, int(count))
        
        return str_val * count_val
        
    except Exception as e:
        self._handle_string_error("Repeat", e)
        return ""

def WordCount(self, string_val):
    """
    Count words in string
    """
    try:
        self.string_tracker._operations_count += 1
        
        if not string_val:
            return 0
        
        str_val = str(string_val).strip()
        if not str_val:
            return 0
        
        # Split by whitespace and count non-empty parts
        words = [word for word in str_val.split() if word]
        return len(words)
        
    except Exception as e:
        self._handle_string_error("WordCount", e)
        return 0

def WordAt(self, string_val, word_index):
    """
    Get word at specified index (1-based)
    """
    try:
        self.string_tracker._operations_count += 1
        
        if not string_val:
            return ""
        
        str_val = str(string_val).strip()
        idx = int(word_index)
        
        if idx < 1:
            return ""
        
        words = [word for word in str_val.split() if word]
        
        if idx > len(words):
            return ""
        
        return words[idx - 1]  # Convert to 0-based
        
    except Exception as e:
        self._handle_string_error("WordAt", e)
        return ""

# =============================================================================
# ERROR HANDLING
# =============================================================================

def _handle_string_error(self, function_name, error):
    """Centralized error handling for string functions"""
    try:
        self.string_tracker._error_count += 1
        self.string_tracker._last_error = f"{function_name}: {str(error)}"
        
        # Log error if in debug mode
        if hasattr(self, '_debug_mode') and self._debug_mode:
            print(f"String Error in {function_name}: {error}")
            
    except Exception:
        pass  # Prevent error handling from causing more errors

def GetLastStringError(self):
    """Returns last string error message"""
    return self.string_tracker._last_error

def GetStringErrorCount(self):
    """Returns total count of string errors"""
    return self.string_tracker._error_count

def ResetStringErrorCount(self):
    """Reset string error counter"""
    self.string_tracker._error_count = 0
    self.string_tracker._last_error = None

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_string_stats(self):
    """Get performance statistics for string operations"""
    try:
        cache_hit_rate = 0.0
        if self.string_tracker._operations_count > 0:
            cache_hit_rate = (self.string_tracker._cache_hits / self.string_tracker._operations_count) * 100.0
        
        return {
            'operations_count': self.string_tracker._operations_count,
            'cache_hits': self.string_tracker._cache_hits,
            'cache_hit_rate_percent': cache_hit_rate,
            'error_count': self.string_tracker._error_count,
            'last_error': self.string_tracker._last_error,
            'cache_size': len(self.string_tracker._string_cache)
        }
        
    except Exception as e:
        print(f"Error getting string stats: {e}")
        return {}

def reset_string_cache(self):
    """Reset string operation counters"""
    try:
        self.string_tracker._operations_count = 0
        self.string_tracker._cache_hits = 0
        self.string_tracker._string_cache.clear()
        print("✓ String functions cache reset completed")
        
    except Exception as e:
        print(f"Error resetting string cache: {e}")

# =============================================================================
# TEST SUITE
# =============================================================================

def run_string_functions_test():
    """Comprehensive test suite for string functions"""
    print("=== STRING FUNCTIONS TEST ===")
    
    class MockStrategy:
        def __init__(self):
            self.__init_string_functions__()
            self._debug_mode = True
    
    strategy = MockStrategy()
    
    # Add all string methods
    import types
    for name, obj in globals().items():
        if callable(obj) and (name.startswith('__init_string') or 
                             name.startswith('_handle_string') or
                             name in ['NumToStr', 'StrToNum', 'LeftStr', 'RightStr', 'MidStr',
                                     'StrLen', 'UpperStr', 'LowerStr', 'ProperStr',
                                     'InStr', 'InStrReverse', 'StrReplace', 'TrimStr',
                                     'PadLeft', 'PadRight', 'ReverseStr', 'FormatNumber',
                                     'StrCompare', 'StrEqual', 'StartsWith', 'EndsWith', 'Contains',
                                     'StrSplit', 'StrJoin', 'CharAt', 'AsciiCode', 'CharFromAscii',
                                     'IsDigit', 'IsAlpha', 'IsNumeric', 'IsEmpty', 'Repeat',
                                     'WordCount', 'WordAt', 'get_string_stats', 'reset_string_cache']):
            setattr(strategy, name, types.MethodType(obj, strategy))
    
    test_string = "Hello, World! This is a Test String 123"
    
    print("Testing Basic String Functions:")
    print(f"  Test String: '{test_string}'")
    print(f"  Length: {strategy.StrLen(test_string)}")
    print(f"  UpperCase: '{strategy.UpperStr(test_string)}'")
    print(f"  LowerCase: '{strategy.LowerStr(test_string)}'")
    print(f"  ProperCase: '{strategy.ProperStr('hello world')}'")
    
    print("\nTesting String Extraction:")
    print(f"  LeftStr(5): '{strategy.LeftStr(test_string, 5)}'")
    print(f"  RightStr(6): '{strategy.RightStr(test_string, 6)}'")
    print(f"  MidStr(8, 5): '{strategy.MidStr(test_string, 8, 5)}'")
    
    print("\nTesting String Searching:")
    print(f"  InStr('World'): {strategy.InStr(test_string, 'World')}")
    print(f"  InStr('xyz'): {strategy.InStr(test_string, 'xyz')}")
    print(f"  StartsWith('Hello'): {strategy.StartsWith(test_string, 'Hello')}")
    print(f"  EndsWith('123'): {strategy.EndsWith(test_string, '123')}")
    print(f"  Contains('Test'): {strategy.Contains(test_string, 'Test')}")
    
    print("\nTesting String Manipulation:")
    print(f"  Replace 'Test' with 'Demo': '{strategy.StrReplace(test_string, 'Test', 'Demo')}'")
    print(f"  Trim '  spaced  ': '{strategy.TrimStr('  spaced  ')}'")
    print(f"  PadLeft('Hi', 10): '{strategy.PadLeft('Hi', 10)}'")
    print(f"  PadRight('Hi', 10, '*'): '{strategy.PadRight('Hi', 10, '*')}'")
    print(f"  Reverse 'Hello': '{strategy.ReverseStr('Hello')}'")
    
    print("\nTesting Number Conversion:")
    print(f"  NumToStr(123.456, 2): '{strategy.NumToStr(123.456, 2)}'")
    print(f"  NumToStr(123.456, 0): '{strategy.NumToStr(123.456, 0)}'")
    print(f"  StrToNum('123.45'): {strategy.StrToNum('123.45')}")
    print(f"  StrToNum('$1,234.56'): {strategy.StrToNum('$1,234.56')}")
    print(f"  StrToNum('invalid'): {strategy.StrToNum('invalid')}")
    
    print("\nTesting Character Functions:")
    print(f"  CharAt(test_string, 1): '{strategy.CharAt(test_string, 1)}'")
    print(f"  CharAt(test_string, 7): '{strategy.CharAt(test_string, 7)}'")
    print(f"  AsciiCode('A'): {strategy.AsciiCode('A')}")
    print(f"  CharFromAscii(65): '{strategy.CharFromAscii(65)}'")
    print(f"  IsDigit('5'): {strategy.IsDigit('5')}")
    print(f"  IsAlpha('A'): {strategy.IsAlpha('A')}")
    
    print("\nTesting Validation Functions:")
    print(f"  IsNumeric('123.45'): {strategy.IsNumeric('123.45')}")
    print(f"  IsNumeric('abc'): {strategy.IsNumeric('abc')}")
    print(f"  IsEmpty(''): {strategy.IsEmpty('')}")
    print(f"  IsEmpty('test'): {strategy.IsEmpty('test')}")
    
    print("\nTesting Advanced Functions:")
    print(f"  Repeat('Hi', 3): '{strategy.Repeat('Hi', 3)}'")
    print(f"  WordCount(test_string): {strategy.WordCount(test_string)}")
    print(f"  WordAt(test_string, 3): '{strategy.WordAt(test_string, 3)}'")
    print(f"  WordAt(test_string, 5): '{strategy.WordAt(test_string, 5)}'")
    
    print("\nTesting String Comparison:")
    print(f"  StrCompare('abc', 'def'): {strategy.StrCompare('abc', 'def')}")
    print(f"  StrCompare('abc', 'ABC', False): {strategy.StrCompare('abc', 'ABC', False)}")
    print(f"  StrEqual('test', 'TEST', False): {strategy.StrEqual('test', 'TEST', False)}")
    
    print("\nTesting Format Functions:")
    print(f"  FormatNumber(1234.5, 'currency'): '{strategy.FormatNumber(1234.5, 'currency')}'")
    print(f"  FormatNumber(0.125, 'percent'): '{strategy.FormatNumber(0.125, 'percent')}'")
    print(f"  FormatNumber(1234.5, 'scientific'): '{strategy.FormatNumber(1234.5, 'scientific')}'")
    
    print("\nPerformance Test:")
    import time
    start_time = time.time()
    
    # Perform many string operations
    for i in range(1000):
        test_val = f"Test string {i}"
        upper_val = strategy.UpperStr(test_val)
        len_val = strategy.StrLen(upper_val)
        left_val = strategy.LeftStr(upper_val, 5)
    
    end_time = time.time()
    print(f"  1000 operations in {(end_time - start_time)*1000:.2f} ms")
    
    # Performance statistics
    stats = strategy.get_string_stats()
    print(f"\nPerformance Statistics:")
    print(f"  Operations: {stats['operations_count']}")
    print(f"  Error Count: {stats['error_count']}")
    
    # Test cache reset
    strategy.reset_string_cache()
    
    print("✓ String Functions test completed")

if __name__ == "__main__":
    run_string_functions_test()