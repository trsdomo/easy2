# =============================================================================
# MATHEMATICAL BASE FUNCTIONS - EASYLANGUAGE INTEGRATION
# Core Mathematical Operations with High Performance and Error Handling
# =============================================================================

import math
import cmath
from functools import lru_cache
import warnings
import random

class MathTracker:
    """
    Tracker per operazioni matematiche con caching e ottimizzazioni
    """
    def __init__(self):
        # Performance counters
        self._calculations_count = 0
        self._cache_hits = 0
        
        # Error tracking
        self._error_count = 0
        self._last_error = None
        
        # Constants cache
        self._constants = {
            'PI': math.pi,
            'E': math.e,
            'SQRT2': math.sqrt(2),
            'SQRT3': math.sqrt(3),
            'LN2': math.log(2),
            'LN10': math.log(10),
            'LOG2E': math.log2(math.e),
            'LOG10E': math.log10(math.e)
        }
        
        # Random seed management
        self._random_seed = None

def __init_math_base__(self):
    """
    Inizializzazione sistema matematico di base ottimizzato
    """
    self.math_tracker = MathTracker()
    
    # Performance flags
    self._use_fast_math = True
    self._use_cache = True
    
    print("✓ Mathematical Base Functions initialized")

# =============================================================================
# BASIC ARITHMETIC FUNCTIONS
# =============================================================================

def AbsValue(self, value):
    """Returns absolute value of number"""
    try:
        self.math_tracker._calculations_count += 1
        return abs(float(value))
        
    except Exception as e:
        self._handle_math_error("AbsValue", e)
        return 0.0

def Square(self, value):
    """Returns square of number (value²)"""
    try:
        self.math_tracker._calculations_count += 1
        val = float(value)
        return val * val
        
    except Exception as e:
        self._handle_math_error("Square", e)
        return 0.0

def SquareRoot(self, value):
    """Returns square root of number"""
    try:
        self.math_tracker._calculations_count += 1
        val = float(value)
        
        if val < 0:
            self._handle_math_error("SquareRoot", "Negative value")
            return 0.0
            
        return math.sqrt(val)
        
    except Exception as e:
        self._handle_math_error("SquareRoot", e)
        return 0.0

def Power(self, base, exponent):
    """Returns base raised to exponent (base^exponent)"""
    try:
        self.math_tracker._calculations_count += 1
        base_val = float(base)
        exp_val = float(exponent)
        
        # Handle special cases
        if base_val == 0 and exp_val < 0:
            self._handle_math_error("Power", "Division by zero")
            return float('inf')
        
        return math.pow(base_val, exp_val)
        
    except Exception as e:
        self._handle_math_error("Power", e)
        return 0.0

def Exp(self, value):
    """Returns e raised to the power of value (e^value)"""
    try:
        self.math_tracker._calculations_count += 1
        val = float(value)
        
        # Prevent overflow
        if val > 700:  # e^700 is close to float limit
            return float('inf')
        if val < -700:
            return 0.0
            
        return math.exp(val)
        
    except Exception as e:
        self._handle_math_error("Exp", e)
        return 1.0

# =============================================================================
# LOGARITHMIC FUNCTIONS
# =============================================================================

def Log(self, value):
    """Returns natural logarithm (base e)"""
    try:
        self.math_tracker._calculations_count += 1
        val = float(value)
        
        if val <= 0:
            self._handle_math_error("Log", "Non-positive value")
            return float('-inf') if val == 0 else float('nan')
            
        return math.log(val)
        
    except Exception as e:
        self._handle_math_error("Log", e)
        return 0.0

def Log10(self, value):
    """Returns logarithm base 10"""
    try:
        self.math_tracker._calculations_count += 1
        val = float(value)
        
        if val <= 0:
            self._handle_math_error("Log10", "Non-positive value")
            return float('-inf') if val == 0 else float('nan')
            
        return math.log10(val)
        
    except Exception as e:
        self._handle_math_error("Log10", e)
        return 0.0

def LogN(self, value, base):
    """Returns logarithm with specified base"""
    try:
        self.math_tracker._calculations_count += 1
        val = float(value)
        base_val = float(base)
        
        if val <= 0 or base_val <= 0 or base_val == 1:
            self._handle_math_error("LogN", "Invalid value or base")
            return float('nan')
            
        return math.log(val) / math.log(base_val)
        
    except Exception as e:
        self._handle_math_error("LogN", e)
        return 0.0

# =============================================================================
# TRIGONOMETRIC FUNCTIONS
# =============================================================================

def Sine(self, value):
    """Returns sine of angle in radians"""
    try:
        self.math_tracker._calculations_count += 1
        return math.sin(float(value))
        
    except Exception as e:
        self._handle_math_error("Sine", e)
        return 0.0

def Cosine(self, value):
    """Returns cosine of angle in radians"""
    try:
        self.math_tracker._calculations_count += 1
        return math.cos(float(value))
        
    except Exception as e:
        self._handle_math_error("Cosine", e)
        return 1.0

def Tangent(self, value):
    """Returns tangent of angle in radians"""
    try:
        self.math_tracker._calculations_count += 1
        val = float(value)
        
        # Check for values close to π/2 + nπ where tan is undefined
        normalized = val % math.pi
        if abs(normalized - math.pi/2) < 1e-10:
            return float('inf')
            
        return math.tan(val)
        
    except Exception as e:
        self._handle_math_error("Tangent", e)
        return 0.0

# Degree versions
def SineDegrees(self, degrees):
    """Returns sine of angle in degrees"""
    return self.Sine(math.radians(float(degrees)))

def CosineDegrees(self, degrees):
    """Returns cosine of angle in degrees"""
    return self.Cosine(math.radians(float(degrees)))

def TangentDegrees(self, degrees):
    """Returns tangent of angle in degrees"""
    return self.Tangent(math.radians(float(degrees)))

# =============================================================================
# INVERSE TRIGONOMETRIC FUNCTIONS
# =============================================================================

def ArcSine(self, value):
    """Returns arcsine in radians"""
    try:
        self.math_tracker._calculations_count += 1
        val = float(value)
        
        if val < -1 or val > 1:
            self._handle_math_error("ArcSine", "Value outside [-1,1] range")
            return float('nan')
            
        return math.asin(val)
        
    except Exception as e:
        self._handle_math_error("ArcSine", e)
        return 0.0

def ArcCosine(self, value):
    """Returns arccosine in radians"""
    try:
        self.math_tracker._calculations_count += 1
        val = float(value)
        
        if val < -1 or val > 1:
            self._handle_math_error("ArcCosine", "Value outside [-1,1] range")
            return float('nan')
            
        return math.acos(val)
        
    except Exception as e:
        self._handle_math_error("ArcCosine", e)
        return math.pi / 2

def ArcTangent(self, value):
    """Returns arctangent in radians"""
    try:
        self.math_tracker._calculations_count += 1
        return math.atan(float(value))
        
    except Exception as e:
        self._handle_math_error("ArcTangent", e)
        return 0.0

def ArcTangent2(self, y, x):
    """Returns arctangent of y/x in correct quadrant"""
    try:
        self.math_tracker._calculations_count += 1
        return math.atan2(float(y), float(x))
        
    except Exception as e:
        self._handle_math_error("ArcTangent2", e)
        return 0.0

# Degree versions
def ArcSineDegrees(self, value):
    """Returns arcsine in degrees"""
    return math.degrees(self.ArcSine(value))

def ArcCosineDegrees(self, value):
    """Returns arccosine in degrees"""
    return math.degrees(self.ArcCosine(value))

def ArcTangentDegrees(self, value):
    """Returns arctangent in degrees"""
    return math.degrees(self.ArcTangent(value))

# =============================================================================
# HYPERBOLIC FUNCTIONS
# =============================================================================

def SinH(self, value):
    """Returns hyperbolic sine"""
    try:
        self.math_tracker._calculations_count += 1
        val = float(value)
        
        # Prevent overflow
        if abs(val) > 700:
            return float('inf') if val > 0 else float('-inf')
            
        return math.sinh(val)
        
    except Exception as e:
        self._handle_math_error("SinH", e)
        return 0.0

def CosH(self, value):
    """Returns hyperbolic cosine"""
    try:
        self.math_tracker._calculations_count += 1
        val = float(value)
        
        # Prevent overflow
        if abs(val) > 700:
            return float('inf')
            
        return math.cosh(val)
        
    except Exception as e:
        self._handle_math_error("CosH", e)
        return 1.0

def TanH(self, value):
    """Returns hyperbolic tangent"""
    try:
        self.math_tracker._calculations_count += 1
        return math.tanh(float(value))
        
    except Exception as e:
        self._handle_math_error("TanH", e)
        return 0.0

# =============================================================================
# ROUNDING AND INTEGER FUNCTIONS
# =============================================================================

def IntPortion(self, value):
    """Returns integer portion of number (truncate towards zero)"""
    try:
        self.math_tracker._calculations_count += 1
        val = float(value)
        return float(int(val))
        
    except Exception as e:
        self._handle_math_error("IntPortion", e)
        return 0.0

def FracPortion(self, value):
    """Returns fractional portion of number"""
    try:
        self.math_tracker._calculations_count += 1
        val = float(value)
        return val - float(int(val))
        
    except Exception as e:
        self._handle_math_error("FracPortion", e)
        return 0.0

def Round(self, value, digits=0):
    """Returns number rounded to specified decimal places"""
    try:
        self.math_tracker._calculations_count += 1
        val = float(value)
        digits = int(digits)
        return round(val, digits)
        
    except Exception as e:
        self._handle_math_error("Round", e)
        return 0.0

def Ceiling(self, value):
    """Returns smallest integer >= value"""
    try:
        self.math_tracker._calculations_count += 1
        return math.ceil(float(value))
        
    except Exception as e:
        self._handle_math_error("Ceiling", e)
        return 0.0

def Floor(self, value):
    """Returns largest integer <= value"""
    try:
        self.math_tracker._calculations_count += 1
        return math.floor(float(value))
        
    except Exception as e:
        self._handle_math_error("Floor", e)
        return 0.0

# =============================================================================
# COMPARISON FUNCTIONS
# =============================================================================

def MinValue(self, value1, value2):
    """Returns smaller of two values"""
    try:
        self.math_tracker._calculations_count += 1
        return min(float(value1), float(value2))
        
    except Exception as e:
        self._handle_math_error("MinValue", e)
        return 0.0

def MaxValue(self, value1, value2):
    """Returns larger of two values"""
    try:
        self.math_tracker._calculations_count += 1
        return max(float(value1), float(value2))
        
    except Exception as e:
        self._handle_math_error("MaxValue", e)
        return 0.0

def MinList(self, *values):
    """Returns minimum value from list of values"""
    try:
        self.math_tracker._calculations_count += 1
        if not values:
            return 0.0
        return min(float(v) for v in values)
        
    except Exception as e:
        self._handle_math_error("MinList", e)
        return 0.0

def MaxList(self, *values):
    """Returns maximum value from list of values"""
    try:
        self.math_tracker._calculations_count += 1
        if not values:
            return 0.0
        return max(float(v) for v in values)
        
    except Exception as e:
        self._handle_math_error("MaxList", e)
        return 0.0

def Clamp(self, value, min_val, max_val):
    """Clamps value between min and max"""
    try:
        self.math_tracker._calculations_count += 1
        val = float(value)
        min_v = float(min_val)
        max_v = float(max_val)
        
        if min_v > max_v:
            min_v, max_v = max_v, min_v
            
        return max(min_v, min(val, max_v))
        
    except Exception as e:
        self._handle_math_error("Clamp", e)
        return 0.0

# =============================================================================
# SIGN AND COMPARISON FUNCTIONS
# =============================================================================

def Sign(self, value):
    """Returns sign of value: -1, 0, or 1"""
    try:
        self.math_tracker._calculations_count += 1
        val = float(value)
        
        if val > 0:
            return 1.0
        elif val < 0:
            return -1.0
        else:
            return 0.0
            
    except Exception as e:
        self._handle_math_error("Sign", e)
        return 0.0

def IsZero(self, value, tolerance=1e-10):
    """Returns True if value is approximately zero"""
    try:
        return abs(float(value)) < tolerance
        
    except Exception as e:
        self._handle_math_error("IsZero", e)
        return True

def IsEqual(self, value1, value2, tolerance=1e-10):
    """Returns True if values are approximately equal"""
    try:
        return abs(float(value1) - float(value2)) < tolerance
        
    except Exception as e:
        self._handle_math_error("IsEqual", e)
        return False

def IsNaN(self, value):
    """Returns True if value is Not a Number"""
    try:
        return math.isnan(float(value))
        
    except Exception:
        return False

def IsInfinite(self, value):
    """Returns True if value is infinite"""
    try:
        return math.isinf(float(value))
        
    except Exception:
        return False

def IsFinite(self, value):
    """Returns True if value is finite (not NaN or infinite)"""
    try:
        val = float(value)
        return math.isfinite(val)
        
    except Exception:
        return False

# =============================================================================
# RANDOM NUMBER FUNCTIONS
# =============================================================================

def Random(self):
    """Returns random number between 0 and 1"""
    try:
        self.math_tracker._calculations_count += 1
        return random.random()
        
    except Exception as e:
        self._handle_math_error("Random", e)
        return 0.5

def RandomRange(self, min_val, max_val):
    """Returns random number between min_val and max_val"""
    try:
        self.math_tracker._calculations_count += 1
        min_v = float(min_val)
        max_v = float(max_val)
        
        if min_v > max_v:
            min_v, max_v = max_v, min_v
            
        return random.uniform(min_v, max_v)
        
    except Exception as e:
        self._handle_math_error("RandomRange", e)
        return 0.0

def RandomInt(self, min_val, max_val):
    """Returns random integer between min_val and max_val (inclusive)"""
    try:
        self.math_tracker._calculations_count += 1
        min_v = int(min_val)
        max_v = int(max_val)
        
        if min_v > max_v:
            min_v, max_v = max_v, min_v
            
        return float(random.randint(min_v, max_v))
        
    except Exception as e:
        self._handle_math_error("RandomInt", e)
        return 0.0

def SetRandomSeed(self, seed):
    """Sets seed for random number generator"""
    try:
        random.seed(int(seed))
        self.math_tracker._random_seed = int(seed)
        print(f"Random seed set to: {seed}")
        
    except Exception as e:
        self._handle_math_error("SetRandomSeed", e)

# =============================================================================
# ADVANCED MATHEMATICAL FUNCTIONS
# =============================================================================

def Factorial(self, n):
    """Returns factorial of n (n!)"""
    try:
        self.math_tracker._calculations_count += 1
        n_val = int(n)
        
        if n_val < 0:
            self._handle_math_error("Factorial", "Negative value")
            return float('nan')
        
        if n_val > 170:  # Factorial overflow limit for float
            return float('inf')
            
        return float(math.factorial(n_val))
        
    except Exception as e:
        self