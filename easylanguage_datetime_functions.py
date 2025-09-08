# =============================================================================
# DATE/TIME FUNCTIONS - EASYLANGUAGE INTEGRATION
# Full EasyLanguage Date/Time Compatibility with Performance Optimization
# =============================================================================

import datetime
import calendar
import math
from functools import lru_cache
import warnings

class DateTimeTracker:
    """
    Tracker ottimizzato per funzioni date/time con caching e conversioni
    """
    def __init__(self):
        # Current bar date/time cache
        self._current_date = 0
        self._current_time = 0
        self._current_datetime = None
        
        # Conversion cache
        self._date_cache = {}
        self._time_cache = {}
        self._julian_cache = {}
        
        # Performance metrics
        self._conversions_count = 0
        self._cache_hits = 0
        
        # Time zone info (default to exchange time)
        self._timezone_offset = 0  # Hours from UTC
        
    def update_current_datetime(self, el_date, el_time):
        """Aggiorna data/ora corrente in formato EasyLanguage"""
        self._current_date = int(el_date)
        self._current_time = int(el_time)
        
        # Convert to Python datetime for internal use
        try:
            year, month, day = self._parse_easylanguage_date(el_date)
            hour, minute = self._parse_easylanguage_time(el_time)
            self._current_datetime = datetime.datetime(year, month, day, hour, minute)
        except:
            self._current_datetime = datetime.datetime.now()
    
    def _parse_easylanguage_date(self, el_date):
        """Parse EasyLanguage date format (YYYYMMDD or YYYMMDD)"""
        date_str = str(int(el_date)).zfill(7)
        
        if len(date_str) == 7:  # YYYMMDD format
            if date_str.startswith('1'):
                year = 2000 + int(date_str[1:3])
            else:
                year = 1900 + int(date_str[0:2])
            month = int(date_str[-4:-2])
            day = int(date_str[-2:])
        else:  # YYYYMMDD format
            date_str = date_str.zfill(8)
            year = int(date_str[0:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
        
        # Validate
        if month < 1 or month > 12:
            month = 1
        if day < 1 or day > 31:
            day = 1
        if year < 1900:
            year = 1900
            
        return year, month, day
    
    def _parse_easylanguage_time(self, el_time):
        """Parse EasyLanguage time format (HHMM or HMM)"""
        time_str = str(int(el_time)).zfill(4)
        hour = int(time_str[0:2])
        minute = int(time_str[2:4])
        
        # Validate
        if hour < 0 or hour > 23:
            hour = 12
        if minute < 0 or minute > 59:
            minute = 0
            
        return hour, minute

def __init_datetime_functions__(self):
    """
    Inizializzazione sistema date/time ottimizzato
    """
    self.datetime_tracker = DateTimeTracker()
    
    # Integration with data source
    self._last_datetime_update = None
    
    # Performance cache
    self._datetime_cache = {}
    
    print("✓ Date/Time Functions initialized")

def _update_datetime_data(self):
    """Aggiorna i dati date/time dalla barra corrente"""
    try:
        if hasattr(self, 'data') and len(self.data.close) > 0:
            # Get current date/time from data
            current_date = self.Date() if hasattr(self, 'Date') else 20240101
            current_time = self.Time() if hasattr(self, 'Time') else 1200
            
            # Update tracker se necessario
            if (current_date, current_time) != self._last_datetime_update:
                self.datetime_tracker.update_current_datetime(current_date, current_time)
                self._last_datetime_update = (current_date, current_time)
                
    except Exception as e:
        print(f"Error updating datetime data: {e}")

# =============================================================================
# BASIC DATE/TIME FUNCTIONS (These rely on base data from EasyLanguage)
# =============================================================================

def Date(self, bars_back=0):
    """Returns date in EasyLanguage format (YYYYMMDD) for specified bar"""
    # This function should be provided by the base EasyLanguage system
    # We provide a fallback implementation
    try:
        if hasattr(self, 'data') and hasattr(self.data, 'datetime'):
            # Try to get from Backtrader data
            if bars_back == 0:
                dt = self.data.datetime.datetime(0)
            else:
                dt = self.data.datetime.datetime(-bars_back)
            return int(dt.strftime('%Y%m%d'))
        else:
            # Fallback to current date
            dt = datetime.datetime.now()
            return int(dt.strftime('%Y%m%d'))
    except:
        return 20240101

def Time(self, bars_back=0):
    """Returns time in EasyLanguage format (HHMM) for specified bar"""
    # This function should be provided by the base EasyLanguage system
    try:
        if hasattr(self, 'data') and hasattr(self.data, 'datetime'):
            # Try to get from Backtrader data
            if bars_back == 0:
                dt = self.data.datetime.datetime(0)
            else:
                dt = self.data.datetime.datetime(-bars_back)
            return int(dt.strftime('%H%M'))
        else:
            # Fallback to current time
            dt = datetime.datetime.now()
            return int(dt.strftime('%H%M'))
    except:
        return 1200

# =============================================================================
# DATE CALCULATION FUNCTIONS
# =============================================================================

def CalcDate(self, base_date, days_to_add):
    """
    Add/subtract days from EasyLanguage date
    Returns new date in EasyLanguage format
    """
    try:
        self._update_datetime_data()
        
        # Cache key for performance
        cache_key = f"calcdate_{base_date}_{days_to_add}"
        if cache_key in self.datetime_tracker._date_cache:
            self.datetime_tracker._cache_hits += 1
            return self.datetime_tracker._date_cache[cache_key]
        
        # Parse base date
        year, month, day = self.datetime_tracker._parse_easylanguage_date(base_date)
        base_datetime = datetime.date(year, month, day)
        
        # Add days
        new_date = base_datetime + datetime.timedelta(days=days_to_add)
        
        # Convert back to EasyLanguage format
        result = int(new_date.strftime('%Y%m%d'))
        
        # Cache result
        self.datetime_tracker._date_cache[cache_key] = result
        self.datetime_tracker._conversions_count += 1
        
        return result
        
    except Exception as e:
        print(f"CalcDate error: {e}")
        return int(base_date) if base_date else 20240101

def CalcTime(self, base_time, minutes_to_add):
    """
    Add/subtract minutes from EasyLanguage time
    Returns new time in EasyLanguage format
    """
    try:
        self._update_datetime_data()
        
        # Cache key for performance
        cache_key = f"calctime_{base_time}_{minutes_to_add}"
        if cache_key in self.datetime_tracker._time_cache:
            self.datetime_tracker._cache_hits += 1
            return self.datetime_tracker._time_cache[cache_key]
        
        # Parse base time
        hour, minute = self.datetime_tracker._parse_easylanguage_time(base_time)
        
        # Create datetime for calculation (use arbitrary date)
        base_datetime = datetime.datetime(2024, 1, 1, hour, minute)
        
        # Add minutes
        new_datetime = base_datetime + datetime.timedelta(minutes=minutes_to_add)
        
        # Convert back to EasyLanguage format
        result = int(new_datetime.strftime('%H%M'))
        
        # Cache result
        self.datetime_tracker._time_cache[cache_key] = result
        self.datetime_tracker._conversions_count += 1
        
        return result
        
    except Exception as e:
        print(f"CalcTime error: {e}")
        return int(base_time) if base_time else 1200

def DaysToDate(self, target_date):
    """
    Calculate days between current date and target date
    Positive = target is in future, Negative = target is in past
    """
    try:
        current_date = self.Date()
        
        # Parse dates
        curr_year, curr_month, curr_day = self.datetime_tracker._parse_easylanguage_date(current_date)
        target_year, target_month, target_day = self.datetime_tracker._parse_easylanguage_date(target_date)
        
        # Create date objects
        current_dt = datetime.date(curr_year, curr_month, curr_day)
        target_dt = datetime.date(target_year, target_month, target_day)
        
        # Calculate difference
        diff = target_dt - current_dt
        return diff.days
        
    except Exception as e:
        print(f"DaysToDate error: {e}")
        return 0

def MinutesToTime(self, target_time):
    """
    Calculate minutes between current time and target time
    """
    try:
        current_time = self.Time()
        
        # Parse times
        curr_hour, curr_minute = self.datetime_tracker._parse_easylanguage_time(current_time)
        target_hour, target_minute = self.datetime_tracker._parse_easylanguage_time(target_time)
        
        # Convert to total minutes
        current_minutes = curr_hour * 60 + curr_minute
        target_minutes = target_hour * 60 + target_minute
        
        return target_minutes - current_minutes
        
    except Exception as e:
        print(f"MinutesToTime error: {e}")
        return 0

# =============================================================================
# DATE CONVERSION FUNCTIONS
# =============================================================================

def DateToJulian(self, el_date):
    """Convert EasyLanguage date to Julian day number"""
    try:
        cache_key = f"julian_{el_date}"
        if cache_key in self.datetime_tracker._julian_cache:
            return self.datetime_tracker._julian_cache[cache_key]
        
        year, month, day = self.datetime_tracker._parse_easylanguage_date(el_date)
        
        # Julian day calculation (simplified)
        if month <= 2:
            year -= 1
            month += 12
        
        a = year // 100
        b = 2 - a + (a // 4)
        
        julian = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524
        
        # Cache result
        self.datetime_tracker._julian_cache[cache_key] = julian
        return julian
        
    except Exception as e:
        print(f"DateToJulian error: {e}")
        return 0

def JulianToDate(self, julian_day):
    """Convert Julian day number to EasyLanguage date"""
    try:
        # Julian to Gregorian conversion
        a = julian_day + 32044
        b = (4 * a + 3) // 146097
        c = a - (146097 * b) // 4
        d = (4 * c + 3) // 1461
        e = c - (1461 * d) // 4
        m = (5 * e + 2) // 153
        
        day = e - (153 * m + 2) // 5 + 1
        month = m + 3 - 12 * (m // 10)
        year = 100 * b + d - 4800 + m // 10
        
        # Convert to EasyLanguage format
        return int(f"{year:04d}{month:02d}{day:02d}")
        
    except Exception as e:
        print(f"JulianToDate error: {e}")
        return 20240101

# =============================================================================
# DATE COMPONENT EXTRACTION FUNCTIONS
# =============================================================================

def DayOfWeek(self, el_date):
    """
    Returns day of week (1=Sunday, 2=Monday, ..., 7=Saturday)
    EasyLanguage format
    """
    try:
        year, month, day = self.datetime_tracker._parse_easylanguage_date(el_date)
        date_obj = datetime.date(year, month, day)
        
        # Python weekday: 0=Monday, 6=Sunday
        # Convert to EasyLanguage: 1=Sunday, 7=Saturday
        python_weekday = date_obj.weekday()
        easylanguage_weekday = ((python_weekday + 1) % 7) + 1
        
        return easylanguage_weekday
        
    except Exception as e:
        print(f"DayOfWeek error: {e}")
        return 1

def Month(self, el_date):
    """Returns month (1-12) from EasyLanguage date"""
    try:
        year, month, day = self.datetime_tracker._parse_easylanguage_date(el_date)
        return month
        
    except Exception as e:
        print(f"Month error: {e}")
        return 1

def Day(self, el_date):
    """Returns day (1-31) from EasyLanguage date"""
    try:
        year, month, day = self.datetime_tracker._parse_easylanguage_date(el_date)
        return day
        
    except Exception as e:
        print(f"Day error: {e}")
        return 1

def Year(self, el_date):
    """Returns year (YYYY) from EasyLanguage date"""
    try:
        year, month, day = self.datetime_tracker._parse_easylanguage_date(el_date)
        return year
        
    except Exception as e:
        print(f"Year error: {e}")
        return 2024

def Hour(self, el_time):
    """Returns hour (0-23) from EasyLanguage time"""
    try:
        hour, minute = self.datetime_tracker._parse_easylanguage_time(el_time)
        return hour
        
    except Exception as e:
        print(f"Hour error: {e}")
        return 12

def Minute(self, el_time):
    """Returns minute (0-59) from EasyLanguage time"""
    try:
        hour, minute = self.datetime_tracker._parse_easylanguage_time(el_time)
        return minute
        
    except Exception as e:
        print(f"Minute error: {e}")
        return 0

# =============================================================================
# CURRENT SYSTEM DATE/TIME FUNCTIONS
# =============================================================================

def CurrentDate(self):
    """Returns current system date in EasyLanguage format"""
    try:
        return int(datetime.date.today().strftime('%Y%m%d'))
    except:
        return 20240101

def CurrentTime(self):
    """Returns current system time in EasyLanguage format"""
    try:
        return int(datetime.datetime.now().strftime('%H%M'))
    except:
        return 1200

# =============================================================================
# ADVANCED DATE FUNCTIONS
# =============================================================================

def DaysInMonth(self, el_date):
    """Returns number of days in the month of specified date"""
    try:
        year, month, day = self.datetime_tracker._parse_easylanguage_date(el_date)
        return calendar.monthrange(year, month)[1]
        
    except Exception as e:
        print(f"DaysInMonth error: {e}")
        return 30

def IsLeapYear(self, year):
    """Returns True if year is a leap year"""
    try:
        return calendar.isleap(int(year))
        
    except Exception as e:
        print(f"IsLeapYear error: {e}")
        return False

def WeekOfYear(self, el_date):
    """Returns week number of year (1-53)"""
    try:
        year, month, day = self.datetime_tracker._parse_easylanguage_date(el_date)
        date_obj = datetime.date(year, month, day)
        return date_obj.isocalendar()[1]
        
    except Exception as e:
        print(f"WeekOfYear error: {e}")
        return 1

def DayOfYear(self, el_date):
    """Returns day of year (1-366)"""
    try:
        year, month, day = self.datetime_tracker._parse_easylanguage_date(el_date)
        date_obj = datetime.date(year, month, day)
        return date_obj.timetuple().tm_yday
        
    except Exception as e:
        print(f"DayOfYear error: {e}")
        return 1

def QuarterOfYear(self, el_date):
    """Returns quarter of year (1-4)"""
    try:
        month = self.Month(el_date)
        return ((month - 1) // 3) + 1
        
    except Exception as e:
        print(f"QuarterOfYear error: {e}")
        return 1

# =============================================================================
# BUSINESS DAY FUNCTIONS
# =============================================================================

def IsBusinessDay(self, el_date):
    """Returns True if date is a business day (Monday-Friday)"""
    try:
        day_of_week = self.DayOfWeek(el_date)
        # EasyLanguage: 2=Monday, 6=Friday
        return 2 <= day_of_week <= 6
        
    except Exception as e:
        print(f"IsBusinessDay error: {e}")
        return True

def NextBusinessDay(self, el_date):
    """Returns next business day from given date"""
    try:
        current_date = el_date
        for _ in range(7):  # Maximum 7 iterations to find next business day
            current_date = self.CalcDate(current_date, 1)
            if self.IsBusinessDay(current_date):
                return current_date
        return current_date
        
    except Exception as e:
        print(f"NextBusinessDay error: {e}")
        return self.CalcDate(el_date, 1)

def PrevBusinessDay(self, el_date):
    """Returns previous business day from given date"""
    try:
        current_date = el_date
        for _ in range(7):  # Maximum 7 iterations to find prev business day
            current_date = self.CalcDate(current_date, -1)
            if self.IsBusinessDay(current_date):
                return current_date
        return current_date
        
    except Exception as e:
        print(f"PrevBusinessDay error: {e}")
        return self.CalcDate(el_date, -1)

def BusinessDaysBetween(self, start_date, end_date):
    """Count business days between two dates"""
    try:
        if start_date > end_date:
            start_date, end_date = end_date, start_date
        
        business_days = 0
        current_date = start_date
        
        while current_date <= end_date:
            if self.IsBusinessDay(current_date):
                business_days += 1
            current_date = self.CalcDate(current_date, 1)
            
            # Safety check to prevent infinite loop
            if business_days > 10000:  # Reasonable limit
                break
        
        return business_days
        
    except Exception as e:
        print(f"BusinessDaysBetween error: {e}")
        return 0

# =============================================================================
# TIME ZONE FUNCTIONS
# =============================================================================

def ConvertTimeZone(self, el_time, from_tz_offset, to_tz_offset):
    """
    Convert time between time zones
    offset in hours from UTC (e.g., -5 for EST, +1 for CET)
    """
    try:
        # Calculate offset difference in minutes
        offset_diff_hours = to_tz_offset - from_tz_offset
        offset_diff_minutes = int(offset_diff_hours * 60)
        
        # Convert time
        new_time = self.CalcTime(el_time, offset_diff_minutes)
        
        return new_time
        
    except Exception as e:
        print(f"ConvertTimeZone error: {e}")
        return el_time

def SetTimeZoneOffset(self, hours_from_utc):
    """Set default timezone offset for calculations"""
    try:
        self.datetime_tracker._timezone_offset = hours_from_utc
        print(f"Timezone offset set to {hours_from_utc} hours from UTC")
        
    except Exception as e:
        print(f"SetTimeZoneOffset error: {e}")

# =============================================================================
# FORMATTING FUNCTIONS
# =============================================================================

def FormatDate(self, el_date, format_string="MM/DD/YYYY"):
    """
    Format date according to format string
    Supported formats: MM, DD, YYYY, YY
    """
    try:
        year, month, day = self.datetime_tracker._parse_easylanguage_date(el_date)
        
        formatted = format_string
        formatted = formatted.replace("YYYY", f"{year:04d}")
        formatted = formatted.replace("YY", f"{year%100:02d}")
        formatted = formatted.replace("MM", f"{month:02d}")
        formatted = formatted.replace("DD", f"{day:02d}")
        
        return formatted
        
    except Exception as e:
        print(f"FormatDate error: {e}")
        return str(el_date)

def FormatTime(self, el_time, format_string="HH:MM"):
    """
    Format time according to format string
    Supported formats: HH, MM, H, M
    """
    try:
        hour, minute = self.datetime_tracker._parse_easylanguage_time(el_time)
        
        formatted = format_string
        formatted = formatted.replace("HH", f"{hour:02d}")
        formatted = formatted.replace("MM", f"{minute:02d}")
        formatted = formatted.replace("H", f"{hour}")
        formatted = formatted.replace("M", f"{minute}")
        
        return formatted
        
    except Exception as e:
        print(f"FormatTime error: {e}")
        return str(el_time)

# =============================================================================
# PERFORMANCE AND UTILITY FUNCTIONS
# =============================================================================

def get_datetime_stats(self):
    """Restituisce statistiche performance del sistema datetime"""
    try:
        cache_hit_rate = 0.0
        if self.datetime_tracker._conversions_count > 0:
            cache_hit_rate = (self.datetime_tracker._cache_hits / self.datetime_tracker._conversions_count) * 100.0
        
        return {
            'conversions_count': self.datetime_tracker._conversions_count,
            'cache_hits': self.datetime_tracker._cache_hits,
            'cache_hit_rate_percent': cache_hit_rate,
            'current_date': self.datetime_tracker._current_date,
            'current_time': self.datetime_tracker._current_time,
            'cache_sizes': {
                'date_cache': len(self.datetime_tracker._date_cache),
                'time_cache': len(self.datetime_tracker._time_cache),
                'julian_cache': len(self.datetime_tracker._julian_cache)
            }
        }
        
    except Exception as e:
        print(f"Error getting datetime stats: {e}")
        return {}

def reset_datetime_cache(self):
    """Reset all datetime caches"""
    try:
        self.datetime_tracker._date_cache.clear()
        self.datetime_tracker._time_cache.clear()
        self.datetime_tracker._julian_cache.clear()
        self._datetime_cache.clear()
        print("✓ DateTime cache reset completed")
        
    except Exception as e:
        print(f"Error resetting datetime cache: {e}")

# =============================================================================
# TEST SUITE
# =============================================================================

def run_datetime_test():
    """Comprehensive test suite per le funzioni datetime"""
    print("=== DATE/TIME FUNCTIONS TEST ===")
    
    class MockStrategy:
        def __init__(self):
            self.__init_datetime_functions__()
            self._test_date = 20240315  # March 15, 2024
            self._test_time = 1430      # 14:30 (2:30 PM)
        
        def Date(self, bars_back=0):
            # Mock implementation
            base_date = self._test_date
            if bars_back > 0:
                return self.CalcDate(base_date, -bars_back)
            return base_date
        
        def Time(self, bars_back=0):
            # Mock implementation
            base_time = self._test_time
            if bars_back > 0:
                return self.CalcTime(base_time, -bars_back * 60)  # 1 hour per bar
            return base_time
    
    strategy = MockStrategy()
    
    # Add all datetime methods
    import types
    for name, obj in globals().items():
        if callable(obj) and (name.startswith('__init_datetime') or 
                             name.startswith('_update_datetime') or
                             name.startswith('CalcDate') or name.startswith('CalcTime') or
                             name in ['DaysToDate', 'MinutesToTime', 'DateToJulian', 'JulianToDate',
                                     'DayOfWeek', 'Month', 'Day', 'Year', 'Hour', 'Minute',
                                     'CurrentDate', 'CurrentTime', 'DaysInMonth', 'IsLeapYear',
                                     'WeekOfYear', 'DayOfYear', 'QuarterOfYear', 'IsBusinessDay',
                                     'NextBusinessDay', 'PrevBusinessDay', 'BusinessDaysBetween',
                                     'ConvertTimeZone', 'FormatDate', 'FormatTime',
                                     'get_datetime_stats', 'reset_datetime_cache']):
            setattr(strategy, name, types.MethodType(obj, strategy))
    
    test_date = 20240315  # March 15, 2024 (Friday)
    test_time = 1430      # 2:30 PM
    
    print("Testing Basic Date/Time Functions:")
    print(f"  Test Date: {test_date} ({strategy.FormatDate(test_date, 'MM/DD/YYYY')})")
    print(f"  Test Time: {test_time} ({strategy.FormatTime(test_time, 'HH:MM')})")
    
    print("\nTesting Date Components:")
    print(f"  Year: {strategy.Year(test_date)}")
    print(f"  Month: {strategy.Month(test_date)}")
    print(f"  Day: {strategy.Day(test_date)}")
    print(f"  Day of Week: {strategy.DayOfWeek(test_date)} (1=Sun, 7=Sat)")
    print(f"  Day of Year: {strategy.DayOfYear(test_date)}")
    print(f"  Week of Year: {strategy.WeekOfYear(test_date)}")
    print(f"  Quarter: {strategy.QuarterOfYear(test_date)}")
    
    print("\nTesting Time Components:")
    print(f"  Hour: {strategy.Hour(test_time)}")
    print(f"  Minute: {strategy.Minute(test_time)}")
    
    print("\nTesting Date Calculations:")
    future_date = strategy.CalcDate(test_date, 30)
    past_date = strategy.CalcDate(test_date, -15)
    print(f"  Date + 30 days: {future_date} ({strategy.FormatDate(future_date, 'MM/DD/YYYY')})")
    print(f"  Date - 15 days: {past_date} ({strategy.FormatDate(past_date, 'MM/DD/YYYY')})")
    print(f"  Days to future date: {strategy.DaysToDate(future_date)}")
    print(f"  Days to past date: {strategy.DaysToDate(past_date)}")
    
    print("\nTesting Time Calculations:")
    future_time = strategy.CalcTime(test_time, 90)  # Add 90 minutes
    past_time = strategy.CalcTime(test_time, -45)   # Subtract 45 minutes
    print(f"  Time + 90 min: {future_time} ({strategy.FormatTime(future_time, 'HH:MM')})")
    print(f"  Time - 45 min: {past_time} ({strategy.FormatTime(past_time, 'HH:MM')})")
    print(f"  Minutes to future time: {strategy.MinutesToTime(future_time)}")
    
    print("\nTesting Julian Date Conversion:")
    julian = strategy.DateToJulian(test_date)
    back_to_date = strategy.JulianToDate(julian)
    print(f"  Julian Day: {julian}")
    print(f"  Back to Date: {back_to_date}")
    print(f"  Conversion Check: {test_date == back_to_date}")
    
    print("\nTesting Business Day Functions:")
    print(f"  Is Business Day: {strategy.IsBusinessDay(test_date)}")
    next_bday = strategy.NextBusinessDay(test_date)
    prev_bday = strategy.PrevBusinessDay(test_date)
    print(f"  Next Business Day: {next_bday} ({strategy.FormatDate(next_bday, 'MM/DD/YYYY')})")
    print(f"  Prev Business Day: {prev_bday} ({strategy.FormatDate(prev_bday, 'MM/DD/YYYY')})")
    
    # Test business days between
    start_date = 20240301  # March 1, 2024
    end_date = 20240331    # March 31, 2024
    bus_days = strategy.BusinessDaysBetween(start_date, end_date)
    print(f"  Business days in March 2024: {bus_days}")
    
    print("\nTesting Calendar Functions:")
    print(f"  Days in March 2024: {strategy.DaysInMonth(test_date)}")
    print(f"  Is 2024 leap year: {strategy.IsLeapYear(2024)}")
    print(f"  Is 2023 leap year: {strategy.IsLeapYear(2023)}")
    
    print("\nTesting Current System Date/Time:")
    print(f"  Current Date: {strategy.CurrentDate()}")
    print(f"  Current Time: {strategy.CurrentTime()}")
    
    print("\nTesting Time Zone Conversion:")
    # Convert from EST (-5) to CET (+1)
    est_time = 1400  # 2:00 PM EST
    cet_time = strategy.ConvertTimeZone(est_time, -5, 1)
    print(f"  2:00 PM EST = {strategy.FormatTime(cet_time, 'HH:MM')} CET")
    
    print("\nTesting Format Functions:")
    print(f"  US Format: {strategy.FormatDate(test_date, 'MM/DD/YYYY')}")
    print(f"  EU Format: {strategy.FormatDate(test_date, 'DD/MM/YYYY')}")
    print(f"  ISO Format: {strategy.FormatDate(test_date, 'YYYY-MM-DD')}")
    print(f"  12h Time: {strategy.FormatTime(test_time, 'H:MM')}")
    print(f"  24h Time: {strategy.FormatTime(test_time, 'HH:MM')}")
    
    print("\nPerformance Test:")
    import time
    start_time = time.time()
    
    # Perform multiple calculations
    for i in range(100):
        calc_date = strategy.CalcDate(test_date, i)
        calc_time = strategy.CalcTime(test_time, i * 15)
        day_of_week = strategy.DayOfWeek(calc_date)
        julian = strategy.DateToJulian(calc_date)
    
    end_time = time.time()
    print(f"  100 calculations in {(end_time - start_time)*1000:.2f} ms")
    
    # Performance statistics
    stats = strategy.get_datetime_stats()
    print(f"\nPerformance Statistics:")
    print(f"  Conversions: {stats['conversions_count']}")
    print(f"  Cache Hits: {stats['cache_hits']}")
    print(f"  Cache Hit Rate: {stats['cache_hit_rate_percent']:.1f}%")
    print(f"  Cache Sizes: {stats['cache_sizes']}")
    
    # Test cache reset
    strategy.reset_datetime_cache()
    
    print("✓ Date/Time Functions test completed")

if __name__ == "__main__":
    run_datetime_test()