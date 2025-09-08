# ========================================================================
# SESSION INFORMATION FUNCTIONS - EasyLanguage for Backtrader
# Gestione avanzata di sessioni, orari e timezone
# ========================================================================

import datetime
import pytz
from typing import Optional, Dict, List, Tuple, Union
from enum import Enum
from dataclasses import dataclass
import calendar

class SessionType(Enum):
    """Tipi di sessione supportati"""
    REGULAR = "regular"
    ELECTRONIC = "electronic" 
    PIT = "pit"
    OVERNIGHT = "overnight"
    CUSTOM = "custom"

@dataclass
class SessionDefinition:
    """Definizione di una sessione di trading"""
    name: str
    start_time: datetime.time
    end_time: datetime.time
    timezone: pytz.BaseTzInfo
    session_type: SessionType = SessionType.REGULAR
    days_of_week: List[int] = None  # 0=Monday, 6=Sunday
    
    def __post_init__(self):
        if self.days_of_week is None:
            self.days_of_week = [0, 1, 2, 3, 4]  # Weekdays by default

class SessionTracker:
    """Helper class per tracciare le sessioni di trading"""
    
    def __init__(self, default_timezone: str = "America/New_York"):
        self.sessions = {}
        self.default_tz = pytz.timezone(default_timezone)
        self.current_session = None
        self.session_history = []
        
        # Definizioni standard
        self._setup_standard_sessions()
        
    def _setup_standard_sessions(self):
        """Setup delle sessioni standard di mercato"""
        # NYSE Regular Session
        self.add_session(
            "NYSE_REGULAR",
            datetime.time(9, 30), datetime.time(16, 0),
            "America/New_York", SessionType.REGULAR
        )
        
        # NYSE Pre-Market
        self.add_session(
            "NYSE_PREMARKET", 
            datetime.time(4, 0), datetime.time(9, 30),
            "America/New_York", SessionType.ELECTRONIC
        )
        
        # NYSE After-Hours
        self.add_session(
            "NYSE_AFTERHOURS",
            datetime.time(16, 0), datetime.time(20, 0),
            "America/New_York", SessionType.ELECTRONIC
        )
        
        # London Session
        self.add_session(
            "LONDON_REGULAR",
            datetime.time(8, 0), datetime.time(16, 30),
            "Europe/London", SessionType.REGULAR
        )
        
        # Tokyo Session
        self.add_session(
            "TOKYO_REGULAR", 
            datetime.time(9, 0), datetime.time(15, 0),
            "Asia/Tokyo", SessionType.REGULAR
        )
        
        # Forex Sessions
        self.add_session(
            "FOREX_SYDNEY",
            datetime.time(22, 0), datetime.time(7, 0),
            "Australia/Sydney", SessionType.ELECTRONIC,
            [6, 0, 1, 2, 3, 4]  # Sun-Fri
        )
        
        self.add_session(
            "FOREX_TOKYO",
            datetime.time(0, 0), datetime.time(9, 0), 
            "Asia/Tokyo", SessionType.ELECTRONIC,
            [0, 1, 2, 3, 4]  # Mon-Fri
        )
        
        self.add_session(
            "FOREX_LONDON",
            datetime.time(8, 0), datetime.time(17, 0),
            "Europe/London", SessionType.ELECTRONIC,
            [0, 1, 2, 3, 4]  # Mon-Fri
        )
        
        self.add_session(
            "FOREX_NEWYORK",
            datetime.time(13, 0), datetime.time(22, 0),
            "America/New_York", SessionType.ELECTRONIC,
            [0, 1, 2, 3, 4]  # Mon-Fri
        )
    
    def add_session(self, name: str, start_time: datetime.time, end_time: datetime.time,
                   timezone: str, session_type: SessionType = SessionType.REGULAR,
                   days_of_week: List[int] = None):
        """Aggiunge una nuova definizione di sessione"""
        tz = pytz.timezone(timezone) if isinstance(timezone, str) else timezone
        self.sessions[name] = SessionDefinition(
            name, start_time, end_time, tz, session_type, days_of_week
        )

# ========================================================================
# METODI DA AGGIUNGERE ALLA CLASSE CompiledStrategy
# ========================================================================

def __init_session_tracking__(self, default_timezone: str = "America/New_York"):
    """Inizializza il sistema di tracking delle sessioni"""
    self._session_tracker = SessionTracker(default_timezone)
    self._current_dt = None
    self._holidays = set()  # Set di date holiday
    
    # Setup holidays USA standard
    self._setup_standard_holidays()

def _setup_standard_holidays(self):
    """Setup dei giorni festivi standard USA"""
    current_year = datetime.datetime.now().year
    for year in range(current_year - 1, current_year + 2):
        # New Year's Day
        self._holidays.add(datetime.date(year, 1, 1))
        
        # MLK Day (3rd Monday in January)
        jan_1 = datetime.date(year, 1, 1)
        days_ahead = 0 - jan_1.weekday()  # Monday is 0
        if days_ahead <= 0:
            days_ahead += 7
        first_monday = jan_1 + datetime.timedelta(days=days_ahead)
        mlk_day = first_monday + datetime.timedelta(weeks=2)
        self._holidays.add(mlk_day)
        
        # Presidents Day (3rd Monday in February)
        feb_1 = datetime.date(year, 2, 1)
        days_ahead = 0 - feb_1.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        first_monday = feb_1 + datetime.timedelta(days=days_ahead)
        presidents_day = first_monday + datetime.timedelta(weeks=2)
        self._holidays.add(presidents_day)
        
        # Good Friday (Easter - 2 days)
        easter = self._calculate_easter(year)
        good_friday = easter - datetime.timedelta(days=2)
        self._holidays.add(good_friday)
        
        # Memorial Day (last Monday in May)
        memorial_day = self._last_monday_of_month(year, 5)
        self._holidays.add(memorial_day)
        
        # Independence Day
        self._holidays.add(datetime.date(year, 7, 4))
        
        # Labor Day (first Monday in September)
        sep_1 = datetime.date(year, 9, 1)
        days_ahead = 0 - sep_1.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        labor_day = sep_1 + datetime.timedelta(days=days_ahead)
        self._holidays.add(labor_day)
        
        # Thanksgiving (4th Thursday in November)
        nov_1 = datetime.date(year, 11, 1)
        days_ahead = 3 - nov_1.weekday()  # Thursday is 3
        if days_ahead < 0:
            days_ahead += 7
        first_thursday = nov_1 + datetime.timedelta(days=days_ahead)
        thanksgiving = first_thursday + datetime.timedelta(weeks=3)
        self._holidays.add(thanksgiving)
        
        # Christmas
        self._holidays.add(datetime.date(year, 12, 25))

def _calculate_easter(self, year: int) -> datetime.date:
    """Calcola la data della Pasqua per un anno"""
    # Algoritmo di Gauss per il calcolo della Pasqua
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return datetime.date(year, month, day)

def _last_monday_of_month(self, year: int, month: int) -> datetime.date:
    """Trova l'ultimo lunedì del mese"""
    # Ultimo giorno del mese
    last_day = calendar.monthrange(year, month)[1]
    last_date = datetime.date(year, month, last_day)
    
    # Trova l'ultimo lunedì
    days_back = (last_date.weekday() - 0) % 7
    return last_date - datetime.timedelta(days=days_back)

# ========================================================================
# SESSION INFORMATION FUNCTIONS
# ========================================================================

def SessionStartTime(self, session: str, occurrence: int = 0) -> float:
    """Restituisce l'orario di inizio della sessione in formato EasyLanguage time"""
    try:
        if session not in self._session_tracker.sessions:
            self._handle_error(f"Unknown session: {session}")
            return 0.0
        
        session_def = self._session_tracker.sessions[session]
        current_date = self.data.datetime.date(0)
        
        # Trova la N-esima occorrenza della sessione
        target_date = self._find_session_occurrence(session, current_date, occurrence)
        if not target_date:
            return 0.0
        
        # Combina data e ora di inizio
        start_datetime = datetime.datetime.combine(target_date, session_def.start_time)
        start_datetime = session_def.timezone.localize(start_datetime)
        
        # Converti in formato EasyLanguage time (HHMM formato decimale)
        return self._time_to_easylanguage_format(session_def.start_time)
        
    except Exception as e:
        self._handle_error(f"SessionStartTime error: {e}")
        return 0.0

def SessionEndTime(self, session: str, occurrence: int = 0) -> float:
    """Restituisce l'orario di fine della sessione in formato EasyLanguage time"""
    try:
        if session not in self._session_tracker.sessions:
            self._handle_error(f"Unknown session: {session}")
            return 0.0
        
        session_def = self._session_tracker.sessions[session]
        current_date = self.data.datetime.date(0)
        
        # Trova la N-esima occorrenza della sessione
        target_date = self._find_session_occurrence(session, current_date, occurrence)
        if not target_date:
            return 0.0
        
        # Converti in formato EasyLanguage time
        return self._time_to_easylanguage_format(session_def.end_time)
        
    except Exception as e:
        self._handle_error(f"SessionEndTime error: {e}")
        return 0.0

def IsInSession(self, session: str) -> bool:
    """Verifica se il momento corrente è dentro la sessione specificata"""
    try:
        if session not in self._session_tracker.sessions:
            return False
        
        session_def = self._session_tracker.sessions[session]
        current_dt = self.data.datetime.datetime(0)
        
        # Converti al timezone della sessione
        if current_dt.tzinfo is None:
            current_dt = self._session_tracker.default_tz.localize(current_dt)
        
        session_dt = current_dt.astimezone(session_def.timezone)
        current_time = session_dt.time()
        current_weekday = session_dt.weekday()
        current_date = session_dt.date()
        
        # Controlla se è un giorno di trading per questa sessione
        if current_weekday not in session_def.days_of_week:
            return False
        
        # Controlla se è un holiday
        if self._is_holiday(current_date):
            return False
        
        # Controlla se l'orario è nella sessione
        if session_def.start_time <= session_def.end_time:
            # Sessione normale (non attraversa mezzanotte)
            return session_def.start_time <= current_time <= session_def.end_time
        else:
            # Sessione overnight (attraversa mezzanotte)
            return current_time >= session_def.start_time or current_time <= session_def.end_time
            
    except Exception as e:
        self._handle_error(f"IsInSession error: {e}")
        return False

def TimeToSession(self, session: str) -> float:
    """Restituisce i minuti mancanti all'inizio della prossima sessione"""
    try:
        if session not in self._session_tracker.sessions:
            return 0.0
        
        session_def = self._session_tracker.sessions[session]
        current_dt = self.data.datetime.datetime(0)
        
        if current_dt.tzinfo is None:
            current_dt = self._session_tracker.default_tz.localize(current_dt)
        
        session_dt = current_dt.astimezone(session_def.timezone)
        
        # Trova la prossima apertura di sessione
        next_session_start = self._find_next_session_start(session_def, session_dt)
        
        # Calcola la differenza in minuti
        time_diff = next_session_start - session_dt
        return time_diff.total_seconds() / 60.0
        
    except Exception as e:
        self._handle_error(f"TimeToSession error: {e}")
        return 0.0

def TimeFromSession(self, session: str) -> float:
    """Restituisce i minuti dall'ultima chiusura della sessione"""
    try:
        if session not in self._session_tracker.sessions:
            return 0.0
        
        session_def = self._session_tracker.sessions[session]
        current_dt = self.data.datetime.datetime(0)
        
        if current_dt.tzinfo is None:
            current_dt = self._session_tracker.default_tz.localize(current_dt)
        
        session_dt = current_dt.astimezone(session_def.timezone)
        
        # Trova l'ultima chiusura di sessione
        last_session_end = self._find_last_session_end(session_def, session_dt)
        
        # Calcola la differenza in minuti
        time_diff = session_dt - last_session_end
        return time_diff.total_seconds() / 60.0
        
    except Exception as e:
        self._handle_error(f"TimeFromSession error: {e}")
        return 0.0

def IsHoliday(self, date_input: Optional[float] = None) -> bool:
    """Verifica se una data è un giorno festivo"""
    try:
        if date_input is None:
            check_date = self.data.datetime.date(0)
        else:
            check_date = self._easylanguage_date_to_python_date(date_input)
        
        return self._is_holiday(check_date)
        
    except Exception as e:
        self._handle_error(f"IsHoliday error: {e}")
        return False

def IsWeekend(self, date_input: Optional[float] = None) -> bool:
    """Verifica se una data è un weekend"""
    try:
        if date_input is None:
            check_date = self.data.datetime.date(0)
        else:
            check_date = self._easylanguage_date_to_python_date(date_input)
        
        return check_date.weekday() in [5, 6]  # Saturday, Sunday
        
    except Exception as e:
        self._handle_error(f"IsWeekend error: {e}")
        return False

def IsTradingDay(self, date_input: Optional[float] = None) -> bool:
    """Verifica se una data è un giorno di trading"""
    try:
        if date_input is None:
            check_date = self.data.datetime.date(0)
        else:
            check_date = self._easylanguage_date_to_python_date(date_input)
        
        return not self.IsWeekend(date_input) and not self.IsHoliday(date_input)
        
    except Exception as e:
        self._handle_error(f"IsTradingDay error: {e}")
        return False

def NextTradingDay(self, date_input: Optional[float] = None) -> float:
    """Restituisce il prossimo giorno di trading"""
    try:
        if date_input is None:
            start_date = self.data.datetime.date(0)
        else:
            start_date = self._easylanguage_date_to_python_date(date_input)
        
        check_date = start_date + datetime.timedelta(days=1)
        
        while not self.IsTradingDay(self._python_date_to_easylanguage_date(check_date)):
            check_date += datetime.timedelta(days=1)
            # Safety check per evitare loop infiniti
            if (check_date - start_date).days > 10:
                break
        
        return self._python_date_to_easylanguage_date(check_date)
        
    except Exception as e:
        self._handle_error(f"NextTradingDay error: {e}")
        return 0.0

def PrevTradingDay(self, date_input: Optional[float] = None) -> float:
    """Restituisce il giorno di trading precedente"""
    try:
        if date_input is None:
            start_date = self.data.datetime.date(0)
        else:
            start_date = self._easylanguage_date_to_python_date(date_input)
        
        check_date = start_date - datetime.timedelta(days=1)
        
        while not self.IsTradingDay(self._python_date_to_easylanguage_date(check_date)):
            check_date -= datetime.timedelta(days=1)
            # Safety check per evitare loop infiniti
            if (start_date - check_date).days > 10:
                break
        
        return self._python_date_to_easylanguage_date(check_date)
        
    except Exception as e:
        self._handle_error(f"PrevTradingDay error: {e}")
        return 0.0

def ConvertTimeZone(self, time_value: float, from_tz: str, to_tz: str) -> float:
    """Converte un orario da un fuso orario ad un altro"""
    try:
        # Converti time_value in datetime
        current_date = self.data.datetime.date(0)
        time_obj = self._easylanguage_time_to_python_time(time_value)
        
        # Crea datetime nel fuso di origine
        dt = datetime.datetime.combine(current_date, time_obj)
        from_timezone = pytz.timezone(from_tz)
        to_timezone = pytz.timezone(to_tz)
        
        # Localizza e converti
        localized_dt = from_timezone.localize(dt)
        converted_dt = localized_dt.astimezone(to_timezone)
        
        # Ritorna in formato EasyLanguage
        return self._time_to_easylanguage_format(converted_dt.time())
        
    except Exception as e:
        self._handle_error(f"ConvertTimeZone error: {e}")
        return time_value

def GetSessionVolume(self, session: str) -> float:
    """Restituisce il volume totale della sessione corrente"""
    try:
        # Implementazione base - andrebbe estesa con tracking specifico
        if not self.IsInSession(session):
            return 0.0
        
        # Placeholder: restituisce volume corrente
        # In una implementazione completa, si trackerebbe il volume per sessione
        return float(self.data.volume[0])
        
    except Exception as e:
        self._handle_error(f"GetSessionVolume error: {e}")
        return 0.0

# ========================================================================
# HELPER METHODS
# ========================================================================

def _find_session_occurrence(self, session: str, from_date: datetime.date, occurrence: int) -> Optional[datetime.date]:
    """Trova la N-esima occorrenza di una sessione"""
    session_def = self._session_tracker.sessions[session]
    
    if occurrence == 0:
        # Occorrenza corrente o più vicina
        if from_date.weekday() in session_def.days_of_week and not self._is_holiday(from_date):
            return from_date
        return self._find_next_trading_day(from_date, session_def.days_of_week)
    elif occurrence > 0:
        # Occorrenze future
        current_date = from_date
        count = 0
        while count < occurrence:
            current_date = self._find_next_trading_day(current_date, session_def.days_of_week)
            if current_date:
                count += 1
            else:
                return None
        return current_date
    else:
        # Occorrenze passate
        current_date = from_date
        count = 0
        while count < abs(occurrence):
            current_date = self._find_prev_trading_day(current_date, session_def.days_of_week)
            if current_date:
                count += 1
            else:
                return None
        return current_date

def _find_next_session_start(self, session_def: SessionDefinition, from_dt: datetime.datetime) -> datetime.datetime:
    """Trova la prossima apertura della sessione"""
    current_date = from_dt.date()
    current_time = from_dt.time()
    
    # Se siamo nello stesso giorno e prima dell'apertura
    if (current_date.weekday() in session_def.days_of_week and 
        not self._is_holiday(current_date) and 
        current_time < session_def.start_time):
        return datetime.datetime.combine(current_date, session_def.start_time)
    
    # Altrimenti trova il prossimo giorno di trading
    next_date = self._find_next_trading_day(current_date, session_def.days_of_week)
    if next_date:
        return datetime.datetime.combine(next_date, session_def.start_time)
    
    return from_dt  # Fallback

def _find_last_session_end(self, session_def: SessionDefinition, from_dt: datetime.datetime) -> datetime.datetime:
    """Trova l'ultima chiusura della sessione"""
    current_date = from_dt.date()
    current_time = from_dt.time()
    
    # Se siamo nello stesso giorno e dopo la chiusura
    if (current_date.weekday() in session_def.days_of_week and 
        not self._is_holiday(current_date) and 
        current_time > session_def.end_time):
        return datetime.datetime.combine(current_date, session_def.end_time)
    
    # Altrimenti trova il giorno di trading precedente
    prev_date = self._find_prev_trading_day(current_date, session_def.days_of_week)
    if prev_date:
        return datetime.datetime.combine(prev_date, session_def.end_time)
    
    return from_dt  # Fallback

def _find_next_trading_day(self, from_date: datetime.date, valid_weekdays: List[int]) -> Optional[datetime.date]:
    """Trova il prossimo giorno di trading valido"""
    check_date = from_date + datetime.timedelta(days=1)
    for _ in range(10):  # Safety limit
        if check_date.weekday() in valid_weekdays and not self._is_holiday(check_date):
            return check_date
        check_date += datetime.timedelta(days=1)
    return None

def _find_prev_trading_day(self, from_date: datetime.date, valid_weekdays: List[int]) -> Optional[datetime.date]:
    """Trova il giorno di trading precedente valido"""
    check_date = from_date - datetime.timedelta(days=1)
    for _ in range(10):  # Safety limit
        if check_date.weekday() in valid_weekdays and not self._is_holiday(check_date):
            return check_date
        check_date -= datetime.timedelta(days=1)
    return None

def _is_holiday(self, date: datetime.date) -> bool:
    """Verifica se una data è festiva"""
    # Se cade nel weekend, sposta al lunedì
    if date in self._holidays:
        return True
    
    # Se un holiday cade nel weekend, viene osservato il lunedì
    if date.weekday() == 0:  # Monday
        # Controlla se il venerdì o sabato precedenti erano holiday
        friday = date - datetime.timedelta(days=3)
        saturday = date - datetime.timedelta(days=2)
        if friday in self._holidays or saturday in self._holidays:
            return True
    
    return False

def _time_to_easylanguage_format(self, time_obj: datetime.time) -> float:
    """Converte un time object in formato EasyLanguage (HHMM.SS)"""
    return float(f"{time_obj.hour:02d}{time_obj.minute:02d}.{time_obj.second:02d}")

def _easylanguage_time_to_python_time(self, el_time: float) -> datetime.time:
    """Converte un time EasyLanguage in time object Python"""
    time_str = f"{el_time:07.2f}"  # HHMM.SS
    hhmm, ss = time_str.split('.')
    hour = int(hhmm[:2]) if len(hhmm) >= 2 else 0
    minute = int(hhmm[2:4]) if len(hhmm) >= 4 else 0
    second = int(ss) if ss else 0
    return datetime.time(hour, minute, second)

def _easylanguage_date_to_python_date(self, el_date: float) -> datetime.date:
    """Converte una data EasyLanguage in date object Python"""
    # Assume formato YYYYMMDD
    date_str = f"{int(el_date):08d}"
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    return datetime.date(year, month, day)

def _python_date_to_easylanguage_date(self, python_date: datetime.date) -> float:
    """Converte un date object Python in formato EasyLanguage"""
    return float(f"{python_date.year:04d}{python_date.month:02d}{python_date.day:02d}")

# ========================================================================
# ESEMPI DI USO IN EASYLANGUAGE
# ========================================================================

"""
ESEMPIO 1 - Trading solo durante regular session:
vars: IsRegularHours(false);

IsRegularHours = IsInSession("NYSE_REGULAR");

if IsRegularHours then begin
    // La tua logica di trading
    if Close > Average(Close, 20) then Buy;
end;

ESEMPIO 2 - Monitoraggio aperture di sessione:
vars: SessionStart(0), MinutesToOpen(0);

SessionStart = SessionStartTime("NYSE_REGULAR", 0);
MinutesToOpen = TimeToSession("NYSE_REGULAR");

if MinutesToOpen <= 30 and MinutesToOpen > 0 then begin
    Alert("NYSE opens in " + NumToStr(MinutesToOpen, 0) + " minutes");
end;

ESEMPIO 3 - Trading cross-session:
vars: 
    IsNYOpen(false), IsLondonOpen(false),
    IsTokyoOpen(false), OverlapTime(false);

IsNYOpen = IsInSession("FOREX_NEWYORK");
IsLondonOpen = IsInSession("FOREX_LONDON"); 
IsTokyoOpen = IsInSession("FOREX_TOKYO");

// Trading durante overlap London-NY
OverlapTime = IsNYOpen and IsLondonOpen;

if OverlapTime then begin
    // Strategia per periodo di alta volatilità
    if Range > Average(Range, 20) * 1.5 then begin
        if Close > High[1] then Buy;
        if Close < Low[1] then Sell Short;
    end;
end;

ESEMPIO 4 - Conversione timezone:
vars: 
    NYTime(0), LondonTime(0), TokyoTime(0),
    CurrentTime(0);

CurrentTime = Time;
NYTime = ConvertTimeZone(CurrentTime, "Europe/London", "America/New_York");
LondonTime = ConvertTimeZone(CurrentTime, "America/New_York", "Europe/London");
TokyoTime = ConvertTimeZone(CurrentTime, "America/New_York", "Asia/Tokyo");

Print("NY: ", NumToStr(NYTime, 2), 
      " London: ", NumToStr(LondonTime, 2),
      " Tokyo: ", NumToStr(TokyoTime, 2));

ESEMPIO 5 - Controllo giorni festivi e weekend:
vars: IsTradingDay(false), NextTradingDay(0);

IsTradingDay = IsTradingDay();

if not IsTradingDay then begin
    NextTradingDay = NextTradingDay();
    Print("Next trading day: ", NumToStr(NextTradingDay, 0));
    // Non effettuare trades
end else begin
    // Logica normale di trading
    if Close > Open then Buy;
end;

ESEMPIO 6 - Session-based risk management:
vars: 
    SessionVol(0), AvgSessionVol(0),
    TimeInSession(0), RiskMultiplier(1);

SessionVol = GetSessionVolume("NYSE_REGULAR");
TimeInSession = TimeFromSession("NYSE_REGULAR");

// Riduce risk nelle prime/ultime ore di sessione
if TimeInSession < 30 or TimeToSession("NYSE_REGULAR") < 30 then begin
    RiskMultiplier = 0.5;
end else begin
    RiskMultiplier = 1.0;
end;

// Usa RiskMultiplier per calcolare position size
if Close > Average(Close, 20) then 
    Buy RiskMultiplier * 100 shares;

ESEMPIO 7 - Pre-market gap detection:
vars: 
    IsPreMarket(false), RegularOpen(0),
    PreMarketHigh(0), PreMarketLow(0), Gap(0);

IsPreMarket = IsInSession("NYSE_PREMARKET");
RegularOpen = SessionStartTime("NYSE_REGULAR", 0);

if IsPreMarket then begin
    // Traccia high/low pre-market
    if High > PreMarketHigh then PreMarketHigh = High;
    if Low < PreMarketLow or PreMarketLow = 0 then PreMarketLow = Low;
end;

// All'apertura regular, calcola gap
if Time = RegularOpen then begin
    Gap = Open - Close[1];
    if AbsValue(Gap) > Average(TrueRange, 20) * 2 then begin
        Alert("Significant gap detected: " + NumToStr(Gap, 2));
    end;
end;

ESEMPIO 8 - Multi-timeframe session analysis:
vars: 
    AsianSession(false), EuropeanSession(false), 
    USSession(false), SessionCount(0);

AsianSession = IsInSession("FOREX_TOKYO");
EuropeanSession = IsInSession("FOREX_LONDON");
USSession = IsInSession("FOREX_NEWYORK");

SessionCount = 0;
if AsianSession then SessionCount = SessionCount + 1;
if EuropeanSession then SessionCount = SessionCount + 1; 
if USSession then SessionCount = USSession + 1;

// Trading solo durante overlap di almeno 2 sessioni
if SessionCount >= 2 then begin
    // Strategia momentum per alta liquidità
    if Close > Highest(High, 20) then Buy;
    if Close < Lowest(Low, 20) then Sell Short;
end;
"""

# ========================================================================
# TEST CASES
# ========================================================================

def run_session_tests():
    """Test suite per le funzioni di sessione"""
    print("Running Session Functions Tests...")
    
    # Test 1: Inizializzazione
    class MockStrategy:
        def __init__(self):
            self.__init_session_tracking__()
            
        def _handle_error(self, msg):
            print(f"Error: {msg}")
            
    strategy = MockStrategy()
    
    # Test session definitions
    assert "NYSE_REGULAR" in strategy._session_tracker.sessions
    assert "FOREX_LONDON" in strategy._session_tracker.sessions
    assert "TOKYO_REGULAR" in strategy._session_tracker.sessions
    
    print("✓ Session initialization tests passed")
    
    # Test 2: Time format conversions
    test_time = datetime.time(14, 30, 45)
    el_format = strategy._time_to_easylanguage_format(test_time)
    assert el_format == 1430.45, f"Expected 1430.45, got {el_format}"
    
    back_to_time = strategy._easylanguage_time_to_python_time(el_format)
    assert back_to_time == test_time, f"Time conversion failed"
    
    print("✓ Time format conversion tests passed")
    
    # Test 3: Date format conversions
    test_date = datetime.date(2024, 3, 15)
    el_date = strategy._python_date_to_easylanguage_date(test_date)
    assert el_date == 20240315, f"Expected 20240315, got {el_date}"
    
    back_to_date = strategy._easylanguage_date_to_python_date(el_date)
    assert back_to_date == test_date, f"Date conversion failed"
    
    print("✓ Date format conversion tests passed")
    
    # Test 4: Holiday detection
    # Test New Year's Day 2024
    new_years = datetime.date(2024, 1, 1)
    assert strategy._is_holiday(new_years), "New Year's Day should be a holiday"
    
    # Test Christmas 2024
    christmas = datetime.date(2024, 12, 25)
    assert strategy._is_holiday(christmas), "Christmas should be a holiday"
    
    # Test regular weekday
    regular_day = datetime.date(2024, 3, 15)  # Friday
    assert not strategy._is_holiday(regular_day), "Regular Friday should not be a holiday"
    
    print("✓ Holiday detection tests passed")
    
    # Test 5: Weekend detection
    saturday = datetime.date(2024, 3, 16)
    sunday = datetime.date(2024, 3, 17)
    monday = datetime.date(2024, 3, 18)
    
    el_saturday = strategy._python_date_to_easylanguage_date(saturday)
    el_sunday = strategy._python_date_to_easylanguage_date(sunday)
    el_monday = strategy._python_date_to_easylanguage_date(monday)
    
    assert strategy.IsWeekend(el_saturday), "Saturday should be weekend"
    assert strategy.IsWeekend(el_sunday), "Sunday should be weekend"
    assert not strategy.IsWeekend(el_monday), "Monday should not be weekend"
    
    print("✓ Weekend detection tests passed")
    
    # Test 6: Trading day logic
    assert not strategy.IsTradingDay(el_saturday), "Saturday should not be trading day"
    assert not strategy.IsTradingDay(el_sunday), "Sunday should not be trading day"
    assert strategy.IsTradingDay(el_monday), "Monday should be trading day"
    
    print("✓ Trading day logic tests passed")
    
    # Test 7: Timezone conversion
    ny_time = 1430.00  # 2:30 PM
    london_time = strategy.ConvertTimeZone(ny_time, "America/New_York", "Europe/London")
    # Should be 7:30 PM London time (5 hours ahead)
    expected_london = 1930.00
    assert abs(london_time - expected_london) < 100, f"Expected ~{expected_london}, got {london_time}"
    
    print("✓ Timezone conversion tests passed")
    
    print("All Session Functions Tests PASSED! ✓")

# ========================================================================
# CUSTOM SESSION DEFINITIONS
# ========================================================================

def AddCustomSession(self, name: str, start_time: str, end_time: str, 
                    timezone: str = "America/New_York", 
                    session_type: str = "REGULAR",
                    days: str = "MON,TUE,WED,THU,FRI"):
    """Aggiunge una sessione personalizzata"""
    try:
        # Parse start_time (formato "HH:MM")
        start_parts = start_time.split(":")
        start_time_obj = datetime.time(int(start_parts[0]), int(start_parts[1]))
        
        # Parse end_time
        end_parts = end_time.split(":")
        end_time_obj = datetime.time(int(end_parts[0]), int(end_parts[1]))
        
        # Parse days
        day_mapping = {
            "MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6
        }
        
        days_list = []
        for day in days.split(","):
            day = day.strip().upper()
            if day in day_mapping:
                days_list.append(day_mapping[day])
        
        # Parse session type
        session_type_obj = SessionType.REGULAR
        if session_type.upper() == "ELECTRONIC":
            session_type_obj = SessionType.ELECTRONIC
        elif session_type.upper() == "PIT":
            session_type_obj = SessionType.PIT
        elif session_type.upper() == "OVERNIGHT":
            session_type_obj = SessionType.OVERNIGHT
        elif session_type.upper() == "CUSTOM":
            session_type_obj = SessionType.CUSTOM
        
        # Aggiungi la sessione
        self._session_tracker.add_session(
            name, start_time_obj, end_time_obj, timezone, session_type_obj, days_list
        )
        
        return True
        
    except Exception as e:
        self._handle_error(f"AddCustomSession error: {e}")
        return False

def RemoveSession(self, name: str) -> bool:
    """Rimuove una sessione"""
    try:
        if name in self._session_tracker.sessions:
            del self._session_tracker.sessions[name]
            return True
        return False
        
    except Exception as e:
        self._handle_error(f"RemoveSession error: {e}")
        return False

def ListSessions(self) -> List[str]:
    """Restituisce la lista di tutte le sessioni disponibili"""
    try:
        return list(self._session_tracker.sessions.keys())
    except Exception as e:
        self._handle_error(f"ListSessions error: {e}")
        return []

def GetSessionInfo(self, session: str) -> Dict[str, Union[str, float]]:
    """Restituisce informazioni dettagliate su una sessione"""
    try:
        if session not in self._session_tracker.sessions:
            return {}
        
        session_def = self._session_tracker.sessions[session]
        
        return {
            "name": session_def.name,
            "start_time": self._time_to_easylanguage_format(session_def.start_time),
            "end_time": self._time_to_easylanguage_format(session_def.end_time),
            "timezone": str(session_def.timezone),
            "type": session_def.session_type.value,
            "days": ",".join([["MON","TUE","WED","THU","FRI","SAT","SUN"][d] for d in session_def.days_of_week])
        }
        
    except Exception as e:
        self._handle_error(f"GetSessionInfo error: {e}")
        return {}

# ========================================================================
# ADVANCED DATE/TIME UTILITIES
# ========================================================================

def AddBusinessDays(self, start_date: float, days_to_add: int) -> float:
    """Aggiunge giorni lavorativi a una data"""
    try:
        current_date = self._easylanguage_date_to_python_date(start_date)
        
        days_added = 0
        while days_added < days_to_add:
            current_date += datetime.timedelta(days=1)
            if self.IsTradingDay(self._python_date_to_easylanguage_date(current_date)):
                days_added += 1
        
        return self._python_date_to_easylanguage_date(current_date)
        
    except Exception as e:
        self._handle_error(f"AddBusinessDays error: {e}")
        return start_date

def BusinessDaysBetween(self, start_date: float, end_date: float) -> int:
    """Calcola i giorni lavorativi tra due date"""
    try:
        start = self._easylanguage_date_to_python_date(start_date)
        end = self._easylanguage_date_to_python_date(end_date)
        
        if start > end:
            start, end = end, start
        
        business_days = 0
        current_date = start
        
        while current_date <= end:
            if self.IsTradingDay(self._python_date_to_easylanguage_date(current_date)):
                business_days += 1
            current_date += datetime.timedelta(days=1)
        
        return business_days
        
    except Exception as e:
        self._handle_error(f"BusinessDaysBetween error: {e}")
        return 0

def GetQuarterStart(self, date_input: Optional[float] = None) -> float:
    """Restituisce l'inizio del trimestre per una data"""
    try:
        if date_input is None:
            check_date = self.data.datetime.date(0)
        else:
            check_date = self._easylanguage_date_to_python_date(date_input)
        
        quarter = (check_date.month - 1) // 3 + 1
        quarter_start_month = (quarter - 1) * 3 + 1
        quarter_start = datetime.date(check_date.year, quarter_start_month, 1)
        
        return self._python_date_to_easylanguage_date(quarter_start)
        
    except Exception as e:
        self._handle_error(f"GetQuarterStart error: {e}")
        return 0.0

def GetQuarterEnd(self, date_input: Optional[float] = None) -> float:
    """Restituisce la fine del trimestre per una data"""
    try:
        quarter_start = self.GetQuarterStart(date_input)
        start_date = self._easylanguage_date_to_python_date(quarter_start)
        
        # Trova l'ultimo giorno del trimestre
        if start_date.month in [1, 4, 7, 10]:  # Q1, Q2, Q3, Q4
            end_months = {1: 3, 4: 6, 7: 9, 10: 12}
            end_month = end_months[start_date.month]
            end_day = calendar.monthrange(start_date.year, end_month)[1]
            quarter_end = datetime.date(start_date.year, end_month, end_day)
        else:
            quarter_end = start_date  # Fallback
        
        return self._python_date_to_easylanguage_date(quarter_end)
        
    except Exception as e:
        self._handle_error(f"GetQuarterEnd error: {e}")
        return 0.0

def GetYearStart(self, date_input: Optional[float] = None) -> float:
    """Restituisce l'inizio dell'anno per una data"""
    try:
        if date_input is None:
            check_date = self.data.datetime.date(0)
        else:
            check_date = self._easylanguage_date_to_python_date(date_input)
        
        year_start = datetime.date(check_date.year, 1, 1)
        return self._python_date_to_easylanguage_date(year_start)
        
    except Exception as e:
        self._handle_error(f"GetYearStart error: {e}")
        return 0.0

def GetMonthStart(self, date_input: Optional[float] = None) -> float:
    """Restituisce l'inizio del mese per una data"""
    try:
        if date_input is None:
            check_date = self.data.datetime.date(0)
        else:
            check_date = self._easylanguage_date_to_python_date(date_input)
        
        month_start = datetime.date(check_date.year, check_date.month, 1)
        return self._python_date_to_easylanguage_date(month_start)
        
    except Exception as e:
        self._handle_error(f"GetMonthStart error: {e}")
        return 0.0

def DaysInMonth(self, date_input: Optional[float] = None) -> int:
    """Restituisce il numero di giorni nel mese"""
    try:
        if date_input is None:
            check_date = self.data.datetime.date(0)
        else:
            check_date = self._easylanguage_date_to_python_date(date_input)
        
        return calendar.monthrange(check_date.year, check_date.month)[1]
        
    except Exception as e:
        self._handle_error(f"DaysInMonth error: {e}")
        return 0

def IsLeapYear(self, year: int) -> bool:
    """Verifica se un anno è bisestile"""
    try:
        return calendar.isleap(year)
    except Exception as e:
        self._handle_error(f"IsLeapYear error: {e}")
        return False

if __name__ == "__main__":
    run_session_tests()

# ========================================================================
# ISTRUZIONI DI INTEGRAZIONE AVANZATE
# ========================================================================

"""
INTEGRAZIONE COMPLETA NELLA CLASSE CompiledStrategy:

1. DIPENDENZE:
   pip install pytz

2. IMPORT NECESSARI:
   import datetime
   import pytz
   from typing import Optional, Dict, List, Tuple, Union
   from enum import Enum
   from dataclasses import dataclass
   import calendar

3. INIZIALIZZAZIONE:
   Nel metodo __init__ della classe CompiledStrategy:
   
   def __init__(self):
       super().__init__()
       # ... altre inizializzazioni ...
       self.__init_session_tracking__("America/New_York")  # o timezone di default

4. PARSER EXTENSIONS:
   Aggiungi al parser EasyLanguage il riconoscimento di:
   
   SESSION_FUNCTIONS = [
       'SessionStartTime', 'SessionEndTime', 'IsInSession', 'TimeToSession',
       'TimeFromSession', 'IsHoliday', 'IsWeekend', 'IsTradingDay',
       'NextTradingDay', 'PrevTradingDay', 'ConvertTimeZone', 'GetSessionVolume',
       'AddCustomSession', 'RemoveSession', 'ListSessions', 'GetSessionInfo',
       'AddBusinessDays', 'BusinessDaysBetween', 'GetQuarterStart', 'GetQuarterEnd',
       'GetYearStart', 'GetMonthStart', 'DaysInMonth', 'IsLeapYear'
   ]

5. CODE GENERATION:
   Nel generatore di codice, traduci:
   
   SessionStartTime("NYSE_REGULAR") -> self.SessionStartTime("NYSE_REGULAR")
   IsInSession("FOREX_LONDON") -> self.IsInSession("FOREX_LONDON")
   ConvertTimeZone(Time, "EST", "GMT") -> self.ConvertTimeZone(self.Time(), "America/New_York", "GMT")

6. SESSIONI PERSONALIZZATE:
   
   // In EasyLanguage:
   AddCustomSession("MY_SESSION", "09:00", "17:00", "Europe/Rome", "REGULAR", "MON,TUE,WED,THU,FRI");
   
   // Viene tradotto in:
   self.AddCustomSession("MY_SESSION", "09:00", "17:00", "Europe/Rome", "REGULAR", "MON,TUE,WED,THU,FRI")

CONFIGURAZIONE TIMEZONE:

Per configurare timezone specifici, modifica il setup in __init__:

self.__init_session_tracking__("Europe/Rome")  # Per mercati europei
self.__init_session_tracking__("Asia/Tokyo")   # Per mercati asiatici

SESSIONI PRE-CONFIGURATE:

Il sistema include automaticamente:
- NYSE_REGULAR, NYSE_PREMARKET, NYSE_AFTERHOURS
- LONDON_REGULAR, TOKYO_REGULAR
- FOREX_SYDNEY, FOREX_TOKYO, FOREX_LONDON, FOREX_NEWYORK

PERFORMANCE CONSIDERATIONS:

- Holiday calculations sono cached automaticamente
- Session lookup è O(1)
- Timezone conversions utilizzano pytz per accuracy
- Date/time operations sono ottimizzate per Backtrader

LIMITAZIONI:

- Holiday calendar è USA-centric (estendibile)
- Sessioni overnight potrebbero richiedere logica custom
- Volume tracking per sessione richiede implementazione estesa

TESTING:

Esegui run_session_tests() per verificare tutte le funzionalità.

Per maggiori dettagli sui formati EasyLanguage date/time:
https://cdn.tradestation.com/uploads/EasyLanguage-Essentials.pdf (Chapter 8)
"""