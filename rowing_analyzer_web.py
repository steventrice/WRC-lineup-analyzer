#!/usr/bin/env python3
"""
Lineup Sandbox - Streamlit Web App
Analyzes potential boat lineups using Paul's Law and boat physics
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import math
import statistics
import io
import html
import json
import base64
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from pathlib import Path
import re

# Google Sheets integration
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False



# =============================================================================
# CONFIGURATION
# =============================================================================

def get_app_password():
    """Get password from Streamlit secrets or fall back to default for local dev"""
    try:
        return st.secrets["app_password"]
    except (KeyError, FileNotFoundError):
        # Fallback for local development
        return "doubleshot"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_bool(value, default=False):
    """Parse boolean value that might be True/False or 'TRUE'/'FALSE' string.

    Google Sheets checkboxes can be exported as:
    - Python bool (True/False)
    - String ("TRUE"/"FALSE")

    Using bool("FALSE") returns True (non-empty string is truthy), so we need
    to explicitly check for string values.
    """
    if pd.isna(value):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().upper() == 'TRUE'
    return bool(value)

# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class Score:
    """Represents an erg score at a specific distance"""
    distance: int
    time_seconds: float
    split_500m: float
    watts: float
    source_sheet: str

    def format_split(self) -> str:
        mins = int(self.split_500m // 60)
        secs = self.split_500m % 60
        return f"{mins}:{secs:04.1f}"


@dataclass
class Rower:
    """Represents a rower with their attributes and scores"""
    name: str
    age: int
    gender: str
    side_port: bool = True
    side_starboard: bool = True
    side_coxswain: bool = False
    pref_order: str = ""
    scores: Dict[int, 'Score'] = field(default_factory=dict)
    regatta_signups: Dict[str, bool] = field(default_factory=dict)

    def add_score(self, score: 'Score'):
        """Add score, keeping best time per distance"""
        if score.distance not in self.scores:
            self.scores[score.distance] = score
        elif score.time_seconds < self.scores[score.distance].time_seconds:
            self.scores[score.distance] = score

    def get_closest_score(self, target_distance: int) -> Optional['Score']:
        """Get the score closest to target distance for projection"""
        if not self.scores:
            return None
        closest_dist = min(self.scores.keys(),
                           key=lambda d: abs(d - target_distance))
        return self.scores[closest_dist]

    def side_preference_str(self) -> str:
        """Return side preference as a string like 'P', 'S', 'PS', 'X', etc.

        If pref_order is set (from Pref column), use that order.
        Otherwise default to P, S, X order.
        """
        # Build set of allowed preferences based on checkboxes
        allowed = set()
        if self.side_port:
            allowed.add('P')
        if self.side_starboard:
            allowed.add('S')
        if self.side_coxswain:
            allowed.add('X')

        if not allowed:
            return '-'

        # If pref_order is specified, use that order (filtering to only allowed chars)
        if self.pref_order:
            # Extract only P, S, X characters from pref_order in the order they appear
            ordered = [c for c in self.pref_order.upper() if c in allowed]
            if ordered:
                return ''.join(ordered)

        # Default order: P, S, X
        parts = []
        if 'P' in allowed:
            parts.append('P')
        if 'S' in allowed:
            parts.append('S')
        if 'X' in allowed:
            parts.append('X')
        return ''.join(parts)

    def is_attending(self, regatta_name: str) -> bool:
        return self.regatta_signups.get(regatta_name, False)

    def scores_summary(self) -> str:
        """Return formatted summary of available scores"""
        if not self.scores:
            return "No scores"
        parts = []
        for d in sorted(self.scores.keys()):
            if d >= 1000:
                parts.append(f"{d//1000}K")
            else:
                parts.append(f"{d}m")
        return ", ".join(parts)

    def get_power_law_data_points(self) -> List[Tuple[int, float]]:
        """Get all (distance, watts) pairs for Power Law fitting"""
        return [(score.distance, score.watts) for score in self.scores.values()]

    def get_two_closest_data_points(self, target_distance: int) -> List[Tuple[int, float]]:
        """Get the two best data points for Power Law projection.

        First preference: One point below and one above target (interpolation)
        Second preference: Two closest points (extrapolation)
        """
        all_points = self.get_power_law_data_points()
        if len(all_points) < 2:
            return all_points

        # Separate points into below and above target
        below = [(d, w) for d, w in all_points if d < target_distance]
        above = [(d, w) for d, w in all_points if d > target_distance]

        # First preference: bracket the target (one below, one above)
        if below and above:
            # Get closest point from each side
            closest_below = max(below, key=lambda p: p[0])  # Highest distance below target
            closest_above = min(above, key=lambda p: p[0])  # Lowest distance above target
            return [closest_below, closest_above]

        # Second preference: two closest points (extrapolation)
        sorted_points = sorted(all_points, key=lambda p: abs(p[0] - target_distance))
        return sorted_points[:2]

    def get_power_law_fit(self, target_distance: int = None) -> Optional[Tuple[float, float]]:
        """Fit Power Law to this rower's data. Returns (k, b) or None.
        If target_distance provided, uses only the two closest data points."""
        if target_distance is not None:
            data_points = self.get_two_closest_data_points(target_distance)
        else:
            data_points = self.get_power_law_data_points()
        return PhysicsEngine.fit_power_law(data_points)

    def project_split_power_law(self, target_distance: int) -> Optional[float]:
        """Project split at target distance using Power Law fit from two closest points."""
        fit = self.get_power_law_fit(target_distance)
        if not fit:
            return None
        k, b = fit
        predicted_watts = PhysicsEngine.power_law_projection(k, b, target_distance)
        return PhysicsEngine.watts_to_split(predicted_watts)


@dataclass
class RegattaEvent:
    """Represents an event at a regatta"""
    regatta: str           # "Vancouver Spring Sprints"
    day: str               # "Sunday, March 8, 2026"
    event_number: int
    event_time: str        # "8:30 AM"
    event_name: str        # "Mixed Masters 8+"
    include: bool          # Targeted event
    priority: bool         # Priority event


@dataclass
class EventEntry:
    """Represents a lineup entered into an event"""
    regatta: str
    day: str
    event_number: int
    event_name: str
    event_time: str
    entry_number: int      # Sequential (1, 2, 3...)
    boat_class: str
    category: str          # "Men's B", "Mixed C", etc.
    rowers: List[str]      # In seat order
    timestamp: str


# =============================================================================
# EVENT ELIGIBILITY HELPERS
# =============================================================================

def format_event_time(time_str: str) -> str:
    """Format event time, removing seconds if present. '08:30:00' -> '8:30 AM'"""
    if not time_str:
        return time_str
    try:
        # Try parsing as HH:MM:SS
        if time_str.count(':') == 2:
            parts = time_str.split(':')
            hour = int(parts[0])
            minute = parts[1]
            am_pm = "AM" if hour < 12 else "PM"
            if hour > 12:
                hour -= 12
            elif hour == 0:
                hour = 12
            return f"{hour}:{minute} {am_pm}"
        # Already formatted or HH:MM
        return time_str
    except:
        return time_str


def normalize_time_format(time_str: str) -> str:
    """Normalize time to standard format: '8:30 AM' (no leading zero, with AM/PM)

    Handles: '8:30 AM', '08:30 AM', '8:30:00 AM', '08:30:00', '8:30', etc.
    """
    if not time_str:
        return time_str

    time_str = str(time_str).strip()

    try:
        # Check if already has AM/PM
        has_ampm = 'AM' in time_str.upper() or 'PM' in time_str.upper()

        # Extract AM/PM if present
        am_pm = None
        time_part = time_str
        if has_ampm:
            time_upper = time_str.upper()
            if 'AM' in time_upper:
                am_pm = 'AM'
                time_part = time_str.upper().replace('AM', '').strip()
            elif 'PM' in time_upper:
                am_pm = 'PM'
                time_part = time_str.upper().replace('PM', '').strip()

        # Parse the time part (could be H:MM, HH:MM, H:MM:SS, HH:MM:SS)
        parts = time_part.split(':')
        hour = int(parts[0])
        minute = parts[1].zfill(2)[:2]  # Ensure 2 digits, take first 2

        # Determine AM/PM if not provided (assume 24-hour format)
        if am_pm is None:
            if hour >= 12:
                am_pm = 'PM'
                if hour > 12:
                    hour -= 12
            else:
                am_pm = 'AM'
                if hour == 0:
                    hour = 12

        # Remove leading zero from hour
        return f"{hour}:{minute} {am_pm}"
    except:
        return time_str


def normalize_day_format(day_str: str) -> str:
    """Normalize day to standard format: 'Sunday, March 8, 2026' (no leading zero on day)

    Handles: 'Sunday, March 8, 2026', 'Sunday, March 08, 2026', '2026-03-08', etc.
    """
    if not day_str:
        return day_str

    day_str = str(day_str).strip()

    try:
        from datetime import datetime

        # Try to parse various formats
        dt = None

        # Format: "Sunday, March 8, 2026" or "Sunday, March 08, 2026"
        for fmt in ["%A, %B %d, %Y", "%A, %B %-d, %Y"]:
            try:
                dt = datetime.strptime(day_str, fmt)
                break
            except:
                pass

        # Format: "2026-03-08" or "2026-03-08 00:00:00"
        if dt is None:
            if day_str[:10].count('-') == 2:
                try:
                    dt = datetime.strptime(day_str[:10], "%Y-%m-%d")
                except:
                    pass

        # Format: "March 8, 2026"
        if dt is None:
            for fmt in ["%B %d, %Y", "%B %-d, %Y"]:
                try:
                    dt = datetime.strptime(day_str, fmt)
                    break
                except:
                    pass

        if dt:
            # Format without leading zero on day
            # Python's strftime doesn't have a cross-platform way to avoid leading zeros
            # so we do it manually
            day_name = dt.strftime("%A")
            month_name = dt.strftime("%B")
            day_num = dt.day  # Integer, no leading zero
            year = dt.year
            return f"{day_name}, {month_name} {day_num}, {year}"

        return day_str
    except:
        return day_str


def names_match(roster_name: str, entry_name: str) -> bool:
    """Check if a roster name matches an entry name, handling shortened names.

    Matches if:
    - Exact match
    - First names match AND one is a prefix of the other (handles shortened names)
    """
    if not roster_name or not entry_name:
        return False
    if roster_name == entry_name:
        return True
    # Check if first names match and one is prefix of other
    roster_first = roster_name.split()[0].lower()
    entry_first = entry_name.split()[0].lower()
    if roster_first == entry_first:
        if (roster_name.lower().startswith(entry_name.lower()) or
            entry_name.lower().startswith(roster_name.lower())):
            return True
    return False


def is_name_in_list(roster_name: str, entry_names: list) -> bool:
    """Check if a roster name matches any name in a list of entry names."""
    for entry_name in entry_names:
        if names_match(roster_name, entry_name):
            return True
    return False


def parse_event_gender(event_name: str) -> Optional[str]:
    """Extract gender from event name. Returns 'M', 'W', or 'Mix'."""
    name_lower = event_name.lower()
    if 'mixed' in name_lower:
        return 'Mix'
    elif 'women' in name_lower or "women's" in name_lower:
        return 'W'
    elif 'men' in name_lower or "men's" in name_lower:
        return 'M'
    return None


def parse_event_boat_class(event_name: str) -> Optional[str]:
    """Extract boat class from event name."""
    # Look for standard boat classes
    boat_patterns = ['8+', '4+', '4-', '4x', '2+', '2-', '2x', '1x']
    for boat in boat_patterns:
        if boat in event_name:
            return boat
    return None


def parse_event_category(event_name: str) -> Optional[str]:
    """Extract masters category from event name.

    Handles both formats:
    - Letter categories: 'Masters A', 'Masters B/C' -> 'A', 'B'
    - Age categories: 'Masters 30+', 'Masters 40+' -> '30+', '40+'
    """
    # First try age-based format: "30+", "40+", "50+", etc.
    age_match = re.search(r'(\d{2})\+', event_name)
    if age_match:
        return f"{age_match.group(1)}+"

    # Then try letter format: "Masters X" or "Masters X/Y" patterns
    letter_match = re.search(r'Masters\s+([A-J]{1,2}(?:/[A-J])?)', event_name, re.IGNORECASE)
    if letter_match:
        # Return first category letter (e.g., "B" from "B/C")
        cat = letter_match.group(1).upper()
        return cat.split('/')[0] if '/' in cat else cat

    # Check for "Open" events (no age restriction)
    if 'open' in event_name.lower():
        return 'Open'
    return None


def get_event_shorthand(event_name: str) -> str:
    """Generate compact shorthand from event name (e.g., 'Mixed Masters 8+' -> 'Mx8+')."""
    gender = parse_event_gender(event_name)
    boat = parse_event_boat_class(event_name)

    # Map gender to prefix
    prefix = {'M': 'M', 'W': 'W', 'Mix': 'Mx'}.get(gender, '')
    boat_str = boat if boat else ''

    return f"{prefix}{boat_str}"


def get_category_min_age(category: str) -> int:
    """Get minimum average age for a masters category.

    Handles both formats:
    - Letter categories: 'A' -> 27, 'B' -> 36, etc.
    - Age categories: '30+' -> 30, '40+' -> 40, etc.
    """
    # Handle age-based format first (e.g., "30+", "40+")
    if category and category.endswith('+'):
        try:
            return int(category[:-1])
        except ValueError:
            pass

    # Handle letter-based format
    category_ages = {
        'AA': 21, 'A': 27, 'B': 36, 'C': 43, 'D': 50,
        'E': 55, 'F': 60, 'G': 65, 'H': 70, 'I': 75, 'J': 80,
        'Open': 0
    }
    return category_ages.get(category, 0)


def get_category_max_age(category: str) -> int:
    """Get maximum average age for a masters category (exclusive)."""
    category_ages = {
        'AA': 27, 'A': 36, 'B': 43, 'C': 50, 'D': 55,
        'E': 60, 'F': 65, 'G': 70, 'H': 75, 'I': 80, 'J': 999,
        'Open': 999
    }
    return category_ages.get(category, 999)


def is_lineup_eligible_for_event(
    lineup_gender: str,      # 'M', 'W', or 'Mix'
    lineup_avg_age: float,
    lineup_boat_class: str,  # '8+', '4+', etc.
    event: RegattaEvent
) -> bool:
    """Check if a lineup is eligible for an event.

    Rules:
    - Gender must match exactly
    - Category: can race "down" (younger/faster) but not "up" (older/slower)
    - Boat class must match exactly
    """
    event_gender = parse_event_gender(event.event_name)
    event_boat = parse_event_boat_class(event.event_name)
    event_category = parse_event_category(event.event_name)

    # Gender must match (if specified in event)
    if event_gender and lineup_gender != event_gender:
        return False

    # Boat class must match (if specified in event)
    if event_boat and lineup_boat_class != event_boat:
        return False

    # Category check: lineup can race "down" but not "up"
    # Racing "down" means a YOUNGER (lower avg age) crew racing in an OLDER category
    # E.g., a 40-year-old avg crew (category B) can race in Masters A (younger), but not C (older)
    if event_category and event_category != 'Open':
        event_min_age = get_category_min_age(event_category)
        # Lineup must have avg age >= event's minimum age requirement
        # This allows racing "down" (younger crew in older category is NOT allowed)
        # Wait, let me re-read the requirement...
        # "b ages can race down (to A) but not up (to C)"
        # So a B crew (36-42 avg) can race in A events (27-35) but not C events (43-49)
        # This means: lineup_category can be OLDER than event_category
        # Or: lineup_avg_age can be HIGHER than event requires
        # So we check: lineup_avg_age >= event_min_age (lineup is old enough or older)
        if lineup_avg_age < event_min_age:
            return False

    return True


# =============================================================================
# TIME PARSING
# =============================================================================

class TimeParser:
    """Parse various time formats into seconds"""

    @staticmethod
    def parse(time_val: Any) -> Optional[float]:
        """Parse time from various formats to seconds"""
        if pd.isna(time_val):
            return None

        if isinstance(time_val, (int, float)):
            return float(time_val)

        time_str = str(time_val).strip()
        if not time_str or time_str.lower() == 'nan':
            return None

        try:
            return float(time_str)
        except ValueError:
            pass

        patterns = [
            r'^(\d{1,2}):(\d{2})\.(\d+)$',
            r'^(\d{1,2}):(\d{2}\.\d+)$',
            r'^(\d{1,2}):(\d{2})$',
            r'^0?(\d{1,2}):(\d{2}):(\d{2})$',
        ]

        for pattern in patterns:
            match = re.match(pattern, time_str)
            if match:
                groups = match.groups()
                if len(groups) == 3 and ':' in time_str and time_str.count(':') == 1:
                    mins = int(groups[0])
                    secs = int(groups[1])
                    decimal = float(f"0.{groups[2]}")
                    return mins * 60 + secs + decimal
                elif len(groups) == 3:
                    hours = int(groups[0])
                    mins = int(groups[1])
                    secs = int(groups[2])
                    return hours * 3600 + mins * 60 + secs
                elif len(groups) == 2:
                    mins = int(groups[0])
                    secs = float(groups[1])
                    return mins * 60 + secs

        return None

    @staticmethod
    def parse_split(split_val: Any) -> Optional[float]:
        """Parse 500m split time to seconds"""
        if pd.isna(split_val):
            return None

        split_str = str(split_val).strip()

        match = re.match(r'^0?(\d{1,2}):(\d{2})\.?(\d*)$', split_str)
        if match:
            mins = int(match.group(1))
            secs = int(match.group(2))
            decimal = float(f"0.{match.group(3)}") if match.group(3) else 0
            return mins * 60 + secs + decimal

        try:
            return float(split_str)
        except ValueError:
            pass

        return None


# =============================================================================
# PHYSICS ENGINE
# =============================================================================

class PhysicsEngine:
    """Rowing physics calculations using Paul's Law and Power Law"""

    @staticmethod
    def calculate_500m_split(total_time: float, distance: int) -> float:
        """Calculate 500m split from total time and distance"""
        return (total_time / distance) * 500

    @staticmethod
    def split_to_watts(split_seconds: float) -> float:
        """Convert 500m split to watts using Concept2 formula"""
        pace_ratio = split_seconds / 500
        return 2.80 / (pace_ratio ** 3)

    @staticmethod
    def watts_to_split(watts: float) -> float:
        """Convert watts back to 500m split"""
        if watts <= 0:
            return float('inf')
        return 500 * (2.80 / watts) ** (1/3)

    @staticmethod
    def time_to_watts(time_seconds: float, distance: int) -> float:
        """Convert total time and distance to watts.
        Formula: Watts = 2.8 × (Seconds/Meters)^3"""
        if time_seconds <= 0 or distance <= 0:
            return 0
        pace = time_seconds / distance  # seconds per meter
        return 2.8 / (pace ** 3)

    @staticmethod
    def watts_to_time(watts: float, distance: int) -> float:
        """Convert watts and distance back to total time.
        Formula: Seconds = (Watts/2.8)^(1/3) × Distance"""
        if watts <= 0:
            return float('inf')
        pace = (2.8 / watts) ** (1/3)  # seconds per meter
        return pace * distance

    @staticmethod
    def pauls_law_projection(known_split: float, known_dist: int, target_dist: int) -> float:
        """
        Apply Paul's Law to project split at different distance.
        Paul's Law: Projected_Split = Known_Split + 6 * log2(Target_Dist / Known_Dist)
        """
        if known_dist == target_dist:
            return known_split
        if known_dist <= 0 or target_dist <= 0:
            return known_split

        distance_ratio = target_dist / known_dist
        split_adjustment = 6 * math.log2(distance_ratio)
        return known_split + split_adjustment

    @staticmethod
    def fit_power_law(data_points: List[Tuple[int, float]]) -> Optional[Tuple[float, float]]:
        """
        Fit Power Law to multiple (distance, watts) data points.
        Equation: Watts = k × Distance^b

        Uses least squares fit on log-transformed data.
        Returns (k, b) tuple, or None if insufficient data.
        """
        if len(data_points) < 2:
            return None

        # Filter out invalid points
        valid_points = [(d, w) for d, w in data_points if d > 0 and w > 0]
        if len(valid_points) < 2:
            return None

        # For 2 points, use exact solution
        if len(valid_points) == 2:
            d1, w1 = valid_points[0]
            d2, w2 = valid_points[1]

            # b = (ln(w2) - ln(w1)) / (ln(d2) - ln(d1))
            ln_d1, ln_d2 = math.log(d1), math.log(d2)
            ln_w1, ln_w2 = math.log(w1), math.log(w2)

            if abs(ln_d2 - ln_d1) < 0.001:  # Distances too similar
                return None

            b = (ln_w2 - ln_w1) / (ln_d2 - ln_d1)
            k = math.exp(ln_w1 - b * ln_d1)
            return (k, b)

        # For 3+ points, use least squares on log-transformed data
        # ln(W) = ln(k) + b * ln(D)
        n = len(valid_points)
        sum_ln_d = sum(math.log(d) for d, w in valid_points)
        sum_ln_w = sum(math.log(w) for d, w in valid_points)
        sum_ln_d_sq = sum(math.log(d) ** 2 for d, w in valid_points)
        sum_ln_d_ln_w = sum(math.log(d) * math.log(w) for d, w in valid_points)

        denominator = n * sum_ln_d_sq - sum_ln_d ** 2
        if abs(denominator) < 0.001:
            return None

        b = (n * sum_ln_d_ln_w - sum_ln_d * sum_ln_w) / denominator
        ln_k = (sum_ln_w - b * sum_ln_d) / n
        k = math.exp(ln_k)

        return (k, b)

    @staticmethod
    def power_law_projection(k: float, b: float, target_dist: int) -> float:
        """
        Predict watts at target distance using fitted Power Law.
        TargetWatts = k × TargetDistance^b
        """
        if target_dist <= 0:
            return 0
        return k * (target_dist ** b)

    @staticmethod
    def calculate_boat_split(watts_list: List[float]) -> float:
        """Calculate boat split by averaging watts, then converting back."""
        if not watts_list:
            return 0
        avg_watts = statistics.mean(watts_list)
        return PhysicsEngine.watts_to_split(avg_watts)


# =============================================================================
# DATA LOADING
# =============================================================================

class RosterManager:
    """Manages loading and accessing rower data from Excel"""

    K_FACTORS = {
        '1x': 0.02, '2x': 0.02, '2-': 0.02,
        '4x': 0.02, '4+': 0.02, '4-': 0.02,
        '8+': 0.02
    }

    # Common name variations for fuzzy matching
    NAME_VARIATIONS = {
        'alex': ['alexander', 'alexis'],
        'alexander': ['alex'],
        'alexis': ['alex'],
        'mike': ['michael'],
        'michael': ['mike'],
        'kat': ['kathryn', 'katherine', 'kate', 'kathy'],
        'kathryn': ['kat', 'kate', 'kathy'],
        'katherine': ['kat', 'kate', 'kathy'],
        'kate': ['kat', 'kathryn', 'katherine'],
        'kathy': ['kat', 'kathryn', 'katherine'],
        'jon': ['jonathan', 'john'],
        'jonathan': ['jon', 'john'],
        'john': ['jon', 'jonathan'],
        'jim': ['james', 'jimmy'],
        'james': ['jim', 'jimmy'],
        'bob': ['robert', 'rob'],
        'robert': ['bob', 'rob'],
        'rob': ['robert', 'bob'],
        'bill': ['william', 'will'],
        'william': ['bill', 'will'],
        'will': ['william', 'bill'],
        'dick': ['richard', 'rick'],
        'richard': ['dick', 'rick'],
        'rick': ['richard', 'dick'],
        'dan': ['daniel', 'danny'],
        'daniel': ['dan', 'danny'],
        'steve': ['steven', 'stephen'],
        'steven': ['steve', 'stephen'],
        'stephen': ['steve', 'steven'],
        'chris': ['christopher'],
        'christopher': ['chris'],
        'matt': ['matthew'],
        'matthew': ['matt'],
        'tom': ['thomas', 'tommy'],
        'thomas': ['tom', 'tommy'],
        'jen': ['jennifer', 'jenny'],
        'jennifer': ['jen', 'jenny'],
        'jenny': ['jen', 'jennifer'],
        'liz': ['elizabeth', 'beth', 'lizzy'],
        'elizabeth': ['liz', 'beth', 'lizzy'],
        'beth': ['elizabeth', 'liz'],
        'sam': ['samuel', 'samantha'],
        'samuel': ['sam'],
        'samantha': ['sam'],
        'nick': ['nicholas'],
        'nicholas': ['nick'],
        'tony': ['anthony'],
        'anthony': ['tony'],
        'ed': ['edward', 'eddie'],
        'edward': ['ed', 'eddie'],
        'joe': ['joseph'],
        'joseph': ['joe'],
        'lindsey': ['lindsay'],
        'lindsay': ['lindsey'],
        'carlee': ['carly', 'carlie'],
        'carly': ['carlee', 'carlie'],
        'sophia': ['sophie'],
        'sophie': ['sophia'],
        'cami': ['camille', 'camilla'],
        'camille': ['cami'],
    }

    def __init__(self):
        self.rowers: Dict[str, Rower] = {}
        self.regattas: List[str] = []
        self.regatta_display_names: Dict[str, str] = {}
        self.load_log: List[str] = []
        self.name_map: Dict[str, str] = {}  # Maps variations to canonical names
        self.regatta_events: Dict[str, List[RegattaEvent]] = {}  # keyed by "RegattaName|Day"
        self.club_boats: List[str] = []  # List of club boat names

    def log(self, msg: str):
        self.load_log.append(msg)

    def _build_name_map(self):
        """Build a map of name variations to canonical roster names"""
        self.name_map = {}
        for canonical_name in self.rowers.keys():
            # Add exact match
            self.name_map[canonical_name.lower()] = canonical_name

            # Parse first and last name
            parts = canonical_name.split()
            if len(parts) >= 2:
                first = parts[0].lower()
                last = ' '.join(parts[1:])

                # Add variations of first name + last name
                variations = self.NAME_VARIATIONS.get(first, [])
                for var in variations:
                    var_name = f"{var.title()} {last}"
                    self.name_map[var_name.lower()] = canonical_name

    def find_rower(self, name: str) -> Optional[str]:
        """Find a rower by name, using fuzzy matching if needed.
        Returns the canonical name from the roster, or None if not found."""
        if not name:
            return None

        # Try exact match first
        if name in self.rowers:
            return name

        # Try case-insensitive match
        name_lower = name.lower().strip()
        if name_lower in self.name_map:
            return self.name_map[name_lower]

        return None

    def load_from_excel(self, filepath: str) -> bool:
        """Load all data from the Excel spreadsheet"""
        try:
            self.log(f"Loading from: {filepath}")
            xl = pd.ExcelFile(filepath)

            self._load_roster(xl)
            self._build_name_map()  # Build fuzzy matching map
            self._load_regatta_signups(xl)
            self._load_score_sheets(xl)
            self._load_regatta_events(xl)

            self.log(f"Loaded {len(self.rowers)} rowers total")
            return True

        except Exception as e:
            self.log(f"ERROR: {e}")
            return False

    def load_from_google_sheets(self, spreadsheet_id: str, credentials_dict: dict) -> bool:
        """Load all data from a Google Sheet"""
        try:
            if not GSPREAD_AVAILABLE:
                self.log("ERROR: gspread not installed")
                return False

            self.log(f"Loading from Google Sheets: {spreadsheet_id}")

            # Authenticate with Google
            scopes = [
                'https://www.googleapis.com/auth/spreadsheets.readonly',
                'https://www.googleapis.com/auth/drive.readonly'
            ]
            creds = Credentials.from_service_account_info(credentials_dict, scopes=scopes)
            client = gspread.authorize(creds)

            # Open the spreadsheet
            spreadsheet = client.open_by_key(spreadsheet_id)
            self.log(f"Opened spreadsheet: {spreadsheet.title}")

            # Create a wrapper that mimics pd.ExcelFile behavior
            class GoogleSheetsWrapper:
                def __init__(self, spreadsheet):
                    self._spreadsheet = spreadsheet
                    self.sheet_names = [ws.title for ws in spreadsheet.worksheets()]

                def parse(self, sheet_name):
                    worksheet = self._spreadsheet.worksheet(sheet_name)
                    data = worksheet.get_all_values()
                    if not data:
                        return pd.DataFrame()
                    # First row is headers
                    headers = data[0]
                    rows = data[1:]
                    return pd.DataFrame(rows, columns=headers)

            gs_wrapper = GoogleSheetsWrapper(spreadsheet)
            self.log(f"Available sheets: {gs_wrapper.sheet_names}")

            self._load_roster(gs_wrapper)
            self._build_name_map()  # Build fuzzy matching map
            self._load_regatta_signups(gs_wrapper)
            self._load_score_sheets(gs_wrapper)
            self._load_regatta_events(gs_wrapper)
            self._load_club_boats(gs_wrapper)

            self.log(f"Loaded {len(self.rowers)} rowers total from Google Sheets")
            return True

        except Exception as e:
            self.log(f"ERROR loading from Google Sheets: {e}")
            return False

    def _load_roster(self, xl: pd.ExcelFile):
        """Load roster from 'Roster' sheet"""
        if 'Roster' not in xl.sheet_names:
            self.log("Warning: No 'Roster' sheet found")
            return

        df = xl.parse('Roster')
        self.log(f"Roster columns: {list(df.columns)}")

        count = 0
        for _, row in df.iterrows():
            first = str(row.get('First Name', '')).strip()
            last = str(row.get('Last Name', '')).strip()

            if not first or first == 'nan':
                continue

            name = f"{first} {last}".strip()

            age = 27
            if 'Age' in row and pd.notna(row['Age']):
                try:
                    age = int(row['Age'])
                except (ValueError, TypeError):
                    pass

            gender = 'M'
            for col in ['GR', 'Gender', 'Sex']:
                if col in row and pd.notna(row[col]):
                    gender = str(row[col]).strip().upper()
                    if gender in ['MALE', 'M']:
                        gender = 'M'
                    elif gender in ['FEMALE', 'F']:
                        gender = 'F'
                    break

            side_port = parse_bool(row.get('P'), default=True)
            side_starboard = parse_bool(row.get('S'), default=True)
            side_cox = parse_bool(row.get('X'), default=False)

            pref_order = ""
            if 'Pref' in row.index and pd.notna(row['Pref']):
                pref_order = str(row['Pref']).strip()

            rower = Rower(
                name=name,
                age=age,
                gender=gender,
                side_port=side_port,
                side_starboard=side_starboard,
                side_coxswain=side_cox,
                pref_order=pref_order
            )
            self.rowers[name] = rower
            count += 1

        self.log(f"Loaded {count} rowers from Roster")

    def _load_regatta_signups(self, xl: pd.ExcelFile):
        """Load regatta signups from 'Regatta Sign Ups' sheet"""
        if 'Regatta Sign Ups' not in xl.sheet_names:
            self.log("Warning: No 'Regatta Sign Ups' sheet found")
            return

        df = xl.parse('Regatta Sign Ups')
        self.log(f"All columns in Regatta Sign Ups: {list(df.columns)}")

        name_col = 'Name'

        exclude_keywords = ['name', 'email', 'age', 'cat', 'category', 'notes',
                           'housing', 'feedback', 'request', 'saturday', 'sunday',
                           'friday', 'thursday', 'monday', 'tuesday', 'wednesday',
                           'unnamed']

        regatta_keywords = ['classic', 'regatta', 'regionals', 'head', 'sprints',
                           'rebellion', 'cascadia', 'rowfest', 'tail', 'bridge',
                           'opening day', 'lake', 'dog', 'charles']

        date_pattern = re.compile(r'\d{1,2}/\d{1,2}/\d{2,4}')
        location_pattern = re.compile(r',\s*[A-Z]{2}\b')
        day_pattern = re.compile(r'^(SUNDAY|SATURDAY|FRIDAY|THURSDAY|MONDAY|TUESDAY|WEDNESDAY)\n', re.IGNORECASE)

        regatta_candidates = {}
        for col in df.columns:
            col_str = str(col).lower()

            if any(kw in col_str for kw in exclude_keywords):
                if not day_pattern.match(str(col)):
                    continue
            if col_str.strip() == '' or col_str == 'nan':
                continue

            if any(word in col_str for word in regatta_keywords):
                has_day_prefix = bool(day_pattern.match(str(col)))
                has_date = bool(date_pattern.search(str(col)))
                has_location = bool(location_pattern.search(str(col)))

                base_name = str(col)
                base_name = re.sub(r'^(SUNDAY|SATURDAY|FRIDAY|THURSDAY|MONDAY|TUESDAY|WEDNESDAY)\n', '', base_name, flags=re.IGNORECASE)
                base_name = re.sub(r'\.\d+$', '', base_name)
                base_name = re.sub(r',?\s*\d{1,2}/\d{1,2}/\d{2,4}(-\d{1,2}/\d{1,2}/\d{2,4})?', '', base_name)
                base_name = re.sub(r'\n\d{1,2}/\d{1,2}/\d{2,4}(-\d{1,2}/\d{1,2}/\d{2,4})?', '', base_name)
                base_name = re.sub(r',\s*[A-Z]{2}\b', '', base_name)
                base_name = re.sub(r'\*$', '', base_name).strip()

                # Handle duplicate column names - df[col] might return DataFrame
                col_data = df[col]
                if isinstance(col_data, pd.DataFrame):
                    col_data = col_data.iloc[:, 0]  # Take first column if duplicates
                col_values = col_data.dropna().astype(str).str.lower().unique()
                has_yes = any('yes' in str(v) for v in col_values)

                # Priority: 1=day prefix (best), 2=plain name, 3=date only (info column)
                if has_day_prefix:
                    priority = 1
                elif has_date:
                    priority = 3  # Date-only columns are usually info, not signup
                elif has_location:
                    priority = 2
                else:
                    priority = 2  # Plain column name - likely the signup column

                if base_name not in regatta_candidates:
                    regatta_candidates[base_name] = []
                regatta_candidates[base_name].append((col, priority, has_yes))

        regatta_cols = []
        regatta_display_names = {}
        for base_name, candidates in regatta_candidates.items():
            # Sort by: has_yes (True first), then priority (lower is better)
            # This prefers columns with actual "yes" values, then by column type
            sorted_candidates = sorted(candidates, key=lambda x: (not x[2], x[1]))
            selected = sorted_candidates[0][0]

            regatta_cols.append(selected)
            regatta_display_names[selected] = base_name

        self.regattas = regatta_cols
        self.regatta_display_names = regatta_display_names
        self.log(f"Selected {len(regatta_cols)} regatta columns")
        for col in regatta_cols:
            display = regatta_display_names.get(col, col)
            self.log(f"  Regatta '{display}' uses column: '{col}'")

        signups_loaded = 0
        for _, row in df.iterrows():
            name = str(row.get(name_col, '')).strip()

            if not name or name == 'nan' or name in ['Dates', 'Registration Opens', 'Regatta Master']:
                continue

            if name not in self.rowers:
                continue

            rower = self.rowers[name]

            for regatta in regatta_cols:
                val = row.get(regatta, '')
                # Handle duplicate columns - val might be a Series
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                val_str = str(val).strip().lower() if pd.notna(val) else ''
                is_attending = bool(val_str) and val_str not in ['no', 'n', 'false', '0', '', 'nan']
                rower.regatta_signups[regatta] = is_attending
                if is_attending:
                    signups_loaded += 1

        self.log(f"Loaded {signups_loaded} regatta signups")

    def _load_score_sheets(self, xl: pd.ExcelFile):
        """Load scores from score sheets (1K, 5K, 100m, 250m, 2', etc.)"""
        score_sheets = []
        short_test_sheets = []

        for sheet in xl.sheet_names:
            # Check for standard K distances (1K, 5K, etc.)
            match = re.search(r'(\d+)\s*[Kk]', sheet)
            if match:
                distance = int(match.group(1)) * 1000
                score_sheets.append((sheet, distance))
                continue

            # Check for meter distances (100m, 250m, etc.)
            match = re.search(r'(\d+)\s*[Mm]', sheet)
            if match:
                distance = int(match.group(1))
                if distance < 1000:  # Only short distances
                    short_test_sheets.append((sheet, distance))
                continue

            # Check for time-based tests (2', 2 min, etc.)
            if re.search(r"2['\s]*min|2'", sheet, re.IGNORECASE):
                short_test_sheets.append((sheet, '2min'))

        self.log(f"Found score sheets: {[(s, d) for s, d in score_sheets]}")
        if short_test_sheets:
            self.log(f"Found short test sheets: {[(s, d) for s, d in short_test_sheets]}")

        for sheet_name, distance in score_sheets:
            self._load_scores_from_sheet(xl, sheet_name, distance)

        # Load short test sheets (100m, 250m, 2-minute pieces)
        for sheet_name, distance in short_test_sheets:
            self._load_short_test_scores(xl, sheet_name, distance)

        # Load misc maxes sheet (contains 100m, 250m, 2' test columns)
        if '2025 Misc Maxes' in xl.sheet_names:
            self._load_misc_maxes(xl, '2025 Misc Maxes')

    def _load_scores_from_sheet(self, xl: pd.ExcelFile, sheet_name: str, distance: int):
        """Load scores from a specific score sheet"""
        df = xl.parse(sheet_name)

        scores_loaded = 0
        fuzzy_matches = 0
        for _, row in df.iterrows():
            name = str(row.get('Name', '')).strip()
            if not name or name == 'nan':
                continue

            # Use fuzzy matching to find rower
            canonical_name = self.find_rower(name)
            if not canonical_name:
                continue

            if canonical_name != name:
                fuzzy_matches += 1

            rower = self.rowers[canonical_name]

            # Collect all valid times from this row
            times_to_try = []

            # Check for direct time in seconds columns
            for col in ['Time in Seconds', 'Time in Secs', 'Total Time']:
                if col in row.index and pd.notna(row[col]):
                    val = row[col]
                    # Handle both numeric and string values (Google Sheets returns strings)
                    try:
                        time_val = float(val) if not isinstance(val, (int, float)) else val
                        if time_val > 0:
                            times_to_try.append(time_val)
                    except (ValueError, TypeError):
                        pass

            # Check for individual time columns (1K Time #1, #2, #3)
            for col in row.index:
                col_str = str(col)
                if 'Time #' in col_str or 'Time#' in col_str:
                    if pd.notna(row[col]):
                        parsed = TimeParser.parse(row[col])
                        if parsed and parsed > 0:
                            times_to_try.append(parsed)

            # Check for named time columns
            for col in ['1K Time', '5K Time', 'Time']:
                if col in row.index and pd.notna(row[col]):
                    parsed = TimeParser.parse(row[col])
                    if parsed and parsed > 0:
                        times_to_try.append(parsed)

            # Use the fastest (minimum) time from all found times
            if not times_to_try:
                continue

            time_seconds = min(times_to_try)

            split_500m = None
            for col in ['Avg. Split', 'Avg Split', 'Split', '500m Split']:
                if col in row.index and pd.notna(row[col]):
                    split_500m = TimeParser.parse_split(row[col])
                    if split_500m and split_500m > 60:
                        break
                    else:
                        split_500m = None

            if time_seconds and not split_500m:
                split_500m = PhysicsEngine.calculate_500m_split(time_seconds, distance)
            elif split_500m and not time_seconds:
                time_seconds = (split_500m / 500) * distance

            if not time_seconds or not split_500m:
                continue

            if split_500m < 80 or split_500m > 180:
                continue

            watts = PhysicsEngine.split_to_watts(split_500m)

            score = Score(
                distance=distance,
                time_seconds=time_seconds,
                split_500m=split_500m,
                watts=watts,
                source_sheet=sheet_name
            )

            rower.add_score(score)
            scores_loaded += 1

        log_msg = f"Loaded {scores_loaded} scores from '{sheet_name}'"
        if fuzzy_matches > 0:
            log_msg += f" ({fuzzy_matches} fuzzy matched)"
        self.log(log_msg)

        # Log names in score sheet that don't match roster (even with fuzzy matching)
        unmatched_names = []
        for _, row in df.iterrows():
            name = str(row.get('Name', '')).strip()
            if name and name != 'nan' and not self.find_rower(name):
                if name not in unmatched_names:
                    unmatched_names.append(name)
        if unmatched_names:
            self.log(f"  Unmatched names in '{sheet_name}': {unmatched_names}")

    def _load_short_test_scores(self, xl: pd.ExcelFile, sheet_name: str, distance_or_type):
        """Load scores from short test sheets (100m, 250m, 2-minute pieces)"""
        df = xl.parse(sheet_name)
        self.log(f"Loading short tests from '{sheet_name}', columns: {list(df.columns)}")

        scores_loaded = 0
        fuzzy_matches = 0

        for _, row in df.iterrows():
            name = str(row.get('Name', '')).strip()
            if not name or name == 'nan':
                continue

            # Use fuzzy matching to find rower
            canonical_name = self.find_rower(name)
            if not canonical_name:
                continue

            if canonical_name != name:
                fuzzy_matches += 1

            rower = self.rowers[canonical_name]

            # For 2-minute tests, we need to get the distance from meters covered
            if distance_or_type == '2min':
                # Look for distance/meters column
                distance = None
                for col in ['Distance', 'Meters', 'Total Meters', 'Distance (m)']:
                    if col in row.index and pd.notna(row[col]):
                        try:
                            distance = int(float(str(row[col]).replace(',', '')))
                            break
                        except (ValueError, TypeError):
                            pass

                if not distance:
                    continue

                # For 2-minute test, time is always 120 seconds
                time_seconds = 120.0
            else:
                # Fixed distance test (100m, 250m)
                distance = distance_or_type

                # Look for time
                time_seconds = None
                for col in ['Time', 'Time in Seconds', 'Time (s)', 'Seconds']:
                    if col in row.index and pd.notna(row[col]):
                        parsed = TimeParser.parse(row[col])
                        if parsed and parsed > 0:
                            time_seconds = parsed
                            break

                if not time_seconds:
                    continue

            # Calculate split and watts
            split_500m = PhysicsEngine.calculate_500m_split(time_seconds, distance)

            # Validate split is reasonable (for short tests, can be faster)
            if split_500m < 60 or split_500m > 180:
                continue

            watts = PhysicsEngine.split_to_watts(split_500m)

            score = Score(
                distance=distance,
                time_seconds=time_seconds,
                split_500m=split_500m,
                watts=watts,
                source_sheet=sheet_name
            )

            rower.add_score(score)
            scores_loaded += 1

        log_msg = f"Loaded {scores_loaded} scores from '{sheet_name}'"
        if fuzzy_matches > 0:
            log_msg += f" ({fuzzy_matches} fuzzy matched)"
        self.log(log_msg)

    def _load_misc_maxes(self, xl: pd.ExcelFile, sheet_name: str):
        """Load scores from misc maxes sheet with split columns for 100m, 250m, 2' tests"""
        df = xl.parse(sheet_name)
        self.log(f"Loading misc maxes from '{sheet_name}', columns: {list(df.columns)}")

        # Define test patterns and their distances
        # Use patterns to match various apostrophe characters (', ', ′)
        test_patterns = [
            (r"100\s*m", 100),
            (r"250\s*m", 250),
            (r"2\s*['''′`]\s*(Test|min)?", '2min'),
        ]

        # Find matching columns
        test_columns = {}
        for col in df.columns:
            col_lower = str(col).lower()
            if 'split' not in col_lower:
                continue
            for pattern, distance in test_patterns:
                if re.search(pattern, col, re.IGNORECASE):
                    test_columns[col] = distance
                    self.log(f"  Matched column '{col}' as {distance} test")
                    break

        scores_loaded = 0
        fuzzy_matches = 0

        for _, row in df.iterrows():
            name = str(row.get('Name', '')).strip()
            if not name or name == 'nan':
                continue

            # Use fuzzy matching to find rower
            canonical_name = self.find_rower(name)
            if not canonical_name:
                continue

            if canonical_name != name:
                fuzzy_matches += 1

            rower = self.rowers[canonical_name]

            # Process each test column
            for col_name, distance_or_type in test_columns.items():
                if col_name not in df.columns:
                    continue

                split_val = row.get(col_name)
                if pd.isna(split_val) or str(split_val).strip() == '':
                    continue

                # Parse the split time
                split_500m = TimeParser.parse_split(split_val)
                if not split_500m or split_500m < 60 or split_500m > 200:
                    continue

                # Calculate distance and time based on test type
                if distance_or_type == '2min':
                    # For 2' test: time is 120 seconds, calculate distance from split
                    # pace = split / 500 (seconds per meter)
                    # distance = time / pace = 120 / (split / 500) = 60000 / split
                    time_seconds = 120.0
                    distance = int(60000 / split_500m)
                else:
                    # Fixed distance test (100m, 250m)
                    distance = distance_or_type
                    # time = distance * (split / 500)
                    time_seconds = distance * (split_500m / 500)

                watts = PhysicsEngine.split_to_watts(split_500m)

                score = Score(
                    distance=distance,
                    time_seconds=time_seconds,
                    split_500m=split_500m,
                    watts=watts,
                    source_sheet=sheet_name
                )

                rower.add_score(score)
                scores_loaded += 1

        log_msg = f"Loaded {scores_loaded} scores from '{sheet_name}'"
        if fuzzy_matches > 0:
            log_msg += f" ({fuzzy_matches} fuzzy matched)"
        self.log(log_msg)

    def _load_regatta_events(self, xl):
        """Load regatta events from 'Regatta_Events' sheet"""
        self.log(f"Available sheets: {xl.sheet_names}")
        if 'Regatta_Events' not in xl.sheet_names:
            self.log("No 'Regatta_Events' sheet found (optional)")
            return

        df = xl.parse('Regatta_Events')
        self.log(f"Loading regatta events from 'Regatta_Events', shape: {df.shape}, columns: {list(df.columns)}")

        # Debug: show first 10 rows
        for idx, row in df.head(10).iterrows():
            col_a = row.iloc[0] if len(row) > 0 else None
            col_b = row.iloc[1] if len(row) > 1 else None
            self.log(f"  Row {idx}: A='{col_a}' B='{col_b}'")

        current_regatta = None
        current_day = None
        events_loaded = 0

        for idx, row in df.iterrows():
            # Get values from first few columns
            col_a = row.iloc[0] if len(row) > 0 else None
            col_b = row.iloc[1] if len(row) > 1 else None
            col_c = row.iloc[2] if len(row) > 2 else None
            col_d = row.iloc[3] if len(row) > 3 else None
            col_e = row.iloc[4] if len(row) > 4 else None

            # Convert to strings for checking
            str_a = str(col_a).strip() if pd.notna(col_a) else ""
            str_b = str(col_b).strip() if pd.notna(col_b) else ""
            str_c = str(col_c).strip() if pd.notna(col_c) else ""

            # Skip empty rows and header row
            if not str_a or str_a.lower() == 'event number':
                continue

            # Check if this is a regatta name row (text in A, nothing meaningful in B/C)
            # Regatta names don't start with a number and don't have event times
            is_number = str_a.replace('.', '').isdigit()
            has_time_pattern = bool(re.search(r'\d{1,2}:\d{2}', str_b))
            # Day can be text like "Sunday, March 8, 2026" OR a date value like "2026-03-08 00:00:00"
            has_day_pattern = bool(re.search(r'(Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday)', str_a, re.IGNORECASE))
            # Also check for date format from Excel (YYYY-MM-DD or datetime)
            is_date_value = bool(re.search(r'^\d{4}-\d{2}-\d{2}', str_a))

            if has_day_pattern or is_date_value:
                # This is a day header row
                # Try to format date value nicely
                if is_date_value:
                    try:
                        from datetime import datetime
                        dt = pd.to_datetime(col_a)
                        current_day = dt.strftime("%A, %B %d, %Y")  # "Sunday, March 08, 2026"
                    except:
                        current_day = str_a
                else:
                    current_day = str_a
                self.log(f"  Found day: '{current_day}' for regatta '{current_regatta}'")
                # Initialize the events list for this regatta+day combo
                if current_regatta:
                    key = f"{current_regatta}|{current_day}"
                    if key not in self.regatta_events:
                        self.regatta_events[key] = []
            elif not is_number and not has_time_pattern and str_a and not str_b:
                # This is a regatta name row (text only in column A)
                current_regatta = str_a
                current_day = None
                self.log(f"  Found regatta: '{current_regatta}'")
            elif is_number and current_regatta and current_day:
                # This is an event row
                try:
                    event_number = int(float(str_a))
                except ValueError:
                    continue

                event_time = str_b
                event_name = str_c

                # Parse Include and Priority booleans
                include = parse_bool(col_d, default=False)
                priority = parse_bool(col_e, default=False)

                event = RegattaEvent(
                    regatta=current_regatta,
                    day=current_day,
                    event_number=event_number,
                    event_time=event_time,
                    event_name=event_name,
                    include=include,
                    priority=priority
                )

                key = f"{current_regatta}|{current_day}"
                if key not in self.regatta_events:
                    self.regatta_events[key] = []
                self.regatta_events[key].append(event)
                events_loaded += 1

        self.log(f"Loaded {events_loaded} regatta events from {len(self.regatta_events)} regatta/day combinations")
        if self.regatta_events:
            self.log(f"  Event keys: {list(self.regatta_events.keys())}")

    def _load_club_boats(self, xl):
        """Load club boats from 'WRC Boats' sheet (only club-owned boats)"""
        if 'WRC Boats' not in xl.sheet_names:
            self.log("No 'WRC Boats' sheet found (optional)")
            return

        df = xl.parse('WRC Boats')
        self.log(f"WRC Boats columns: {list(df.columns)}")

        self.club_boats = []

        # Expected columns: Name (A), Owner (B)
        name_col = 'Name' if 'Name' in df.columns else df.columns[0] if len(df.columns) > 0 else None
        owner_col = 'Owner' if 'Owner' in df.columns else df.columns[1] if len(df.columns) > 1 else None

        if name_col is None:
            self.log("No name column found in WRC Boats")
            return

        for _, row in df.iterrows():
            boat_name = str(row.get(name_col, '')).strip()
            owner = str(row.get(owner_col, '')).strip() if owner_col else ''

            # Skip empty rows, headers, and section dividers
            if not boat_name or boat_name == 'nan' or boat_name.lower() == 'name':
                continue
            if 'CLUB BOATS' in boat_name.upper() or 'PRIVATE BOATS' in boat_name.upper():
                continue

            # Only include club boats (Owner = "Willamette Rowing Club")
            if owner.lower() == 'willamette rowing club':
                self.club_boats.append(boat_name)

        # Add "Private Boat" option at the end for any private equipment
        self.club_boats.append("Private Boat")

        self.log(f"Loaded {len(self.club_boats)} club boats (including Private Boat option)")

    def get_rower(self, name: str) -> Optional[Rower]:
        return self.rowers.get(name)

    def get_attending_rowers(self, regatta_name: str) -> List[str]:
        """Get rowers attending a specific regatta (with scores)"""
        attending = []
        for name, rower in self.rowers.items():
            if rower.is_attending(regatta_name) and rower.scores:
                attending.append(name)
        return sorted(attending)

    def get_all_rowers_with_scores(self) -> List[str]:
        """Get all rowers with scores regardless of regatta"""
        return sorted([name for name, r in self.rowers.items() if r.scores])

    def get_all_rowers(self) -> List[str]:
        """Get all rowers regardless of scores"""
        return sorted(self.rowers.keys())


# =============================================================================
# ERG-TO-WATER CONVERSION (BioRow/Kleshnev)
# =============================================================================

# Boat factors for converting erg times to projected on-water times
# Reference: BioRow/Kleshnev prognostic speeds
BOAT_FACTORS = {
    "8+": 0.93, "4x": 0.96, "4+": 1.04, "4-": 1.00,
    "2x": 1.04, "2-": 1.08, "1x": 1.16
}

# Default tech efficiency for Masters/Club (5% slippage)
DEFAULT_TECH_EFFICIENCY = 1.05


def get_boat_factor(boat_class: str) -> float:
    """Get boat factor for erg-to-water conversion.

    Args:
        boat_class: Boat class string (e.g., "8+", "4x", "2-", "1x")

    Returns:
        Boat factor multiplier for erg-to-water conversion.
    """
    boat_class = boat_class.strip()
    return BOAT_FACTORS.get(boat_class, 1.0)


def apply_erg_to_water(erg_time_seconds: float, boat_class: str,
                       tech_efficiency: float = DEFAULT_TECH_EFFICIENCY) -> float:
    """Convert erg time to projected on-water time.

    Formula: Projected_Time = Erg_Time * Boat_Factor * Tech_Efficiency

    Args:
        erg_time_seconds: Raw erg time in seconds
        boat_class: Boat class string
        tech_efficiency: Technical efficiency multiplier (default 1.05 for Masters/Club)

    Returns:
        Projected on-water time in seconds
    """
    boat_factor = get_boat_factor(boat_class)
    return erg_time_seconds * boat_factor * tech_efficiency


# =============================================================================
# BOAT ANALYZER
# =============================================================================

class BoatAnalyzer:
    """Analyzes boat lineups"""

    K_FACTORS = {
        '1x': 0.02, '2x': 0.02, '2-': 0.02,
        '4x': 0.02, '4+': 0.02, '4-': 0.02,
        '8+': 0.02
    }

    def __init__(self, roster: RosterManager):
        self.roster = roster

    def analyze_lineup(self, rower_names: List[str], target_distance: int,
                       boat_class: str = '4+', calc_method: str = 'watts',
                       pace_predictor: str = 'power_law') -> Dict[str, Any]:
        """Analyze a lineup and return comprehensive results.

        calc_method: 'watts' (average watts, convert to split) or 'split' (average splits directly)
        pace_predictor: 'power_law' (fit personalized curve) or 'pauls_law' (traditional +5s/doubling)
        """

        if not rower_names:
            return {'error': 'No rowers in lineup'}

        is_sculling = 'x' in boat_class.lower()

        rowers = []
        projections = []
        port_watts = []
        starboard_watts = []
        all_watts = []
        all_splits = []  # For split averaging method

        # Calculate seat numbers in descending order (stroke = highest, bow = 1)
        num_rowing_seats = len(rower_names)

        for seat_idx, name in enumerate(rower_names):
            # Convert 0-indexed to rowing convention: stroke (idx 0) = highest seat number
            seat_number = num_rowing_seats - seat_idx

            rower = self.roster.get_rower(name)
            if not rower:
                projections.append({
                    'rower': name,
                    'seat': seat_number,
                    'error': 'Rower not found'
                })
                continue

            rowers.append(rower)

            # Check if rower has actual score at target distance
            actual_score = rower.scores.get(target_distance)
            closest_score = rower.get_closest_score(target_distance)

            if not closest_score:
                projections.append({
                    'rower': name,
                    'seat': seat_number,
                    'age': rower.age,
                    'side': rower.side_preference_str(),
                    'error': 'No scores available'
                })
                continue

            # Use actual score if available at exact target distance
            power_law_points = None
            if actual_score:
                projected_split = actual_score.split_500m
                projected_watts = actual_score.watts
                projection_method = 'actual'
                source_score = actual_score
            else:
                # Project split using selected method
                source_score = closest_score
                projection_method = 'pauls_law'  # Track which method was actually used
                if pace_predictor == 'power_law':
                    # Try Power Law first - requires at least 2 data points
                    power_law_split = rower.project_split_power_law(target_distance)
                    if power_law_split and power_law_split > 0:
                        projected_split = power_law_split
                        projection_method = 'power_law'
                        # Get the two points used for display
                        power_law_points = rower.get_two_closest_data_points(target_distance)
                    else:
                        # Fall back to Paul's Law
                        projected_split = PhysicsEngine.pauls_law_projection(
                            closest_score.split_500m, closest_score.distance, target_distance
                        )
                else:
                    # Use Paul's Law
                    projected_split = PhysicsEngine.pauls_law_projection(
                        closest_score.split_500m, closest_score.distance, target_distance
                    )

                projected_watts = PhysicsEngine.split_to_watts(projected_split)

            seat_side = None
            if not is_sculling:
                seat_side = 'S' if seat_idx % 2 == 0 else 'P'
                if seat_side == 'P':
                    port_watts.append(projected_watts)
                else:
                    starboard_watts.append(projected_watts)

            all_watts.append(projected_watts)
            all_splits.append(projected_split)

            projections.append({
                'rower': name,
                'seat': seat_number,
                'seat_side': seat_side,
                'age': rower.age,
                'side': rower.side_preference_str(),
                'source_distance': source_score.distance,
                'source_split': source_score.split_500m,
                'source_sheet': source_score.source_sheet,
                'projected_split': projected_split,
                'projected_watts': projected_watts,
                'projection_method': projection_method,
                'num_data_points': len(rower.scores),
                'power_law_points': power_law_points
            })

        if not all_watts:
            return {
                'error': 'No valid projections',
                'projections': projections
            }

        # Calculate both methods
        avg_watts = statistics.mean(all_watts)
        boat_split_watts = PhysicsEngine.watts_to_split(avg_watts)

        avg_split = statistics.mean(all_splits)
        boat_split_split = avg_split

        # Use the selected method for primary results
        if calc_method == 'watts':
            boat_split = boat_split_watts
        else:
            boat_split = boat_split_split

        raw_time = (boat_split / 500) * target_distance

        side_balance_pct = None
        if port_watts and starboard_watts:
            total_port = sum(port_watts)
            total_starboard = sum(starboard_watts)
            total_power = total_port + total_starboard
            side_balance_pct = abs(total_port - total_starboard) / total_power * 100

        avg_age = statistics.mean([r.age for r in rowers]) if rowers else 27
        k_factor = self.K_FACTORS.get(boat_class, 0.02)
        distance_multiplier = target_distance / 1000
        handicap_seconds = ((avg_age - 27) ** 2) * k_factor * distance_multiplier
        adjusted_time = raw_time - handicap_seconds

        return {
            'rower_count': len(rowers),
            'projections': projections,
            'target_distance': target_distance,
            'boat_class': boat_class,
            'calc_method': calc_method,
            'avg_watts': avg_watts,
            'boat_split_500m': boat_split,
            'boat_split_watts_method': boat_split_watts,
            'boat_split_split_method': boat_split_split,
            'raw_time': raw_time,
            'avg_age': avg_age,
            'handicap_seconds': handicap_seconds,
            'adjusted_time': adjusted_time,
            'side_balance_pct': side_balance_pct,
            'port_watts_total': sum(port_watts) if port_watts else None,
            'starboard_watts_total': sum(starboard_watts) if starboard_watts else None
        }


# =============================================================================
# LINEUP OPTIMIZER
# =============================================================================

class LineupOptimizer:
    """Finds optimal lineup combinations based on erg scores and constraints."""

    # Maximum combinations before switching from exhaustive to heuristic search
    # C(20,8) = 125,970 combinations takes ~1-2 seconds, which is acceptable
    MAX_EXHAUSTIVE_COMBINATIONS = 150000

    def __init__(self, roster: RosterManager, analyzer: BoatAnalyzer):
        self.roster = roster
        self.analyzer = analyzer

    def get_eligible_rowers(self, regatta: str, gender: Optional[str] = None,
                            min_age: int = 0, excluded_names: Optional[Set[str]] = None) -> List[Rower]:
        """Get rowers eligible based on regatta attendance, gender, and age constraints.

        Args:
            regatta: Regatta key (or "__all__" for all rowers)
            gender: 'M', 'W', or None for no filter
            min_age: Minimum individual age (not average), default 0
            excluded_names: Set of rower names to exclude from results

        Returns:
            List of eligible Rower objects with erg scores
        """
        # Get rower list based on regatta
        if regatta == "__all__":
            rower_names = self.roster.get_all_rowers()
        else:
            # Handle regatta|day format
            regatta_for_filter = regatta.split("|")[0] if "|" in regatta else regatta
            regatta_lower = regatta_for_filter.lower()
            rower_names = []
            for name, rower in self.roster.rowers.items():
                # Direct match or partial match
                if rower.is_attending(regatta_for_filter):
                    rower_names.append(name)
                else:
                    for signup_regatta, attending in rower.regatta_signups.items():
                        if attending and (regatta_lower in signup_regatta.lower() or
                                         signup_regatta.lower() in regatta_lower):
                            rower_names.append(name)
                            break

        eligible = []
        for name in rower_names:
            rower = self.roster.get_rower(name)
            if not rower:
                continue
            # Must have at least one erg score
            if not rower.scores:
                continue
            # Gender filter
            if gender == 'M' and rower.gender != 'M':
                continue
            if gender == 'W' and rower.gender != 'F':
                continue
            # Age filter (individual must meet minimum)
            if rower.age < min_age:
                continue
            # Exclusion filter
            if excluded_names and name in excluded_names:
                continue
            eligible.append(rower)

        return eligible

    def project_rower_watts(self, rower: Rower, target_distance: int,
                            predictor: str = 'power_law') -> Optional[float]:
        """Get projected watts for a rower at target distance.

        Args:
            rower: Rower object
            target_distance: Target distance in meters
            predictor: 'power_law' or 'pauls_law'

        Returns:
            Projected watts or None if no score available
        """
        # Check for exact distance score first
        score = rower.scores.get(target_distance)
        if score:
            return score.watts

        closest_score = rower.get_closest_score(target_distance)
        if not closest_score:
            return None

        if predictor == 'power_law':
            # Try Power Law first
            projected_split = rower.project_split_power_law(target_distance)
            if projected_split and projected_split > 0:
                return PhysicsEngine.split_to_watts(projected_split)

        # Fall back to Paul's Law
        projected_split = PhysicsEngine.pauls_law_projection(
            closest_score.split_500m, closest_score.distance, target_distance
        )
        return PhysicsEngine.split_to_watts(projected_split)

    def _can_row_seat(self, rower: Rower, seat_idx: int, boat_class: str) -> bool:
        """Check if rower can row the given seat based on side preference.

        For sweep boats:
        - Even seat indices (0, 2, 4, 6) = Starboard
        - Odd seat indices (1, 3, 5, 7) = Port

        For sculling boats, always returns True.
        """
        is_sculling = 'x' in boat_class.lower()
        if is_sculling:
            return True

        # Sweep boat - check side preference
        if seat_idx % 2 == 0:  # Starboard seat
            return rower.side_starboard
        else:  # Port seat
            return rower.side_port

    def _is_valid_seat_assignment(self, rowers: List[Rower], boat_class: str) -> bool:
        """Check if all rowers can row their assigned seats."""
        for idx, rower in enumerate(rowers):
            if not self._can_row_seat(rower, idx, boat_class):
                return False
        return True

    def _optimize_seat_assignment(self, rowers: List[Rower], boat_class: str) -> Optional[List[Rower]]:
        """Find optimal seat assignment for sweep boats based on side preferences.

        Returns None if no valid assignment exists.
        """
        is_sculling = 'x' in boat_class.lower()
        if is_sculling:
            return rowers  # No seat optimization needed

        from itertools import permutations

        num_seats = len(rowers)

        # For small crews, try all permutations
        if num_seats <= 4:
            for perm in permutations(rowers):
                perm_list = list(perm)
                if self._is_valid_seat_assignment(perm_list, boat_class):
                    return perm_list
            return None

        # For 8+, use greedy assignment to avoid factorial explosion
        # Separate rowers by side preference
        port_only = [r for r in rowers if r.side_port and not r.side_starboard]
        starboard_only = [r for r in rowers if r.side_starboard and not r.side_port]
        both = [r for r in rowers if r.side_port and r.side_starboard]

        # Need 4 port seats (odd indices: 1,3,5,7) and 4 starboard seats (even indices: 0,2,4,6)
        port_needed = num_seats // 2
        starboard_needed = num_seats // 2

        if len(port_only) > port_needed or len(starboard_only) > starboard_needed:
            return None  # Too many single-side rowers

        # Assign single-side rowers first
        port_rowers = port_only.copy()
        starboard_rowers = starboard_only.copy()

        # Fill remaining slots with "both" rowers
        remaining = both.copy()
        while len(port_rowers) < port_needed and remaining:
            port_rowers.append(remaining.pop(0))
        while len(starboard_rowers) < starboard_needed and remaining:
            starboard_rowers.append(remaining.pop(0))

        if len(port_rowers) != port_needed or len(starboard_rowers) != starboard_needed:
            return None

        # Interleave: stroke(0)=S, 7(1)=P, 6(2)=S, 5(3)=P, ...
        result = []
        for i in range(num_seats):
            if i % 2 == 0:  # Starboard
                result.append(starboard_rowers.pop(0) if starboard_rowers else None)
            else:  # Port
                result.append(port_rowers.pop(0) if port_rowers else None)

        if None in result:
            return None
        return result

    def _optimize_seat_assignment_with_locks(self, rowers: List[Rower], boat_class: str,
                                             locked_rowers: Dict[int, str]) -> Optional[List[Rower]]:
        """Find valid seat assignment for sweep boats while preserving locked positions.

        Locked rowers stay in their assigned seats, and unlocked rowers are assigned
        to remaining seats based on their side preferences.

        Returns None if no valid assignment exists.
        """
        is_sculling = 'x' in boat_class.lower()
        if is_sculling:
            return rowers  # No seat optimization needed for sculling

        num_seats = len(rowers)
        locked_positions = set(locked_rowers.keys())

        # Identify which rowers are locked (by name) and which are unlocked
        locked_names = set(locked_rowers.values())
        locked_rower_list = [r for r in rowers if r.name in locked_names]
        unlocked_rower_list = [r for r in rowers if r.name not in locked_names]

        # Verify locked rowers can row their locked seats
        for seat_idx, name in locked_rowers.items():
            rower = next((r for r in rowers if r.name == name), None)
            if rower and not self._can_row_seat(rower, seat_idx, boat_class):
                return None  # Locked rower can't row their locked seat

        # Determine which unlocked seats need port vs starboard rowers
        unlocked_port_seats = [i for i in range(num_seats) if i not in locked_positions and i % 2 == 1]
        unlocked_starboard_seats = [i for i in range(num_seats) if i not in locked_positions and i % 2 == 0]

        # Separate unlocked rowers by side preference
        port_only = [r for r in unlocked_rower_list if r.side_port and not r.side_starboard]
        starboard_only = [r for r in unlocked_rower_list if r.side_starboard and not r.side_port]
        both = [r for r in unlocked_rower_list if r.side_port and r.side_starboard]

        # Check if we have too many single-side rowers
        if len(port_only) > len(unlocked_port_seats) or len(starboard_only) > len(unlocked_starboard_seats):
            return None

        # Assign single-side rowers first
        port_assignments = port_only.copy()
        starboard_assignments = starboard_only.copy()

        # Fill remaining with "both" rowers
        remaining = both.copy()
        while len(port_assignments) < len(unlocked_port_seats) and remaining:
            port_assignments.append(remaining.pop(0))
        while len(starboard_assignments) < len(unlocked_starboard_seats) and remaining:
            starboard_assignments.append(remaining.pop(0))

        if len(port_assignments) != len(unlocked_port_seats) or len(starboard_assignments) != len(unlocked_starboard_seats):
            return None

        # Build the final result with locked rowers in place
        result = [None] * num_seats

        # Place locked rowers
        for seat_idx, name in locked_rowers.items():
            rower = next((r for r in rowers if r.name == name), None)
            result[seat_idx] = rower

        # Place unlocked rowers
        for seat_idx in unlocked_starboard_seats:
            if starboard_assignments:
                result[seat_idx] = starboard_assignments.pop(0)
        for seat_idx in unlocked_port_seats:
            if port_assignments:
                result[seat_idx] = port_assignments.pop(0)

        if None in result:
            return None
        return result

    def find_optimal_lineups(self, regatta: str, num_seats: int, boat_class: str,
                             target_distance: int, calc_method: str = 'watts',
                             predictor: str = 'power_law', optimize_for: str = 'raw',
                             gender: str = "Men's", min_avg_age: int = 0,
                             num_results: int = 3, excluded_rowers: Optional[Set[str]] = None,
                             locked_rowers: Optional[Dict[int, str]] = None) -> List[Dict]:
        """Find optimal lineup combinations.

        Args:
            regatta: Regatta key for filtering rowers
            num_seats: Number of rowing seats needed
            boat_class: Boat class (e.g., '4+', '8+', '2x')
            target_distance: Target erg distance
            calc_method: 'watts' or 'split' averaging
            predictor: 'power_law' or 'pauls_law'
            optimize_for: 'raw' for fastest raw time, 'adjusted' for handicap-adjusted
            gender: "Men's", "Women's", or "Mixed"
            min_avg_age: Minimum average age for the lineup
            num_results: Number of top lineups to return
            excluded_rowers: Set of rower names to exclude from consideration
            locked_rowers: Dict mapping seat index to rower name (seats to preserve)

        Returns:
            List of dicts with 'rowers', 'raw_time', 'adjusted_time', 'avg_age'
        """
        from itertools import combinations
        import math as math_module

        # Handle locked rowers: exclude them from available pool, adjust seats needed
        locked_rowers = locked_rowers or {}
        locked_names = set(locked_rowers.values())
        unlocked_seats = num_seats - len(locked_rowers)

        # If all seats are locked, just evaluate the locked lineup
        if unlocked_seats == 0:
            locked_rower_names = [locked_rowers.get(i) for i in range(num_seats)]
            if None in locked_rower_names:
                return []  # Invalid locked configuration
            # Build analysis for locked lineup
            rowers = [self.roster.get_rower(name) for name in locked_rower_names]
            if None in rowers:
                return []
            analysis = self.analyzer.analyze_lineup(
                locked_rower_names, target_distance, boat_class, calc_method, predictor
            )
            if not analysis or analysis['raw_time'] is None:
                return []
            avg_age = statistics.mean([r.age for r in rowers])
            return [{
                'rowers': locked_rower_names,
                'raw_time': analysis['raw_time'],
                'adjusted_time': analysis['adjusted_time'],
                'avg_age': avg_age
            }]

        # Map UI gender to filter gender
        gender_filter = None
        if gender == "Men's":
            gender_filter = 'M'
        elif gender == "Women's":
            gender_filter = 'W'
        # Mixed = None (no filter, but we'll verify mix in lineup)

        # Combine exclusions: user-excluded rowers + locked rowers (already assigned)
        all_excluded = set(excluded_rowers or set()) | locked_names

        # Get eligible rowers
        eligible = self.get_eligible_rowers(regatta, gender_filter, min_age=0, excluded_names=all_excluded)

        if len(eligible) < unlocked_seats:
            return []

        # Pre-compute projected watts for all eligible rowers
        rower_watts = {}
        for rower in eligible:
            watts = self.project_rower_watts(rower, target_distance, predictor)
            if watts and watts > 0:
                rower_watts[rower.name] = watts

        # Filter to only rowers with valid projections
        eligible = [r for r in eligible if r.name in rower_watts]

        if len(eligible) < unlocked_seats:
            return []

        # Calculate number of combinations (only for unlocked seats)
        n = len(eligible)
        k = unlocked_seats
        num_combinations = math_module.comb(n, k)

        results = []

        # For adjusted optimization, compute handicap-adjusted score for each rower
        # This helps the heuristic search consider age benefits
        rower_adjusted_score = {}
        if optimize_for == 'adjusted':
            # Handicap formula: adjusted_time = raw_time - ((avg_age - 27)^2) * k * distance_mult
            # OLDER crews get MORE seconds subtracted, so LOWER adjusted time (faster)
            # For individual ranking, we want rowers who contribute to lower adjusted time:
            #   - Higher watts (lower raw time contribution)
            #   - Higher age (more handicap benefit when averaged into lineup)
            k_factor = 0.02
            distance_multiplier = target_distance / 1000
            for rower in eligible:
                watts = rower_watts.get(rower.name, 0)
                # Calculate this rower's handicap contribution (seconds that would be subtracted)
                handicap_benefit = ((rower.age - 27) ** 2) * k_factor * distance_multiplier
                # Convert handicap seconds to approximate watts equivalent
                # ~4 watts ≈ 1 second at typical splits (rough approximation)
                watts_equivalent_bonus = handicap_benefit * 4
                # Higher score = better for adjusted time (fast + old)
                rower_adjusted_score[rower.name] = watts + watts_equivalent_bonus
        elif optimize_for == 'category':
            # For category optimization: find fastest raw time that meets min_avg_age
            # Favor fast rowers, but also consider age to help meet the category minimum
            # We want rowers who are: fast AND old enough to help meet the threshold
            for rower in eligible:
                watts = rower_watts.get(rower.name, 0)
                # Small bonus for being above min_avg_age (helps ensure we meet threshold)
                # But primary factor is still speed (watts)
                age_bonus = 0
                if min_avg_age > 0 and rower.age >= min_avg_age:
                    # Rower meets category on their own - small bonus
                    age_bonus = 10  # Small watts-equivalent bonus
                rower_adjusted_score[rower.name] = watts + age_bonus
        else:
            rower_adjusted_score = rower_watts.copy()

        # Debug counters
        debug_counts = {'age_rejected': 0, 'gender_rejected': 0, 'seat_rejected': 0, 'total_tried': 0}

        # Calculate gender needs for Mixed lineups with locked rowers
        locked_men = sum(1 for name in locked_names if self.roster.get_rower(name) and self.roster.get_rower(name).gender == 'M')
        locked_women = sum(1 for name in locked_names if self.roster.get_rower(name) and self.roster.get_rower(name).gender == 'F')
        need_men = (num_seats // 2) - locked_men if gender == "Mixed" else None
        need_women = (num_seats // 2) - locked_women if gender == "Mixed" else None

        if num_combinations <= self.MAX_EXHAUSTIVE_COMBINATIONS:
            # Exhaustive search - evaluate ALL combinations
            for combo in combinations(eligible, unlocked_seats):
                debug_counts['total_tried'] += 1
                result = self._evaluate_lineup(
                    list(combo), boat_class, target_distance, calc_method, predictor,
                    gender, min_avg_age, rower_watts, debug_counts, locked_rowers, num_seats
                )
                if result:
                    results.append(result)
        else:
            # Heuristic search: greedy with diversity
            seen_combos = set()

            # For adjusted optimization, first try age-optimized combinations
            if optimize_for == 'adjusted':
                # Sort by age descending (older = more handicap benefit)
                age_sorted = sorted(eligible, key=lambda r: r.age, reverse=True)
                # Try top N oldest rowers as a baseline
                if len(age_sorted) >= unlocked_seats:
                    if gender == "Mixed" and need_men is not None and need_women is not None:
                        # For Mixed with locks, get oldest from each gender based on what's still needed
                        males_by_age = sorted([r for r in eligible if r.gender == 'M'], key=lambda r: r.age, reverse=True)
                        females_by_age = sorted([r for r in eligible if r.gender == 'F'], key=lambda r: r.age, reverse=True)
                        if len(males_by_age) >= need_men and len(females_by_age) >= need_women:
                            oldest_combo = males_by_age[:need_men] + females_by_age[:need_women]
                        else:
                            oldest_combo = []
                    else:
                        oldest_combo = age_sorted[:unlocked_seats]

                    if len(oldest_combo) == unlocked_seats:
                        combo_key = tuple(sorted(r.name for r in oldest_combo))
                        seen_combos.add(combo_key)
                        debug_counts['total_tried'] += 1
                        result = self._evaluate_lineup(
                            oldest_combo, boat_class, target_distance, calc_method, predictor,
                            gender, min_avg_age, rower_watts, debug_counts, locked_rowers, num_seats
                        )
                        if result:
                            results.append(result)

            # Sort by appropriate score (raw watts or adjusted score)
            sort_score = rower_adjusted_score if optimize_for == 'adjusted' else rower_watts
            sorted_rowers = sorted(eligible, key=lambda r: sort_score.get(r.name, 0), reverse=True)

            # For Mixed gender, separate by gender for 50/50 split combo building
            if gender == "Mixed" and need_men is not None and need_women is not None:
                males_sorted = [r for r in sorted_rowers if r.gender == 'M']
                females_sorted = [r for r in sorted_rowers if r.gender == 'F']
                # Need enough of each gender to fill unlocked seats
                if len(males_sorted) < need_men or len(females_sorted) < need_women:
                    return []  # Can't form a 50/50 mixed lineup

            # Also try the top-scored rowers as another baseline
            if len(sorted_rowers) >= unlocked_seats:
                if gender == "Mixed" and need_men is not None and need_women is not None:
                    # For Mixed, build gender split based on what's still needed
                    top_combo = males_sorted[:need_men] + females_sorted[:need_women]
                else:
                    top_combo = sorted_rowers[:unlocked_seats]

                if len(top_combo) == unlocked_seats:
                    combo_key = tuple(sorted(r.name for r in top_combo))
                    if combo_key not in seen_combos:
                        seen_combos.add(combo_key)
                        debug_counts['total_tried'] += 1
                        result = self._evaluate_lineup(
                            top_combo, boat_class, target_distance, calc_method, predictor,
                            gender, min_avg_age, rower_watts, debug_counts, locked_rowers, num_seats
                        )
                        if result:
                            results.append(result)

            # For category optimization, try strategic age-balanced combinations
            if optimize_for == 'category' and min_avg_age > 0:
                if gender == "Mixed" and need_men is not None and need_women is not None:
                    # For Mixed category, do age/speed balancing within each gender
                    males_by_age = sorted(males_sorted, key=lambda r: r.age, reverse=True)
                    females_by_age = sorted(females_sorted, key=lambda r: r.age, reverse=True)

                    # Try mixing: N fastest + (need-N) oldest, for each gender
                    for num_fast_m in range(need_men + 1):
                        num_old_m = need_men - num_fast_m
                        fast_males = males_sorted[:num_fast_m]
                        old_males = [r for r in males_by_age if r not in fast_males][:num_old_m]

                        for num_fast_f in range(need_women + 1):
                            num_old_f = need_women - num_fast_f
                            fast_females = females_sorted[:num_fast_f]
                            old_females = [r for r in females_by_age if r not in fast_females][:num_old_f]

                            combo = fast_males + old_males + fast_females + old_females
                            if len(combo) == unlocked_seats:
                                combo_key = tuple(sorted(r.name for r in combo))
                                if combo_key not in seen_combos:
                                    seen_combos.add(combo_key)
                                    debug_counts['total_tried'] += 1
                                    result = self._evaluate_lineup(
                                        combo, boat_class, target_distance, calc_method, predictor,
                                        gender, min_avg_age, rower_watts, debug_counts, locked_rowers, num_seats
                                    )
                                    if result:
                                        results.append(result)
                else:
                    # Sort by age and watts separately
                    by_age = sorted(eligible, key=lambda r: r.age, reverse=True)
                    by_watts = sorted(eligible, key=lambda r: rower_watts.get(r.name, 0), reverse=True)

                    # Try mixing: N fastest + (seats-N) oldest, for various N
                    for num_fast in range(unlocked_seats + 1):
                        num_old = unlocked_seats - num_fast
                        fast_rowers = by_watts[:num_fast]
                        # Get oldest rowers not already in fast list
                        old_rowers = [r for r in by_age if r not in fast_rowers][:num_old]

                        if len(fast_rowers) + len(old_rowers) == unlocked_seats:
                            combo = fast_rowers + old_rowers
                            combo_key = tuple(sorted(r.name for r in combo))
                            if combo_key not in seen_combos:
                                seen_combos.add(combo_key)
                                debug_counts['total_tried'] += 1
                                result = self._evaluate_lineup(
                                    combo, boat_class, target_distance, calc_method, predictor,
                                    gender, min_avg_age, rower_watts, debug_counts, locked_rowers, num_seats
                                )
                                if result:
                                    results.append(result)

            # Generate diverse lineups by varying top selections
            attempts = 0
            max_attempts = self.MAX_EXHAUSTIVE_COMBINATIONS
            # Ensure we explore at least 100 valid candidates for better diversity
            min_candidates = max(100, num_results * 20)

            while len(results) < min_candidates and attempts < max_attempts:
                # Greedy selection with some randomization
                import random
                selected = []

                if gender == "Mixed" and need_men is not None and need_women is not None:
                    # For Mixed, select from each gender pool based on what's needed
                    male_pool = males_sorted.copy()
                    female_pool = females_sorted.copy()

                    # Select needed men
                    for _ in range(need_men):
                        if not male_pool:
                            break
                        top_n = 5 if optimize_for == 'adjusted' else 3
                        candidates = male_pool[:min(top_n, len(male_pool))]
                        weights = list(range(top_n, 0, -1))[:len(candidates)]
                        chosen = random.choices(candidates, weights=weights)[0]
                        selected.append(chosen)
                        male_pool.remove(chosen)

                    # Select needed women
                    for _ in range(need_women):
                        if not female_pool:
                            break
                        top_n = 5 if optimize_for == 'adjusted' else 3
                        candidates = female_pool[:min(top_n, len(female_pool))]
                        weights = list(range(top_n, 0, -1))[:len(candidates)]
                        chosen = random.choices(candidates, weights=weights)[0]
                        selected.append(chosen)
                        female_pool.remove(chosen)
                else:
                    pool = sorted_rowers.copy()
                    for _ in range(unlocked_seats):
                        if not pool:
                            break
                        # Take from top candidates with weighted random (more diversity for adjusted)
                        top_n = 5 if optimize_for == 'adjusted' else 3
                        candidates = pool[:min(top_n, len(pool))]
                        weights = list(range(top_n, 0, -1))[:len(candidates)]
                        chosen = random.choices(candidates, weights=weights)[0]
                        selected.append(chosen)
                        pool.remove(chosen)

                if len(selected) == unlocked_seats:
                    combo_key = tuple(sorted(r.name for r in selected))
                    if combo_key not in seen_combos:
                        seen_combos.add(combo_key)
                        debug_counts['total_tried'] += 1
                        result = self._evaluate_lineup(
                            selected, boat_class, target_distance, calc_method, predictor,
                            gender, min_avg_age, rower_watts, debug_counts, locked_rowers, num_seats
                        )
                        if result:
                            results.append(result)

                attempts += 1

        # Sort by optimization target
        if optimize_for == 'adjusted':
            results.sort(key=lambda x: x['adjusted_time'])
        else:
            # Both 'raw' and 'category' sort by raw_time
            results.sort(key=lambda x: x['raw_time'])


        # Debug: log top results for category mode
        if optimize_for == 'category' and results:
            top_3 = results[:3]
            debug_info = [f"{r['raw_time']:.1f}s (age {r['avg_age']:.1f})" for r in top_3]
            st.toast(f"Top {len(top_3)} by raw time: {', '.join(debug_info)}")

        return results[:num_results]

    def _evaluate_lineup(self, rowers: List[Rower], boat_class: str, target_distance: int,
                         calc_method: str, predictor: str, gender: str, min_avg_age: int,
                         rower_watts: Dict[str, float], debug_counts: Dict = None,
                         locked_rowers: Optional[Dict[int, str]] = None, total_seats: Optional[int] = None) -> Optional[Dict]:
        """Evaluate a lineup combination and return results if valid.

        Args:
            rowers: List of unlocked Rower objects to evaluate
            locked_rowers: Dict mapping seat index to rower name (locked seats)
            total_seats: Total number of seats in the boat (for merging locked rowers)
        """
        # Merge locked rowers with unlocked rowers to form full lineup
        if locked_rowers and total_seats:
            # Get locked rower objects
            locked_rower_objects = {}
            for seat_idx, name in locked_rowers.items():
                locked_rower_obj = self.roster.get_rower(name)
                if locked_rower_obj:
                    locked_rower_objects[seat_idx] = locked_rower_obj

            # Build full lineup: place locked rowers at their seats, fill rest with unlocked
            full_rowers = [None] * total_seats
            unlocked_idx = 0
            for i in range(total_seats):
                if i in locked_rower_objects:
                    full_rowers[i] = locked_rower_objects[i]
                elif unlocked_idx < len(rowers):
                    full_rowers[i] = rowers[unlocked_idx]
                    unlocked_idx += 1

            if None in full_rowers:
                return None  # Couldn't fill all seats
            rowers = full_rowers

        # Check average age constraint
        avg_age = statistics.mean([r.age for r in rowers])
        if avg_age < min_avg_age:
            if debug_counts is not None:
                debug_counts['age_rejected'] = debug_counts.get('age_rejected', 0) + 1
            return None

        # Check mixed gender requirement
        if gender == "Mixed":
            genders = set(r.gender for r in rowers)
            if 'M' not in genders or 'F' not in genders:
                if debug_counts is not None:
                    debug_counts['gender_rejected'] = debug_counts.get('gender_rejected', 0) + 1
                return None

        # For sweep boats, find valid seat assignment
        # When we have locked rowers, we need a different approach to preserve their positions
        is_sculling = 'x' in boat_class.lower()
        if not is_sculling:
            if locked_rowers:
                # With locked seats, use specialized assignment that preserves locked positions
                assigned = self._optimize_seat_assignment_with_locks(rowers, boat_class, locked_rowers)
            else:
                assigned = self._optimize_seat_assignment(rowers, boat_class)
            if assigned is None:
                if debug_counts is not None:
                    debug_counts['seat_rejected'] = debug_counts.get('seat_rejected', 0) + 1
                return None
            rowers = assigned

        # Calculate lineup time using BoatAnalyzer
        rower_names = [r.name for r in rowers]
        analysis = self.analyzer.analyze_lineup(
            rower_names, target_distance, boat_class, calc_method, predictor
        )

        if 'error' in analysis and 'raw_time' not in analysis:
            return None

        raw_time = analysis.get('raw_time', float('inf'))
        adjusted_time = analysis.get('adjusted_time', float('inf'))

        if raw_time == float('inf'):
            return None

        return {
            'rowers': rower_names,
            'rower_objects': rowers,
            'raw_time': raw_time,
            'adjusted_time': adjusted_time,
            'avg_age': avg_age,
            'analysis': analysis
        }


# =============================================================================
# UTILITIES
# =============================================================================

def format_time(seconds: float) -> str:
    """Format seconds as MM:SS.d"""
    if seconds is None or seconds <= 0:
        return "N/A"
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}:{secs:04.1f}"


def format_split(seconds: float) -> str:
    """Format 500m split as M:SS.d"""
    if seconds is None or seconds <= 0:
        return "N/A"
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}:{secs:04.1f}"


def get_last_name_abbrev(full_name: str, length: int = 4) -> str:
    """Extract and abbreviate last name from full name.

    Args:
        full_name: Full name like "John Smith" or "Smith, John"
        length: Number of characters to abbreviate to (default 4)

    Returns:
        Abbreviated last name in uppercase, e.g., "SMIT"
    """
    if not full_name:
        return "----"

    # Handle "Last, First" format
    if ',' in full_name:
        last_name = full_name.split(',')[0].strip()
    else:
        # Handle "First Last" format - take the last word
        parts = full_name.strip().split()
        last_name = parts[-1] if parts else full_name

    return last_name[:length].upper()


def get_first_name(full_name: str) -> str:
    """Extract first name from full name.

    Args:
        full_name: Full name like "John Smith" or "Smith, John"

    Returns:
        First name, e.g., "John"
    """
    if not full_name:
        return "-"

    # Handle "Last, First" format
    if ',' in full_name:
        parts = full_name.split(',')
        if len(parts) > 1:
            return parts[1].strip().split()[0]  # Get first word after comma
        return full_name

    # Handle "First Last" format - take the first word
    parts = full_name.strip().split()
    return parts[0] if parts else full_name


def format_lineup_display(lineup_id: str, rower_names: List[str], boat_class: str) -> str:
    """Format lineup display name with rower first names.

    Args:
        lineup_id: Original lineup ID ("A", "B", "C")
        rower_names: List of rower names in seat order (stroke first, bow last)
        boat_class: Boat class like "2x", "4+", "8+"

    Returns:
        Formatted string like "A - John/Jane" for pairs or "A - John" for larger boats
    """
    if not rower_names:
        return lineup_id

    # Get number of seats from boat class
    num_seats = len(rower_names)

    # For 1x - just show the single rower's first name
    if num_seats == 1:
        return f"{lineup_id} - {get_first_name(rower_names[0])}"

    # For 2x/2- - show "stroke / bow" format with first names
    if num_seats == 2:
        stroke = get_first_name(rower_names[0])
        bow = get_first_name(rower_names[1])
        return f"{lineup_id} - {stroke}/{bow}"

    # For larger boats (4+, 4-, 4x, 8+) - just show stroke's first name
    stroke = get_first_name(rower_names[0])
    return f"{lineup_id} - {stroke}"


# =============================================================================
# STREAMLIT APP
# =============================================================================

def check_password():
    """Returns True if the user has the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if "password" not in st.session_state:
            return
        if st.session_state["password"] == get_app_password():
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        col_logo, col_title = st.columns([1, 10])
        with col_logo:
            st.image("wrc-badge-red.png", width=60)
        with col_title:
            st.title("Regatta Analytics")
        st.text_input(
            "Enter password to access the app:",
            type="password",
            on_change=password_entered,
            key="password"
        )
        st.info("Contact WRC leadership for access credentials.")
        return False

    if not st.session_state["password_correct"]:
        col_logo, col_title = st.columns([1, 10])
        with col_logo:
            st.image("wrc-badge-red.png", width=60)
        with col_title:
            st.title("Regatta Analytics")
        st.text_input(
            "Enter password to access the app:",
            type="password",
            on_change=password_entered,
            key="password"
        )
        st.error("Incorrect password. Please try again.")
        return False

    return True


def get_google_sheets_config():
    """Get Google Sheets configuration from Streamlit secrets"""
    try:
        # Check if Google Sheets is configured
        if "gcp_service_account" not in st.secrets:
            return None, None
        if "spreadsheet_id" not in st.secrets:
            return None, None

        credentials = dict(st.secrets["gcp_service_account"])
        spreadsheet_id = st.secrets["spreadsheet_id"]
        return credentials, spreadsheet_id
    except Exception:
        return None, None


def get_event_entries_sheet_id():
    """Get the Event Entries Google Sheet ID from secrets"""
    try:
        if "event_entries_sheet_id" in st.secrets:
            return st.secrets["event_entries_sheet_id"]
        return None
    except Exception:
        return None


def shorten_names(names: List[str]) -> List[str]:
    """Convert full names to short format: First or FirstL if duplicate first names.

    Example: ["John Smith", "Jane Doe", "John Adams"] -> ["JohnS", "Jane", "JohnA"]
    """
    if not names:
        return []

    # Extract first names and count duplicates
    first_names = []
    for name in names:
        if name and ' ' in name:
            first_names.append(name.split()[0])
        elif name:
            first_names.append(name)
        else:
            first_names.append("")

    # Count occurrences of each first name
    from collections import Counter
    first_name_counts = Counter(first_names)

    # Build short names
    short_names = []
    for name in names:
        if name and ' ' in name:
            parts = name.split()
            first_name = parts[0]
            last_initial = parts[-1][0].upper() if parts[-1] else ""
            # Use FirstL format if duplicate first names
            if first_name_counts.get(first_name, 0) > 1:
                short_names.append(f"{first_name}{last_initial}")
            else:
                short_names.append(first_name)
        elif name:
            short_names.append(name)
        else:
            short_names.append("")

    return short_names


def format_lineup_string(rowers: List[str], boat_class: str) -> str:
    """Format rowers list as lineup string: (Cox)-8-7-6-5-4-3-2-1 (stern to bow)

    For boats without cox (4-, 2-, 1x), omit the cox position.
    Rowers list is assumed to be in seat order (stroke to bow, cox last if present).
    Cox is wrapped in parentheses. Uses full names for data integrity.
    """
    # Clean up any "(needs cox)" fragments stuck to rower names
    cleaned_rowers = []
    for r in rowers:
        if r and isinstance(r, str):
            # Remove "(needs cox)" prefix if stuck to a name
            if r.lower().startswith("(needs cox)"):
                remainder = r[11:]  # len("(needs cox)") = 11
                if remainder:
                    cleaned_rowers.append(remainder)
            else:
                cleaned_rowers.append(r)
        elif r:
            cleaned_rowers.append(r)
    rowers = cleaned_rowers

    has_cox = '+' in boat_class or boat_class == '8+'
    boat_seats = {'1x': 1, '2x': 2, '2-': 2, '4x': 4, '4+': 4, '4-': 4, '8+': 8}
    expected_seats = boat_seats.get(boat_class, 4)

    if has_cox:
        # Check if cox is included (list has seats + 1 for cox)
        if len(rowers) > expected_seats:
            # Cox is included as last element
            cox_name = rowers[-1]
            cox = f"({cox_name})" if cox_name else "(needs cox)"
            seats = rowers[:expected_seats]
        else:
            # No cox included - just seats
            cox = "(needs cox)"
            seats = list(rowers)

        # We want: (Cox)-Stroke-7-6-5-4-3-2-Bow format
        all_parts = [cox] + seats
        return "-".join(all_parts)
    else:
        # No cox - just seats in stern-to-bow order
        return "-".join(rowers)


def parse_lineup_string(lineup_str: str, boat_class: str) -> List[str]:
    """Parse lineup string back to rowers list

    Input format is (Cox)-8-7-6-5-4-3-2-1 (cox first in parentheses for coxed boats).
    Output should be [8, 7, 6, 5, 4, 3, 2, 1, Cox] (cox last) to match internal format.
    If cox is "(needs cox)", it's a placeholder and should not be included.
    """
    if not lineup_str:
        return []

    # Handle malformed "(needs cox)Name" without hyphen separator
    if lineup_str.lower().startswith("(needs cox)") and not lineup_str.startswith("(needs cox)-"):
        # Strip the "(needs cox)" prefix and keep the rest
        lineup_str = lineup_str[11:]  # len("(needs cox)") = 11
        if lineup_str.startswith("-"):
            lineup_str = lineup_str[1:]

    parts = lineup_str.split("-")

    # Clean up any "(needs cox)" fragments that may be stuck to rower names
    cleaned_parts = []
    for part in parts:
        # Remove "(needs cox)" prefix if stuck to a name
        if part.lower().startswith("(needs cox)"):
            remainder = part[11:]  # len("(needs cox)") = 11
            if remainder:
                cleaned_parts.append(remainder)
            # else: skip empty string
        else:
            cleaned_parts.append(part)
    parts = cleaned_parts

    # For coxed boats, first element is cox - move to end
    has_cox = '+' in boat_class or boat_class == '8+'
    if has_cox and len(parts) > 1:
        cox = parts[0]
        seats = parts[1:]

        # Strip parentheses from cox name if present
        if cox.startswith("(") and cox.endswith(")"):
            cox = cox[1:-1]

        # "(needs cox)" is a placeholder - don't include it
        if cox.lower() == "needs cox":
            return seats
        else:
            return seats + [cox]

    return parts


def get_entries_gsheet_client():
    """Get authenticated gspread client for event entries sheet"""
    if not GSPREAD_AVAILABLE:
        st.warning("gspread library not available")
        return None, None

    credentials, _ = get_google_sheets_config()
    entries_sheet_id = get_event_entries_sheet_id()

    if not credentials:
        st.warning("No Google credentials found in secrets")
        return None, None

    if not entries_sheet_id:
        st.warning("No event_entries_sheet_id found in secrets")
        return None, None

    try:
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',  # Read/write access
            'https://www.googleapis.com/auth/drive.readonly'
        ]
        creds = Credentials.from_service_account_info(credentials, scopes=scopes)
        client = gspread.authorize(creds)
        spreadsheet = client.open_by_key(entries_sheet_id)
        # Try to get "Entries" worksheet, fall back to first sheet
        try:
            worksheet = spreadsheet.worksheet("Entries")
        except gspread.exceptions.WorksheetNotFound:
            worksheet = spreadsheet.sheet1
            st.info("Using first sheet (rename to 'Entries' for clarity)")
        return spreadsheet, worksheet
    except Exception as e:
        st.error(f"Failed to connect to Event Entries sheet: {type(e).__name__}: {e}")
        return None, None


def load_entries_from_gsheet() -> List[dict]:
    """Load all event entries from Google Sheet"""
    _, worksheet = get_entries_gsheet_client()
    if not worksheet:
        return []

    try:
        records = worksheet.get_all_records()
        entries = []
        for record in records:
            entry = {
                'regatta': record.get('Regatta', ''),
                'day': record.get('Day', ''),
                'event_number': int(record.get('Event Number', 0)) if record.get('Event Number') else 0,
                'event_name': record.get('Event Name', ''),
                'event_time': record.get('Event Time', ''),
                'entry_number': int(record.get('Entry Number', 0)) if record.get('Entry Number') else 0,
                'boat_class': record.get('Boat Class', ''),
                'category': record.get('Category', ''),
                'avg_age': float(record.get('Avg Age', 0)) if record.get('Avg Age') else 0,
                'rowers': parse_lineup_string(record.get('Lineup', ''), record.get('Boat Class', '')),
                'boat': record.get('Boat', ''),  # Club boat assignment
                'timestamp': record.get('Timestamp', ''),
                'row_number': records.index(record) + 2  # +2 for header row and 0-indexing
            }
            entries.append(entry)
        return entries
    except Exception as e:
        st.error(f"Failed to load entries: {e}")
        return []


def save_entry_to_gsheet(entry: dict) -> bool:
    """Save a single entry to Google Sheet"""
    _, worksheet = get_entries_gsheet_client()
    if not worksheet:
        return False

    try:
        lineup_str = format_lineup_string(entry['rowers'], entry['boat_class'])
        # Normalize day and time formats for consistency
        normalized_day = normalize_day_format(entry['day'])
        normalized_time = normalize_time_format(entry['event_time'])
        row = [
            entry['regatta'],
            normalized_day,
            entry['event_number'],
            entry['event_name'],
            normalized_time,
            entry['entry_number'],
            entry['boat_class'],
            entry['category'],
            entry.get('avg_age', ''),
            lineup_str,
            entry.get('boat', ''),  # Club boat assignment
            entry['timestamp']
        ]
        worksheet.append_row(row, value_input_option='USER_ENTERED')
        return True
    except Exception as e:
        st.error(f"Failed to save entry: {e}")
        return False


def delete_entry_from_gsheet(entry: dict) -> bool:
    """Delete an entry from Google Sheet by finding matching row"""
    _, worksheet = get_entries_gsheet_client()
    if not worksheet:
        return False

    try:
        # Find the row with matching key fields
        records = worksheet.get_all_records()
        for idx, record in enumerate(records):
            if (record.get('Regatta') == entry['regatta'] and
                record.get('Day') == entry['day'] and
                int(record.get('Event Number', 0)) == entry['event_number'] and
                int(record.get('Entry Number', 0)) == entry['entry_number']):
                # Found it - delete row (idx + 2 for header and 0-indexing)
                worksheet.delete_rows(idx + 2)
                return True
        return False
    except Exception as e:
        st.error(f"Failed to delete entry: {e}")
        return False


def update_entry_in_gsheet(original_entry: dict, updated_entry: dict) -> bool:
    """Update an existing entry in Google Sheet by finding and replacing the row"""
    _, worksheet = get_entries_gsheet_client()
    if not worksheet:
        return False

    try:
        # Find the row with matching key fields from original entry
        records = worksheet.get_all_records()
        for idx, record in enumerate(records):
            if (record.get('Regatta') == original_entry['regatta'] and
                record.get('Day') == original_entry['day'] and
                int(record.get('Event Number', 0)) == original_entry['event_number'] and
                int(record.get('Entry Number', 0)) == original_entry['entry_number']):
                # Found it - update the row
                row_num = idx + 2  # +2 for header and 0-indexing
                lineup_str = format_lineup_string(updated_entry['rowers'], updated_entry['boat_class'])
                # Normalize day and time formats for consistency
                normalized_day = normalize_day_format(updated_entry['day'])
                normalized_time = normalize_time_format(updated_entry['event_time'])
                row_data = [
                    updated_entry['regatta'],
                    normalized_day,
                    updated_entry['event_number'],
                    updated_entry['event_name'],
                    normalized_time,
                    updated_entry['entry_number'],
                    updated_entry['boat_class'],
                    updated_entry['category'],
                    updated_entry.get('avg_age', ''),
                    lineup_str,
                    updated_entry.get('boat', ''),  # Club boat assignment
                    updated_entry['timestamp']
                ]
                # Update all cells in the row
                for col_idx, value in enumerate(row_data):
                    worksheet.update_cell(row_num, col_idx + 1, value)
                return True
        return False
    except Exception as e:
        st.error(f"Failed to update entry: {e}")
        return False


def cleanup_entry_names_in_gsheet() -> int:
    """One-time cleanup: fix cox parentheses format in all entries.

    For coxed boats, ensures cox is in parentheses at the front.
    Uses full names (no shortening) for data integrity.
    Returns the number of entries updated.
    """
    _, worksheet = get_entries_gsheet_client()
    if not worksheet:
        return 0

    try:
        records = worksheet.get_all_records()
        updated_count = 0
        boat_seats = {'1x': 1, '2x': 2, '2-': 2, '4x': 4, '4+': 4, '4-': 4, '8+': 8}

        for idx, record in enumerate(records):
            lineup_str = record.get('Lineup', '')
            boat_class = record.get('Boat Class', '')

            if not lineup_str:
                continue

            has_cox = '+' in boat_class or boat_class == '8+'
            expected_seats = boat_seats.get(boat_class, 4)

            # Skip if already correctly formatted with "(needs cox)" or proper "(Name)"
            if lineup_str.startswith("(needs cox)-"):
                continue

            # Check if already has proper cox format "(Name)-..."
            if has_cox and lineup_str.startswith("("):
                close_idx = lineup_str.find(")")
                if close_idx > 1 and lineup_str[close_idx:close_idx+2] == ")-":
                    # Already has proper format - skip
                    continue

            # Repair broken "(needs-" entries (from previous buggy cleanup)
            if "needs" in lineup_str.lower() and "(" in lineup_str:
                # This is a broken entry - extract just the names
                cleaned = lineup_str
                for fragment in ["(needs-", "-cox)", "(needs cox)", "(needs", "cox)", "(", ")"]:
                    cleaned = cleaned.replace(fragment, "")
                parts = [p.strip() for p in cleaned.split("-") if p.strip()]
                if has_cox:
                    new_lineup_str = "(needs cox)-" + "-".join(parts)
                else:
                    new_lineup_str = "-".join(parts)
                row_num = idx + 2
                worksheet.update_cell(row_num, 10, new_lineup_str)
                updated_count += 1
                continue

            # Parse the current lineup
            parts = lineup_str.split("-")

            # Check if cox needs parentheses (coxed boat but first part not in parens)
            needs_parens = has_cox and len(parts) > 0 and parts[0] and not parts[0].startswith("(")

            if needs_parens:
                # For coxed boats, handle cox formatting
                if len(parts) > expected_seats:
                    # Has more than seats - last one must be cox (old format with cox at end)
                    cox = parts[-1]
                    seats = parts[:-1]
                    parts = [f"({cox})"] + seats
                else:
                    # No cox - add placeholder at front
                    parts = ["(needs cox)"] + parts

                new_lineup_str = "-".join(parts)

                # Update the cell (Lineup is column J = 10)
                row_num = idx + 2  # +2 for header and 0-indexing
                worksheet.update_cell(row_num, 10, new_lineup_str)
                updated_count += 1

        return updated_count
    except Exception as e:
        st.error(f"Failed to cleanup entries: {e}")
        return 0


def cleanup_entry_formats_in_gsheet() -> int:
    """Normalize day and time formats in existing entries for consistency.

    Standardizes:
    - Day format: "Sunday, March 8, 2026" (no leading zero on day)
    - Time format: "8:30 AM" (no leading zero on hour, with AM/PM)

    Returns number of rows updated.
    """
    _, worksheet = get_entries_gsheet_client()
    if not worksheet:
        return 0

    try:
        records = worksheet.get_all_records()
        updated_count = 0

        for idx, record in enumerate(records):
            day_value = record.get('Day', '')
            time_value = record.get('Event Time', '')

            # Normalize formats
            normalized_day = normalize_day_format(day_value)
            normalized_time = normalize_time_format(time_value)

            # Check if either changed
            day_changed = normalized_day != day_value
            time_changed = normalized_time != time_value

            if day_changed or time_changed:
                row_num = idx + 2  # +2 for header and 0-indexing

                # Update Day (column B = 2) if changed
                if day_changed:
                    worksheet.update_cell(row_num, 2, normalized_day)

                # Update Event Time (column E = 5) if changed
                if time_changed:
                    worksheet.update_cell(row_num, 5, normalized_time)

                updated_count += 1

        return updated_count
    except Exception as e:
        st.error(f"Failed to cleanup entry formats: {e}")
        return 0


def _load_data_impl():
    """Load data from Google Sheets (preferred) or local Excel file (fallback)"""
    roster_manager = RosterManager()
    errors = []

    # Try Google Sheets first
    credentials, spreadsheet_id = get_google_sheets_config()
    if not GSPREAD_AVAILABLE:
        errors.append("gspread library not available")
    elif not credentials:
        errors.append("No gcp_service_account in secrets")
    elif not spreadsheet_id:
        errors.append("No spreadsheet_id in secrets")
    else:
        try:
            success = roster_manager.load_from_google_sheets(spreadsheet_id, credentials)
            if success:
                return roster_manager, None, "google_sheets"
            else:
                errors.append(f"Google Sheets load failed: {'; '.join(roster_manager.load_log[-3:])}")
        except Exception as e:
            errors.append(f"Google Sheets error: {type(e).__name__}: {e}")

    # Fall back to local Excel file - use script directory for reliable path
    script_dir = Path(__file__).parent
    filepath = script_dir / "2026 WRC Racing Spreadsheet.xlsx"
    if filepath.exists():
        success = roster_manager.load_from_excel(str(filepath))
        if success:
            return roster_manager, None, "local_excel"
        else:
            errors.append(f"Excel load failed: {'; '.join(roster_manager.load_log[-3:])}")
    else:
        errors.append(f"Excel file not found at: {filepath}")

    return None, f"No data source available. Errors: {' | '.join(errors)}", None


@st.cache_resource(ttl=3600)  # Cache for 1 hour, then refresh from Google Sheets
def load_data(cache_version: int = 0):
    """Cached wrapper - pass different cache_version to force refresh"""
    import datetime
    result = _load_data_impl()
    load_time = datetime.datetime.now().strftime("%H:%M:%S")
    if result[0] is None:
        # Don't cache failures - clear and return
        st.cache_resource.clear()
        return result[0], result[1], result[2], load_time
    return result[0], result[1], result[2], load_time


def get_seat_labels(num_seats: int) -> List[str]:
    """Get seat labels for a given boat size"""
    labels = {
        1: ["Stroke"],
        2: ["Stroke", "Bow"],
        4: ["Stroke", "3", "2", "Bow"],
        8: ["Stroke", "7", "6", "5", "4", "3", "2", "Bow"],
    }
    return labels.get(num_seats, [str(i) for i in range(1, num_seats + 1)])


def render_dashboard(selected_regatta: str, roster_manager, format_event_time_func):
    """Render the regatta dashboard view with athlete x event grid"""
    import re
    from datetime import datetime

    if not selected_regatta or selected_regatta == "__all__":
        st.warning("Please select a specific regatta to view the dashboard.")
        return

    # Get entries for selected regatta (with partial matching)
    regatta_name = selected_regatta.split("|")[0] if "|" in selected_regatta else selected_regatta
    regatta_lower = regatta_name.lower()

    dashboard_entries = []
    for entry in st.session_state.event_entries:
        entry_regatta = entry.get('regatta', '').lower()
        if (entry_regatta == regatta_lower or
            entry_regatta in regatta_lower or
            regatta_lower in entry_regatta):
            dashboard_entries.append(entry)

    # Helper to parse event time for sorting
    def parse_time_for_sort(time_str: str):
        """Parse time string to sortable datetime"""
        if not time_str:
            return None
        time_str = time_str.strip().upper()
        # Remove seconds if present (only if there are two colons, e.g., 10:10:30 AM)
        if time_str.count(':') >= 2:
            time_str = re.sub(r':\d{2}(?=\s|$)', '', time_str)
        for fmt in ['%I:%M %p', '%H:%M', '%I:%M%p', '%I:%M  %p']:
            try:
                return datetime.strptime(time_str.replace('  ', ' '), fmt)
            except:
                continue
        return None

    # Build data structures for the grid
    # 1. Get all unique athletes from entries
    all_athletes = set()
    for entry in dashboard_entries:
        for rower in entry.get('rowers', []):
            all_athletes.add(rower)
    all_athletes = sorted(all_athletes)

    # 2. Get events - include targeted events from regatta_events AND events with entries
    events_dict = {}  # event_number -> event info
    events_with_entries = set()  # Track which events have entries

    # First, add all targeted events from the regatta_events data
    regatta_events_list = roster_manager.regatta_events.get(selected_regatta, [])
    for event in regatta_events_list:
        if event.include:  # Only targeted events
            events_dict[event.event_number] = {
                'number': event.event_number,
                'name': event.event_name,
                'time': event.event_time,
                'parsed_time': parse_time_for_sort(event.event_time),
                'has_entries': False
            }

    # Then, add/update events from entries
    for entry in dashboard_entries:
        event_num = entry.get('event_number')
        events_with_entries.add(event_num)
        if event_num not in events_dict:
            events_dict[event_num] = {
                'number': event_num,
                'name': entry.get('event_name', ''),
                'time': entry.get('event_time', ''),
                'parsed_time': parse_time_for_sort(entry.get('event_time', '')),
                'has_entries': True
            }
        else:
            events_dict[event_num]['has_entries'] = True

    # Sort events by time
    sorted_events = sorted(events_dict.values(), key=lambda e: (e['parsed_time'] or datetime.min, e['number']))

    if not sorted_events:
        st.info(f"No targeted events found for {regatta_name}.")
        return

    # 3. Build athlete -> events mapping with conflict detection
    athlete_events = {athlete: {} for athlete in all_athletes}  # athlete -> {event_num: [entries]}
    # Also track boat (equipment) usage
    all_boats_used = set()  # All club boats used in entries
    boat_events = {}  # boat_name -> {event_num: [entries]}

    for entry in dashboard_entries:
        event_num = entry.get('event_number')
        boat_class = entry.get('boat_class', '')
        entry_num = entry.get('entry_number', 1)
        rowers = entry.get('rowers', [])
        club_boat = entry.get('boat', '')  # Club boat assignment

        # Track boat usage (exclude "Private Boat" from hot seat tracking)
        if club_boat and club_boat.strip() and club_boat != "Private Boat":
            all_boats_used.add(club_boat)
            if club_boat not in boat_events:
                boat_events[club_boat] = {}
            if event_num not in boat_events[club_boat]:
                boat_events[club_boat][event_num] = []
            boat_events[club_boat][event_num].append({
                'boat_class': boat_class,
                'entry_num': entry_num,
                'event_name': entry.get('event_name', ''),
                'category': entry.get('category', '')
            })

        # Check if missing cox
        needs_cox = False
        if '+' in boat_class:
            expected = {'4+': 5, '8+': 9}.get(boat_class, 0)
            needs_cox = len(rowers) < expected

        # Find seat position for each rower
        for idx, rower in enumerate(rowers):
            if rower in athlete_events:
                if event_num not in athlete_events[rower]:
                    athlete_events[rower][event_num] = []
                # Determine seat label
                is_coxed = '+' in boat_class
                expected_seats = {'1x': 1, '2x': 2, '2-': 2, '4x': 4, '4+': 4, '4-': 4, '8+': 8}.get(boat_class, 4)
                if is_coxed and idx >= expected_seats:
                    seat = "Cox"
                else:
                    seat_labels_map = {
                        1: ["Stroke"],
                        2: ["Stroke", "Bow"],
                        4: ["Stroke", "3-seat", "2-seat", "Bow"],
                        8: ["Stroke", "7-seat", "6-seat", "5-seat", "4-seat", "3-seat", "2-seat", "Bow"]
                    }
                    labels = seat_labels_map.get(expected_seats, [])
                    seat = labels[idx] if idx < len(labels) else str(idx + 1)

                athlete_events[rower][event_num].append({
                    'seat': seat,
                    'boat': boat_class,
                    'club_boat': club_boat,  # Club boat name (equipment)
                    'entry_num': entry_num,
                    'needs_cox': needs_cox
                })

    # 4. Calculate hot seating colors for each athlete
    def get_minutes_between(time1, time2):
        """Get minutes between two parsed times"""
        if time1 and time2:
            delta = time2 - time1
            return delta.total_seconds() / 60
        return 999  # Unknown = assume OK

    athlete_colors = {athlete: {} for athlete in all_athletes}
    for athlete in all_athletes:
        events_for_athlete = []
        for event_num, entries in athlete_events[athlete].items():
            event_info = events_dict.get(event_num, {})
            events_for_athlete.append((event_num, event_info.get('parsed_time')))

        # Sort by time
        events_for_athlete.sort(key=lambda x: x[1] or datetime.min)

        prev_time = None
        for i, (event_num, event_time) in enumerate(events_for_athlete):
            if i == 0:
                color = "🟢"  # First event
            else:
                gap = get_minutes_between(prev_time, event_time)
                if gap >= 90:
                    color = "🟢"
                elif gap >= 60:
                    color = "🟡"
                elif gap >= 30:
                    color = "🟠"
                else:
                    color = "🔴"
            athlete_colors[athlete][event_num] = color
            prev_time = event_time

    # 4b. Calculate hot seating colors for each boat (equipment)
    boat_colors = {boat: {} for boat in all_boats_used}
    for boat in all_boats_used:
        events_for_boat = []
        for event_num, entries in boat_events[boat].items():
            event_info = events_dict.get(event_num, {})
            events_for_boat.append((event_num, event_info.get('parsed_time')))

        # Sort by time
        events_for_boat.sort(key=lambda x: x[1] or datetime.min)

        prev_time = None
        for i, (event_num, event_time) in enumerate(events_for_boat):
            if i == 0:
                color = "🟢"  # First event
            else:
                gap = get_minutes_between(prev_time, event_time)
                if gap >= 90:
                    color = "🟢"
                elif gap >= 60:
                    color = "🟡"
                elif gap >= 30:
                    color = "🟠"
                else:
                    color = "🔴"
            boat_colors[boat][event_num] = color
            prev_time = event_time

    # 5. Summary stats
    total_entries = len(dashboard_entries)
    total_targeted_events = len([e for e in sorted_events if True])  # All events in our list
    events_with_entries_count = len([e for e in sorted_events if e.get('has_entries', False)])
    athletes_with_hot_seats = sum(1 for a in athlete_colors if any(c in ['🟠', '🔴'] for c in athlete_colors[a].values()))
    boats_with_hot_seats = sum(1 for b in boat_colors if any(c in ['🟠', '🔴'] for c in boat_colors[b].values()))
    conflicts = sum(1 for a in athlete_events for e in athlete_events[a].values() if len(e) > 1)
    needs_cox_count = sum(1 for entry in dashboard_entries if '+' in entry.get('boat_class', '') and len(entry.get('rowers', [])) < {'4+': 5, '8+': 9}.get(entry.get('boat_class', ''), 0))
    needs_boat_count = sum(1 for entry in dashboard_entries if not entry.get('boat', '').strip())

    # Display summary
    st.subheader(f"📊 {regatta_name}")
    summary_cols = st.columns(8)
    with summary_cols[0]:
        st.metric("Events", f"{events_with_entries_count}/{total_targeted_events}")
    with summary_cols[1]:
        st.metric("Entries", total_entries)
    with summary_cols[2]:
        st.metric("Athletes", len(all_athletes))
    with summary_cols[3]:
        if athletes_with_hot_seats > 0:
            st.metric("🔥 Hot Seats", athletes_with_hot_seats)
        else:
            st.metric("Hot Seats", 0)
    with summary_cols[4]:
        if boats_with_hot_seats > 0:
            st.metric("🛶 Boat Hot Seats", boats_with_hot_seats)
        else:
            st.metric("Boat Hot Seats", 0)
    with summary_cols[5]:
        if needs_cox_count > 0:
            st.metric("📣 Needs Cox", needs_cox_count)
        else:
            st.metric("Needs Cox", 0)
    with summary_cols[6]:
        if needs_boat_count > 0:
            st.metric("🚣 Needs Boat", needs_boat_count)
        else:
            st.metric("Needs Boat", 0)
    with summary_cols[7]:
        if conflicts > 0:
            st.metric("⚠️ Conflicts", conflicts)
        else:
            st.metric("Conflicts", 0)

    # Athlete event count breakdown
    from collections import Counter
    event_counts = Counter(len(athlete_events[athlete]) for athlete in all_athletes)

    breakdown_parts = []
    for count in sorted(event_counts.keys()):
        num_athletes = event_counts[count]
        event_word = "event" if count == 1 else "events"
        breakdown_parts.append(f"{num_athletes} with {count} {event_word}")

    st.caption(f"**Athlete breakdown:** {' | '.join(breakdown_parts)}")

    # Legend and sort control
    legend_col, sort_col, overview_col = st.columns([4, 1, 1])
    with legend_col:
        st.caption("**Hot Seat Legend:** 🟢 90+ min gap | 🟡 60-89 min | 🟠 30-59 min | 🔴 <30 min")
    with sort_col:
        sort_options = {
            "Name (A-Z)": "name_asc",
            "Name (Z-A)": "name_desc",
            "Events (most)": "events_desc",
            "Events (least)": "events_asc",
            "Issues First": "issues_first"
        }
        athlete_sort = st.selectbox(
            "Sort",
            options=list(sort_options.keys()),
            label_visibility="collapsed",
            key="dashboard_athlete_sort"
        )
        sort_mode = sort_options[athlete_sort]
    with overview_col:
        if st.button("📋 Day Overview", key="day_overview_btn"):
            st.session_state.show_minimap = True

    # Day Overview Mini-Map Dialog
    @st.dialog("Day Overview", width="large")
    def show_minimap_dialog():
        """Render compact color-only grid showing entire day at a glance."""
        # Build minimap HTML
        minimap_html = """
        <style>
            .minimap-container {
                overflow: auto;
                max-height: 70vh;
            }
            .minimap-table {
                border-collapse: collapse;
                font-family: sans-serif;
                font-size: 12px;
            }
            .minimap-table th {
                background-color: #f0f0f0;
                padding: 2px 4px;
                font-size: 10px;
                font-weight: normal;
                border: 1px solid #ccc;
                white-space: nowrap;
                position: sticky;
                top: 0;
                z-index: 1;
            }
            .minimap-table th.corner {
                position: sticky;
                left: 0;
                top: 0;
                z-index: 2;
                background-color: #f0f0f0;
            }
            .minimap-table td.athlete-name {
                text-align: left;
                font-size: 11px;
                padding: 2px 6px;
                white-space: nowrap;
                position: sticky;
                left: 0;
                background-color: #fff;
                border: 1px solid #ccc;
                z-index: 1;
            }
            .minimap-table tr:nth-child(even) td.athlete-name {
                background-color: #fafafa;
            }
            .minimap-cell {
                width: 16px;
                height: 16px;
                min-width: 16px;
                min-height: 16px;
                border: 1px solid #ddd;
                padding: 0;
                position: relative;
                cursor: default;
            }
            .minimap-cell:hover {
                outline: 2px solid #333;
                z-index: 10;
            }
            .minimap-green { background-color: #4CAF50; }
            .minimap-yellow { background-color: #FFD700; }
            .minimap-orange { background-color: #FF9800; }
            .minimap-red { background-color: #F44336; }
            .minimap-gray { background-color: #e0e0e0; }
            .minimap-empty { background-color: #fff; }
            .minimap-conflict {
                background: repeating-linear-gradient(
                    45deg,
                    #F44336,
                    #F44336 3px,
                    #ffcccc 3px,
                    #ffcccc 6px
                );
                border: 2px solid #F44336;
            }
            .minimap-cox-warning.minimap-green {
                background: repeating-linear-gradient(
                    -45deg,
                    #4CAF50,
                    #4CAF50 3px,
                    #81C784 3px,
                    #81C784 6px
                );
            }
            .minimap-cox-warning.minimap-yellow {
                background: repeating-linear-gradient(
                    -45deg,
                    #FFD700,
                    #FFD700 3px,
                    #FFEB3B 3px,
                    #FFEB3B 6px
                );
            }
            .minimap-cox-warning.minimap-orange {
                background: repeating-linear-gradient(
                    -45deg,
                    #FF9800,
                    #FF9800 3px,
                    #FFB74D 3px,
                    #FFB74D 6px
                );
            }
            .minimap-cox-warning.minimap-red {
                background: repeating-linear-gradient(
                    -45deg,
                    #F44336,
                    #F44336 3px,
                    #E57373 3px,
                    #E57373 6px
                );
            }
        </style>
        <div class="minimap-container">
        <table class="minimap-table">
            <thead><tr><th class="corner"></th>
        """

        # Add event headers (shorthand)
        for event in sorted_events:
            shorthand = get_event_shorthand(event['name'])
            time_str = format_event_time_func(event['time']) if event['time'] else ""
            tooltip = f"{event['name']} @ {time_str}"
            minimap_html += f'<th title="{tooltip}">{shorthand}</th>'

        minimap_html += "</tr></thead><tbody>"

        # Add athlete rows
        for athlete in all_athletes:
            # Abbreviate long names
            display_name = athlete if len(athlete) <= 12 else athlete[:10] + "..."
            minimap_html += f'<tr><td class="athlete-name" title="{athlete}">{display_name}</td>'

            for event in sorted_events:
                event_num = event['number']
                has_entries = event.get('has_entries', False)

                if event_num in athlete_events[athlete]:
                    entries = athlete_events[athlete][event_num]
                    color = athlete_colors[athlete].get(event_num, "⚪")

                    # Check for conflict
                    if len(entries) > 1:
                        time_str = format_event_time_func(event['time']) if event['time'] else ""
                        tooltip = f"CONFLICT: {event['name']} @ {time_str}"
                        minimap_html += f'<td class="minimap-cell minimap-conflict" title="{tooltip}"></td>'
                    else:
                        entry_info = entries[0]
                        needs_cox = entry_info['needs_cox']
                        seat = entry_info['seat']
                        boat = entry_info['boat']
                        time_str = format_event_time_func(event['time']) if event['time'] else ""

                        # Map emoji color to CSS class
                        color_class = {
                            '🟢': 'minimap-green',
                            '🟡': 'minimap-yellow',
                            '🟠': 'minimap-orange',
                            '🔴': 'minimap-red'
                        }.get(color, 'minimap-gray')

                        cox_class = 'minimap-cox-warning' if needs_cox else ''
                        tooltip = f"{event['name']} @ {time_str}\\n{seat} in {boat}"
                        if needs_cox:
                            tooltip += "\\nNeeds Cox!"

                        minimap_html += f'<td class="minimap-cell {color_class} {cox_class}" title="{tooltip}"></td>'
                else:
                    # Empty cell - gray if event exists but rower not entered
                    if has_entries:
                        minimap_html += '<td class="minimap-cell minimap-gray"></td>'
                    else:
                        minimap_html += '<td class="minimap-cell minimap-empty"></td>'

            minimap_html += '</tr>'

        # Add separator row if there are boats
        if all_boats_used:
            minimap_html += f'<tr><td class="athlete-name" style="background-color:#e8e8e8;font-weight:bold;border-top:2px solid #999;">🛶 Equipment</td>'
            for _ in sorted_events:
                minimap_html += '<td class="minimap-cell" style="border-top:2px solid #999;"></td>'
            minimap_html += '</tr>'

            # Add boat rows
            for boat in sorted(all_boats_used):
                # Abbreviate long names
                display_name = boat if len(boat) <= 12 else boat[:10] + "..."
                minimap_html += f'<tr><td class="athlete-name" title="{boat}" style="font-style:italic;">{display_name}</td>'

                for event in sorted_events:
                    event_num = event['number']
                    has_entries = event.get('has_entries', False)

                    if boat in boat_events and event_num in boat_events[boat]:
                        entries = boat_events[boat][event_num]
                        color = boat_colors[boat].get(event_num, "⚪")
                        time_str = format_event_time_func(event['time']) if event['time'] else ""

                        # Map emoji color to CSS class
                        color_class = {
                            '🟢': 'minimap-green',
                            '🟡': 'minimap-yellow',
                            '🟠': 'minimap-orange',
                            '🔴': 'minimap-red'
                        }.get(color, 'minimap-gray')

                        entry_info = entries[0]
                        tooltip = f"{event['name']} @ {time_str}\\n{boat} ({entry_info['boat_class']})"

                        minimap_html += f'<td class="minimap-cell {color_class}" title="{tooltip}"></td>'
                    else:
                        # Empty cell - gray if event exists but boat not used
                        if has_entries:
                            minimap_html += '<td class="minimap-cell minimap-gray"></td>'
                        else:
                            minimap_html += '<td class="minimap-cell minimap-empty"></td>'

                minimap_html += '</tr>'

        minimap_html += "</tbody></table></div>"

        # Legend for minimap
        st.markdown("""
        **Legend:**
        <span style="display:inline-block;width:14px;height:14px;background:#4CAF50;border:1px solid #ccc;vertical-align:middle;"></span> 90+ min |
        <span style="display:inline-block;width:14px;height:14px;background:#FFD700;border:1px solid #ccc;vertical-align:middle;"></span> 60-89 min |
        <span style="display:inline-block;width:14px;height:14px;background:#FF9800;border:1px solid #ccc;vertical-align:middle;"></span> 30-59 min |
        <span style="display:inline-block;width:14px;height:14px;background:#F44336;border:1px solid #ccc;vertical-align:middle;"></span> <30 min |
        <span style="display:inline-block;width:14px;height:14px;background:#e0e0e0;border:1px solid #ccc;vertical-align:middle;"></span> Not entered |
        <span style="display:inline-block;width:14px;height:14px;background:repeating-linear-gradient(45deg,#F44336,#F44336 2px,#ffcccc 2px,#ffcccc 4px);border:2px solid #F44336;vertical-align:middle;"></span> Conflict |
        <span style="display:inline-block;width:14px;height:14px;background:repeating-linear-gradient(-45deg,#4CAF50,#4CAF50 3px,#81C784 3px,#81C784 6px);border:1px solid #ccc;vertical-align:middle;"></span> Needs Cox
        """, unsafe_allow_html=True)

        st.markdown(minimap_html, unsafe_allow_html=True)

        st.caption(f"Showing {len(all_athletes)} athletes + {len(all_boats_used)} boats × {len(sorted_events)} events")

    # Show dialog if triggered
    if st.session_state.get('show_minimap', False):
        st.session_state.show_minimap = False
        show_minimap_dialog()

    st.divider()

    # Equipment Schedule section (collapsible) - show only if boats are assigned
    if all_boats_used:
        with st.expander(f"🛶 Equipment Schedule ({len(all_boats_used)} boats)", expanded=False):
            # Build compact schedule for each boat
            for boat in sorted(all_boats_used):
                if boat not in boat_events:
                    continue
                boat_schedule = []
                for event_num in sorted(boat_events[boat].keys()):
                    event_info = events_dict.get(event_num, {})
                    time_str = format_event_time_func(event_info.get('time')) if event_info.get('time') else ""
                    color = boat_colors[boat].get(event_num, "⚪")
                    entry = boat_events[boat][event_num][0]
                    boat_schedule.append(f"{color} {time_str} - {entry['boat_class']} {entry['category']}")

                st.markdown(f"**{boat}:** {' → '.join(boat_schedule)}")

            # Show warning for boats with hot seats
            hot_seat_boats = [b for b in all_boats_used if any(c in ['🟠', '🔴'] for c in boat_colors.get(b, {}).values())]
            if hot_seat_boats:
                st.warning(f"⚠️ Hot seat boats (tight turnaround): {', '.join(hot_seat_boats)}")

    # 6. Build and display the grid using HTML table for better formatting
    if len(sorted_events) == 0:
        st.info("No events to display.")
        return

    # Sort athletes based on selection
    def get_athlete_event_count(athlete):
        return len(athlete_events.get(athlete, {}))

    def get_athlete_issue_count(athlete):
        # Count hot seats (orange/red only) + conflicts
        hot_seats = sum(1 for c in athlete_colors.get(athlete, {}).values() if c in ['🟠', '🔴'])
        conflicts = sum(1 for entries in athlete_events.get(athlete, {}).values() if len(entries) > 1)
        return hot_seats + conflicts

    if sort_mode == "name_asc":
        all_athletes = sorted(all_athletes)
    elif sort_mode == "name_desc":
        all_athletes = sorted(all_athletes, reverse=True)
    elif sort_mode == "events_desc":
        all_athletes = sorted(all_athletes, key=get_athlete_event_count, reverse=True)
    elif sort_mode == "events_asc":
        all_athletes = sorted(all_athletes, key=get_athlete_event_count)
    elif sort_mode == "issues_first":
        all_athletes = sorted(all_athletes, key=lambda a: (-get_athlete_issue_count(a), a))

    # Build HTML table with horizontal and vertical scroll container
    html = """
    <style>
        .dashboard-scroll-container {
            overflow: auto;
            max-width: 100%;
            max-height: 500px;
            border: 1px solid #ccc;
            border-radius: 4px;
        }
        .dashboard-table {
            border-collapse: separate;
            border-spacing: 0;
            font-family: sans-serif;
            font-size: 14px;
            min-width: 100%;
        }
        .dashboard-table th, .dashboard-table td {
            border: 1px solid #ccc;
            padding: 8px 12px;
            text-align: center;
            vertical-align: middle;
        }
        .dashboard-table thead th {
            background-color: #f0f0f0;
            font-weight: bold;
            position: sticky;
            top: 0;
            z-index: 2;
        }
        .dashboard-table th.athlete-header {
            position: sticky;
            left: 0;
            top: 0;
            z-index: 3;
            background-color: #f0f0f0;
        }
        .dashboard-table tr:nth-child(even) {
            background-color: #fafafa;
        }
        .dashboard-table tr:hover {
            background-color: #e8f4e8;
        }
        .dashboard-table .athlete-name {
            text-align: left;
            font-weight: bold;
            white-space: nowrap;
            position: sticky;
            left: 0;
            background-color: #ffffff;
            z-index: 1;
            border-right: 2px solid #999;
        }
        .dashboard-table tr:nth-child(odd) .athlete-name {
            background-color: #ffffff;
        }
        .dashboard-table tr:nth-child(even) .athlete-name {
            background-color: #fafafa;
        }
        .dashboard-table tr:hover .athlete-name {
            background-color: #e8f4e8;
        }
        .dashboard-table .event-header {
            min-width: 90px;
            max-width: 120px;
        }
        .dashboard-table .event-time {
            font-weight: bold;
            font-size: 13px;
        }
        .dashboard-table .event-name {
            font-size: 11px;
            color: #444;
            white-space: normal;
            word-wrap: break-word;
            line-height: 1.2;
        }
        .dashboard-table .cell-content {
            font-size: 13px;
        }
        .dashboard-table .boat-name {
            font-size: 9px;
            color: #666;
            display: block;
            line-height: 1.1;
        }
        .dashboard-table .conflict-cell {
            background-color: #ffcccc !important;
        }
        .dashboard-table .no-entries-header {
            background-color: #e0e0e0;
            color: #888;
        }
        .dashboard-table .no-entries-cell {
            background-color: #f5f5f5;
        }
    </style>
    <div class="dashboard-scroll-container">
    <table class="dashboard-table">
        <thead>
            <tr>
                <th class="athlete-header">Athlete</th>
    """

    # Add event headers
    for event in sorted_events:
        time_display = format_event_time_func(event['time']) if event['time'] else ""
        event_name = event['name']
        has_entries = event.get('has_entries', False)
        header_class = "event-header" if has_entries else "event-header no-entries-header"
        # Create tooltip with full name
        html += f'''<th class="{header_class}" title="{event_name}">
            <div class="event-time">{time_display}</div>
            <div class="event-name">{event_name}</div>
        </th>'''

    html += "</tr></thead><tbody>"

    # Add athlete rows
    for athlete in all_athletes:
        has_conflict = any(len(entries) > 1 for entries in athlete_events[athlete].values())
        prefix = "⚠️ " if has_conflict else ""
        event_count = len(athlete_events[athlete])
        count_display = f" ({event_count})" if event_count > 0 else ""

        html += f'<tr><td class="athlete-name">{prefix}{athlete}{count_display}</td>'

        for event in sorted_events:
            event_num = event['number']
            has_entries = event.get('has_entries', False)

            if event_num in athlete_events[athlete]:
                entries = athlete_events[athlete][event_num]
                color = athlete_colors[athlete].get(event_num, "⚪")

                if len(entries) > 1:
                    html += '<td class="conflict-cell"><span class="cell-content">⚠️ CONFLICT</span></td>'
                else:
                    entry_info = entries[0]
                    cox_warning = "📣" if entry_info['needs_cox'] else ""
                    boat = entry_info['boat']
                    seat = entry_info['seat']
                    club_boat = entry_info.get('club_boat', '')
                    club_boat_display = f'<span class="boat-name">{club_boat}</span>' if club_boat else ''
                    # Tooltip with boat class
                    html += f'<td title="{boat}"><span class="cell-content">{color} {seat}{cox_warning}</span>{club_boat_display}</td>'
            else:
                # Empty cell - gray if event has no entries at all
                cell_class = "no-entries-cell" if not has_entries else ""
                html += f'<td class="{cell_class}"></td>'

        html += '</tr>'

    html += "</tbody></table></div>"

    st.markdown(html, unsafe_allow_html=True)

    # =========================================================================
    # AVAILABLE ATHLETES PANEL - Help fill empty events
    # =========================================================================

    st.divider()
    st.subheader("🎯 Find Available Athletes")

    # Build list of events that need entries (targeted events without full entries)
    events_needing_athletes = []
    for event in sorted_events:
        event_num = event['number']
        # Count current entries for this event
        current_entries = sum(1 for e in dashboard_entries if e.get('event_number') == event_num)
        events_needing_athletes.append({
            'number': event_num,
            'name': event['name'],
            'time': event['time'],
            'parsed_time': event.get('parsed_time'),
            'current_entries': current_entries,
            'has_entries': event.get('has_entries', False)
        })

    # Create selectbox options
    event_options = {"-- Select an event --": None}
    for evt in events_needing_athletes:
        time_display = format_event_time_func(evt['time']) if evt['time'] else ""
        entry_count = f" [{evt['current_entries']} entries]" if evt['current_entries'] > 0 else " [no entries]"
        label = f"{time_display} - {evt['name']}{entry_count}"
        event_options[label] = evt

    selected_event_label = st.selectbox(
        "Show available athletes for:",
        options=list(event_options.keys()),
        key="available_athletes_event_select"
    )
    selected_event_for_fill = event_options.get(selected_event_label)

    if selected_event_for_fill:
        event_num = selected_event_for_fill['number']
        event_name = selected_event_for_fill['name']
        event_time = selected_event_for_fill.get('parsed_time')

        # Parse event requirements from name
        event_gender = parse_event_gender(event_name)  # 'M', 'W', 'Mixed', or None
        event_boat = parse_event_boat_class(event_name)  # '8+', '4+', etc or None
        event_category = parse_event_category(event_name)  # 'AA', 'A', 'B', etc or None

        # Auto-switch boat class and clear lineups if event changed
        valid_boats = ["1x", "2x", "2-", "4x", "4+", "4-", "8+"]
        last_event_key = 'last_find_athletes_event'

        # Only trigger when event actually changes (not on every boat class mismatch)
        last_event = st.session_state.get(last_event_key)
        event_changed = last_event != event_num

        if event_changed:
            # Event or boat changed - clear lineups and set boat class if detected
            st.session_state.pending_clear_lineups = True
            if event_boat and event_boat in valid_boats:
                st.session_state.pending_boat_class = event_boat
            st.session_state[last_event_key] = event_num
            st.rerun()
        st.session_state[last_event_key] = event_num

        st.caption(f"**Event:** {event_name}")
        st.caption(f"**Detected:** Gender={event_gender or 'Any'}, Boat={event_boat or 'Any'}, Category={event_category or 'Any'}")

        # Get all rowers signed up for this regatta
        regatta_for_filter = regatta_name  # Use the regatta name from earlier in function

        def is_attending_regatta(rower, regatta_check):
            if rower.is_attending(regatta_check):
                return True
            regatta_lower = regatta_check.lower()
            for signup_regatta, attending in rower.regatta_signups.items():
                if attending and (regatta_lower in signup_regatta.lower() or signup_regatta.lower() in regatta_lower):
                    return True
            return False

        # Get athletes already in this event
        athletes_in_event = []
        for entry in dashboard_entries:
            if entry.get('event_number') == event_num:
                for rower in entry.get('rowers', []):
                    athletes_in_event.append(rower)

        # Filter available athletes
        available_athletes = []
        for name, rower in roster_manager.rowers.items():
            # Must be signed up for regatta
            if not is_attending_regatta(rower, regatta_for_filter):
                continue

            # Must not already be in this event (use robust name matching)
            if is_name_in_list(name, athletes_in_event):
                continue


            # Check gender eligibility
            if event_gender:
                if event_gender == 'Mixed':
                    pass  # Anyone can row mixed
                elif event_gender == 'M' and rower.gender != 'M':
                    continue
                elif event_gender == 'W' and rower.gender != 'F':
                    continue

            # Check category eligibility (can race "down" but not "up")
            if event_category and rower.age:
                category_min_ages = {'AA': 0, 'A': 27, 'B': 36, 'C': 43, 'D': 50, 'E': 55, 'F': 60, 'G': 65, 'H': 70, 'I': 75, 'J': 80}
                event_min_age = category_min_ages.get(event_category, 0)
                if rower.age < event_min_age:
                    continue  # Too young for this category

            # Calculate hot-seat impact if added to this event
            hot_seat_color = "🟢"  # Default: good (no conflicts)
            min_gap_minutes = None

            if event_time:
                # Find all events this athlete is already entered in
                athlete_event_times = []
                for entry in dashboard_entries:
                    entry_rowers = entry.get('rowers', [])
                    if is_name_in_list(name, entry_rowers):
                        entry_event_num = entry.get('event_number')
                        # Get the parsed time for this entry's event
                        entry_event_info = events_dict.get(entry_event_num, {})
                        entry_event_time = entry_event_info.get('parsed_time')
                        if entry_event_time:
                            athlete_event_times.append(entry_event_time)

                if athlete_event_times:
                    # Calculate minimum gap to the potential new event
                    min_gap = float('inf')
                    for existing_time in athlete_event_times:
                        gap = abs((event_time - existing_time).total_seconds() / 60)
                        if gap < min_gap:
                            min_gap = gap

                    min_gap_minutes = min_gap
                    if min_gap < 30:
                        hot_seat_color = "🔴"
                    elif min_gap < 60:
                        hot_seat_color = "🟠"
                    elif min_gap < 90:
                        hot_seat_color = "🟡"
                    # else stays green (>= 90 min gap)

            # Get erg score info
            erg_info = ""
            erg_time_seconds = None  # For sorting
            if rower.scores:
                # Try to get 1k score
                score_1k = rower.scores.get(1000)
                if score_1k:
                    mins = int(score_1k.time_seconds // 60)
                    secs = score_1k.time_seconds % 60
                    erg_info = f"1k: {mins}:{secs:04.1f}"
                    erg_time_seconds = score_1k.time_seconds
                else:
                    # Get any available score
                    for dist, score in rower.scores.items():
                        mins = int(score.time_seconds // 60)
                        secs = score.time_seconds % 60
                        erg_info = f"{dist}m: {mins}:{secs:04.1f}"
                        erg_time_seconds = score.time_seconds
                        break

            # Count events this athlete is already entered in
            events_count = sum(1 for e in dashboard_entries if is_name_in_list(name, e.get('rowers', [])))

            available_athletes.append({
                'name': name,
                'rower': rower,
                'hot_seat_color': hot_seat_color,
                'min_gap': min_gap_minutes,
                'erg_info': erg_info,
                'erg_time_seconds': erg_time_seconds,
                'age': rower.age,
                'gender': rower.gender,
                'events_entered': events_count
            })

        # Initialize secondary sort state if needed (primary is always gap)
        # Format: 'name', 'age_desc', 'age_asc', 'erg_asc', 'erg_desc'
        if 'available_athletes_secondary_sort' not in st.session_state:
            st.session_state.available_athletes_secondary_sort = 'name'

        # Sort: primary by gap (hot-seat color), secondary by user choice
        color_order = {'🟢': 0, '🟡': 1, '🟠': 2, '🔴': 3}
        sort_mode = st.session_state.available_athletes_secondary_sort

        if sort_mode == 'age_desc':
            # Age descending (older first), None values last
            available_athletes.sort(key=lambda a: (
                color_order.get(a['hot_seat_color'], 4),
                a['age'] is None,
                -(a['age'] or 0),
                a['name']
            ))
        elif sort_mode == 'age_asc':
            # Age ascending (younger first), None values last
            available_athletes.sort(key=lambda a: (
                color_order.get(a['hot_seat_color'], 4),
                a['age'] is None,
                a['age'] or 999,
                a['name']
            ))
        elif sort_mode == 'erg_asc':
            # Erg time ascending (faster first), None values last
            available_athletes.sort(key=lambda a: (
                color_order.get(a['hot_seat_color'], 4),
                a['erg_time_seconds'] is None,
                a['erg_time_seconds'] or float('inf'),
                a['name']
            ))
        elif sort_mode == 'erg_desc':
            # Erg time descending (slower first), None values last
            available_athletes.sort(key=lambda a: (
                color_order.get(a['hot_seat_color'], 4),
                a['erg_time_seconds'] is None,
                -(a['erg_time_seconds'] or 0),
                a['name']
            ))
        else:
            # Secondary: name (default)
            available_athletes.sort(key=lambda a: (color_order.get(a['hot_seat_color'], 4), a['name']))

        if not available_athletes:
            st.info("No available athletes found for this event. Check eligibility requirements or regatta signups.")
        else:
            # Header row with sort buttons
            header_cols = st.columns([0.5, 2, 1, 1, 1.5, 0.5])
            with header_cols[0]:
                st.caption("")  # Hot-seat color column - no label needed
            with header_cols[1]:
                st.caption("Athlete")
            with header_cols[2]:
                # Age sort button - toggle between desc/asc
                if sort_mode == 'age_desc':
                    age_label = "**Age ↓**"
                elif sort_mode == 'age_asc':
                    age_label = "**Age ↑**"
                else:
                    age_label = "Age"
                if st.button(age_label, key="sort_age", use_container_width=True):
                    if sort_mode == 'age_desc':
                        st.session_state.available_athletes_secondary_sort = 'age_asc'
                    else:
                        st.session_state.available_athletes_secondary_sort = 'age_desc'
                    st.rerun()
            with header_cols[3]:
                st.caption("Time Gap")
            with header_cols[4]:
                # Erg sort button - toggle between asc/desc
                if sort_mode == 'erg_asc':
                    erg_label = "**Erg ↑**"
                elif sort_mode == 'erg_desc':
                    erg_label = "**Erg ↓**"
                else:
                    erg_label = "Erg"
                if st.button(erg_label, key="sort_erg", use_container_width=True):
                    if sort_mode == 'erg_asc':
                        st.session_state.available_athletes_secondary_sort = 'erg_desc'
                    else:
                        st.session_state.available_athletes_secondary_sort = 'erg_asc'
                    st.rerun()
            with header_cols[5]:
                st.caption("")  # Checkbox column - no label needed

            st.markdown("---")

            # Initialize selection state
            if 'selected_athletes' not in st.session_state:
                st.session_state.selected_athletes = set()

            # Get max seats for this boat class
            boat_seats = {'1x': 1, '2x': 2, '2-': 2, '4x': 4, '4+': 4, '4-': 4, '8+': 8}
            max_seats = boat_seats.get(event_boat, 8) if event_boat else 8

            # Calculate selected athletes from checkbox widget states (these are updated BEFORE rerun)
            # This ensures buttons reflect current selection even on the rerun triggered by checkbox click
            # Limit selections to max seats for the boat class
            selected_for_event = []
            for athlete in available_athletes:
                checkbox_key = f"select_{athlete['name']}_{event_num}"
                # Check widget state directly - Streamlit updates this before rerun
                if st.session_state.get(checkbox_key, False):
                    if len(selected_for_event) < max_seats:
                        selected_for_event.append(athlete['name'])
                        st.session_state.selected_athletes.add(athlete['name'])
                    else:
                        # Uncheck - exceeded max seats
                        st.session_state[checkbox_key] = False
                        st.session_state.selected_athletes.discard(athlete['name'])
                else:
                    st.session_state.selected_athletes.discard(athlete['name'])

            # Action bar - always visible, disabled when no selection
            btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
            has_selection = len(selected_for_event) > 0

            def clear_all_selections():
                """Clear both selected_athletes set and checkbox widget states"""
                st.session_state.selected_athletes.clear()
                for athlete in available_athletes:
                    checkbox_key = f"select_{athlete['name']}_{event_num}"
                    if checkbox_key in st.session_state:
                        st.session_state[checkbox_key] = False

            def add_to_lineup(lineup_key, lineup_name):
                lineup = st.session_state[lineup_key]
                added_count = 0
                for name in selected_for_event:
                    for i in range(len(lineup)):
                        if lineup[i] is None:
                            st.session_state[lineup_key][i] = name
                            added_count += 1
                            break
                clear_all_selections()
                if added_count > 0:
                    st.toast(f"Added {added_count} athlete(s) to {lineup_name}")
                if added_count < len(selected_for_event):
                    st.warning(f"{lineup_name} is full! Only added {added_count} of {len(selected_for_event)}")
                st.rerun()

            with btn_col1:
                if st.button("➕ Lineup A", key="add_sel_to_a", use_container_width=True, disabled=not has_selection):
                    add_to_lineup('lineup_a', 'Lineup A')
            with btn_col2:
                if st.button("➕ Lineup B", key="add_sel_to_b", use_container_width=True, disabled=not has_selection):
                    add_to_lineup('lineup_b', 'Lineup B')
            with btn_col3:
                if st.button("➕ Lineup C", key="add_sel_to_c", use_container_width=True, disabled=not has_selection):
                    add_to_lineup('lineup_c', 'Lineup C')
            with btn_col4:
                if st.button("✖ Clear", key="clear_sel", use_container_width=True, disabled=not has_selection):
                    clear_all_selections()
                    st.rerun()

            selection_info = f" ({len(selected_for_event)}/{max_seats} selected)" if selected_for_event else ""
            st.success(f"**{len(available_athletes)} available athletes**{selection_info}")

            # Display as a compact table with checkboxes
            for athlete in available_athletes:
                col1, col2, col3, col4, col5, col6 = st.columns([0.5, 2, 1, 1, 1.5, 0.5])
                with col1:
                    st.markdown(athlete['hot_seat_color'])
                with col2:
                    events_badge = f" ({athlete['events_entered']})" if athlete['events_entered'] > 0 else ""
                    st.markdown(f"**{athlete['name']}**{events_badge}")
                with col3:
                    gender_display = "♀" if athlete['gender'] == 'F' else "♂"
                    st.caption(f"{gender_display} {athlete['age'] or '-'}y")
                with col4:
                    if athlete['min_gap'] is not None:
                        st.caption(f"{int(athlete['min_gap'])}min gap")
                    else:
                        st.caption("No events")
                with col5:
                    st.caption(athlete['erg_info'] or "No erg")
                with col6:
                    # Disable checkbox if max seats reached and this athlete not already selected
                    is_selected = athlete['name'] in selected_for_event
                    at_max = len(selected_for_event) >= max_seats
                    st.checkbox("", key=f"select_{athlete['name']}_{event_num}",
                               disabled=(at_max and not is_selected),
                               label_visibility="collapsed")


def main():
    st.set_page_config(
        page_title="Regatta Analytics",
        page_icon="wrc-badge-red.png",
        layout="wide"
    )

    # Hide tooltips on selectbox options
    st.markdown("""
        <style>
        div[data-baseweb="select"] * {
            pointer-events: auto !important;
        }
        div[data-baseweb="select"] [title] {
            title: none !important;
        }
        div[data-baseweb="select"] *::after {
            content: none !important;
        }
        /* Disable browser native tooltips */
        [title] {
            pointer-events: auto;
        }
        </style>
    """, unsafe_allow_html=True)

    # Check password first
    if not check_password():
        return

    # Initialize cache version for refresh functionality
    if 'cache_version' not in st.session_state:
        st.session_state.cache_version = 0

    # Load data (pass cache_version to force refresh when incremented)
    roster_manager, error, data_source, load_time = load_data(st.session_state.cache_version)

    if error:
        st.error(f"Error: {error}")
        return

    # Store load time for display
    st.session_state.last_load_time = load_time

    analyzer = BoatAnalyzer(roster_manager)

    # Store data source for display
    if 'data_source' not in st.session_state:
        st.session_state.data_source = data_source

    # Initialize session state for lineups
    if 'lineup_a' not in st.session_state:
        st.session_state.lineup_a = [None] * 4
    if 'lineup_b' not in st.session_state:
        st.session_state.lineup_b = [None] * 4
    if 'lineup_c' not in st.session_state:
        st.session_state.lineup_c = [None] * 4

    # Initialize cox state for coxed boats (4+, 8+)
    if 'cox_a' not in st.session_state:
        st.session_state.cox_a = None
    if 'cox_b' not in st.session_state:
        st.session_state.cox_b = None
    if 'cox_c' not in st.session_state:
        st.session_state.cox_c = None

    # Initialize boat assignment state for club boats
    if 'boat_a' not in st.session_state:
        st.session_state.boat_a = None
    if 'boat_b' not in st.session_state:
        st.session_state.boat_b = None
    if 'boat_c' not in st.session_state:
        st.session_state.boat_c = None

    # Initialize locked seats state for each lineup (set of seat indices)
    if 'locked_seats_a' not in st.session_state:
        st.session_state.locked_seats_a = set()
    if 'locked_seats_b' not in st.session_state:
        st.session_state.locked_seats_b = set()
    if 'locked_seats_c' not in st.session_state:
        st.session_state.locked_seats_c = set()

    # Initialize selected rower state
    if 'selected_rower' not in st.session_state:
        st.session_state.selected_rower = None

    # Initialize event entries list - load from Google Sheets if available
    if 'event_entries' not in st.session_state:
        st.session_state.event_entries = load_entries_from_gsheet()
        if st.session_state.event_entries:
            st.toast(f"Loaded {len(st.session_state.event_entries)} event entries")

    # Initialize view mode (lineup or dashboard)
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = 'lineup'

    # Initialize selected regatta (to persist across view toggles)
    if 'selected_regatta_display' not in st.session_state:
        st.session_state.selected_regatta_display = "All Rowers"

    # Initialize autofill settings
    if 'autofill_gender' not in st.session_state:
        st.session_state.autofill_gender = "Men's"
    if 'autofill_target' not in st.session_state:
        st.session_state.autofill_target = "Lineup A"
    if 'autofill_use_event_constraints' not in st.session_state:
        st.session_state.autofill_use_event_constraints = False
    if 'autofill_checked_events' not in st.session_state:
        st.session_state.autofill_checked_events = set()
    if 'show_autofill_controls' not in st.session_state:
        st.session_state.show_autofill_controls = False

    # Initialize excluded rowers set (for autofill exclusion)
    if 'excluded_rowers' not in st.session_state:
        st.session_state.excluded_rowers = set()

    # =========================================================================
    # HEADER: Logo, Title, and View Toggle
    # =========================================================================

    title_cols = st.columns([1, 8, 2])
    with title_cols[0]:
        st.image("wrc-badge-red.png", width=50)
    with title_cols[1]:
        if st.session_state.view_mode == 'lineup':
            st.markdown("### Lineup Sandbox")
        else:
            st.markdown("### Regatta Dashboard")
    with title_cols[2]:
        if st.session_state.view_mode == 'lineup':
            if st.button("📊 Switch to Regatta Dashboard", type="primary", use_container_width=True):
                st.session_state.view_mode = 'dashboard'
                st.rerun()
        else:
            if st.button("🚣 Switch to Lineup Sandbox", type="primary", use_container_width=True):
                st.session_state.view_mode = 'lineup'
                st.rerun()

    # =========================================================================
    # CONTROLS: Regatta (always), plus Distance, Boat, etc. (lineup mode only)
    # =========================================================================

    # Regatta selection - always visible
    regatta_options = {"All Rowers": "__all__"}

    # First, group events by regatta name to know which have events and how many days
    # Use list to preserve order from events tab (assumed to be in calendar order)
    events_by_regatta = {}
    events_regatta_order = []
    for key in roster_manager.regatta_events.keys():
        regatta, day = key.split("|")
        if regatta not in events_by_regatta:
            events_by_regatta[regatta] = []
            events_regatta_order.append(regatta)
        events_by_regatta[regatta].append((key, day))

    # Track which signup regattas have events coverage
    regattas_with_events = set()
    for regatta_name in events_by_regatta.keys():
        # Check if any signup regatta matches this events regatta (partial match)
        for col in roster_manager.regattas:
            display = roster_manager.regatta_display_names.get(col, col)
            if regatta_name.lower() in display.lower() or display.lower() in regatta_name.lower():
                regattas_with_events.add(col)

    # Add events tab entries FIRST (regattas with events are priority)
    for regatta_name in events_regatta_order:
        day_list = events_by_regatta[regatta_name]
        if len(day_list) == 1:
            # Single day - no day suffix needed
            key, day = day_list[0]
            regatta_options[regatta_name] = key
        else:
            # Multiple days - add day suffix
            for key, day in day_list:
                short_day = day.split(",")[0] if "," in day else day
                display_name = f"{regatta_name} - {short_day}"
                regatta_options[display_name] = key

    # Add remaining signup sheet regattas (skip ones with events coverage)
    for col in roster_manager.regattas:
        if col not in regattas_with_events:
            display = roster_manager.regatta_display_names.get(col, col)
            regatta_options[display] = col

    # Get default index from session state
    regatta_options_list = list(regatta_options.keys())
    default_index = 0
    if st.session_state.selected_regatta_display in regatta_options_list:
        default_index = regatta_options_list.index(st.session_state.selected_regatta_display)

    # Different layouts for Dashboard vs Lineup Sandbox
    if st.session_state.view_mode == 'lineup':
        # Lineup mode: show all controls in a row
        control_cols = st.columns([2, 2, 2, 2, 2, 2])
        with control_cols[0]:
            selected_regatta_display = st.selectbox(
                "Regatta",
                options=regatta_options_list,
                index=default_index,
                label_visibility="collapsed",
                key=f"regatta_select_{st.session_state.cache_version}"
            )
    else:
        # Dashboard mode: just regatta dropdown
        selected_regatta_display = st.selectbox(
            "Regatta",
            options=regatta_options_list,
            index=default_index,
            label_visibility="collapsed",
            key=f"regatta_select_{st.session_state.cache_version}"
        )

    # Store in session state for persistence across view toggles
    st.session_state.selected_regatta_display = selected_regatta_display
    selected_regatta = regatta_options.get(selected_regatta_display, "__all__")

    # Initialize per-regatta distance storage
    if 'regatta_distances' not in st.session_state:
        st.session_state.regatta_distances = {}

    # Get saved distance for this regatta, default to 2000
    saved_distance = st.session_state.regatta_distances.get(selected_regatta, 2000)

    # Lineup mode: show remaining controls
    if st.session_state.view_mode == 'lineup':
        # Distance input (combined race/target distance)
        with control_cols[1]:
            race_distance = st.number_input(
                "Distance",
                min_value=500,
                max_value=10000,
                value=saved_distance,
                step=500,
                label_visibility="collapsed",
                help="Race distance in meters (500-10000)",
                key=f"distance_{selected_regatta}"
            )
            # Save distance for this regatta
            st.session_state.regatta_distances[selected_regatta] = race_distance
            target_distance = race_distance
    else:
        # Dashboard mode: use defaults
        race_distance = saved_distance
        target_distance = saved_distance

    # Boat class selection
    boat_options_list = ["1x", "2x", "2-", "4x", "4+", "4-", "8+"]
    # Initialize boat class in session state if not present
    if 'selected_boat_class' not in st.session_state:
        st.session_state.selected_boat_class = "4+"

    # Check for pending clear lineups from Find Available Athletes (new event selected)
    if 'pending_clear_lineups' in st.session_state:
        del st.session_state.pending_clear_lineups
        # Get boat class (use pending if available, otherwise current)
        boat_for_clear = st.session_state.get('pending_boat_class', st.session_state.selected_boat_class)
        boat_seats = {'1x': 1, '2x': 2, '2-': 2, '4x': 4, '4+': 4, '4-': 4, '8+': 8}
        new_seats = boat_seats.get(boat_for_clear, 4)
        st.session_state.lineup_a = [None] * new_seats
        st.session_state.lineup_b = [None] * new_seats
        st.session_state.lineup_c = [None] * new_seats
        st.session_state.cox_a = None
        st.session_state.cox_b = None
        st.session_state.cox_c = None
        st.session_state.boat_a = None
        st.session_state.boat_b = None
        st.session_state.boat_c = None
        # Clear locked seats
        st.session_state.locked_seats_a = set()
        st.session_state.locked_seats_b = set()
        st.session_state.locked_seats_c = set()

    # Check for pending boat class change from Find Available Athletes
    if 'pending_boat_class' in st.session_state:
        new_boat = st.session_state.pending_boat_class
        st.session_state.selected_boat_class = new_boat
        # Also set the widget's key directly so it updates
        st.session_state.boat_class_select = new_boat
        del st.session_state.pending_boat_class

    # Check for pending club boat assignment from Edit Entry
    if 'pending_boat_a' in st.session_state:
        new_club_boat = st.session_state.pending_boat_a
        st.session_state.boat_a = new_club_boat
        # Also set the widget's key directly so selectbox updates
        if new_club_boat and new_club_boat in roster_manager.club_boats:
            st.session_state.boat_select_lineup_a = new_club_boat
        else:
            st.session_state.boat_select_lineup_a = "(No boat assigned)"
        del st.session_state.pending_boat_a

    # Lineup mode: show boat selection and action buttons
    if st.session_state.view_mode == 'lineup':
        with control_cols[2]:
            # Get the index for the current boat class
            current_boat_value = st.session_state.get('selected_boat_class', '4+')
            boat_index = boat_options_list.index(current_boat_value) if current_boat_value in boat_options_list else 4
            boat_class = st.selectbox(
                "Boat",
                options=boat_options_list,
                index=boat_index,
                key="boat_class_select",
                label_visibility="collapsed"
            )
            # Sync back to session state
            st.session_state.selected_boat_class = boat_class

        with control_cols[3]:
            if st.button("Analyze", type="primary", use_container_width=True):
                st.session_state.analyze_clicked = True

        with control_cols[4]:
            if st.button("Clear All", type="secondary", use_container_width=True):
                boat_seats = {'1x': 1, '2x': 2, '2-': 2, '4x': 4, '4+': 4, '4-': 4, '8+': 8}
                num_seats = boat_seats.get(boat_class, 4)
                st.session_state.lineup_a = [None] * num_seats
                st.session_state.lineup_b = [None] * num_seats
                st.session_state.lineup_c = [None] * num_seats
                st.session_state.cox_a = None
                st.session_state.cox_b = None
                st.session_state.cox_c = None
                st.session_state.boat_a = None
                st.session_state.boat_b = None
                st.session_state.boat_c = None
                # Clear locked seats
                st.session_state.locked_seats_a = set()
                st.session_state.locked_seats_b = set()
                st.session_state.locked_seats_c = set()
                st.session_state.selected_rower = None
                st.rerun()

        with control_cols[5]:
            if st.button("Reload", type="secondary", use_container_width=True, help=f"Reload data from Google Sheets"):
                st.session_state.cache_version += 1
                # Also reload event entries
                st.session_state.event_entries = load_entries_from_gsheet()
                st.rerun()
    else:
        # Dashboard mode: use defaults
        boat_class = st.session_state.selected_boat_class

    # Calculate number of seats
    boat_seats = {
        '1x': 1, '2x': 2, '2-': 2, '4x': 4, '4+': 4, '4-': 4, '8+': 8
    }
    num_seats = boat_seats.get(boat_class, 4)

    # Resize lineups if boat size changed
    for lineup_key in ['lineup_a', 'lineup_b', 'lineup_c']:
        current = st.session_state[lineup_key]
        if len(current) != num_seats:
            new_lineup = [None] * num_seats
            for i in range(min(len(current), num_seats)):
                new_lineup[i] = current[i]
            st.session_state[lineup_key] = new_lineup
            # Clear locked seats when boat class changes (seat indices may no longer be valid)
            locked_key = f"locked_seats_{lineup_key.split('_')[1]}"
            st.session_state[locked_key] = set()

    # Check if selected regatta has events (uses "|" separator for regatta+day combos)
    has_events = selected_regatta in roster_manager.regatta_events
    # Always show events when regatta has events
    show_events_panel = has_events

    # Lineup mode: show Settings and Autofill expanders
    if st.session_state.view_mode == 'lineup':
        # Settings expander - Calculation, Predictor, Erg-to-Water
        with st.expander("⚙️ Settings", expanded=False):
            settings_col1, settings_col2, settings_col3 = st.columns([1, 1, 1])
            with settings_col1:
                calc_method = st.radio(
                    "Calculation",
                    options=["Split", "Watts"],
                    horizontal=True,
                    help="""**Split method**: Averages splits directly. More conservative, may better reflect real-world crew dynamics with varied abilities.

**Watts method**: Averages power (watts) then converts to split. Faster prediction, assumes perfect synchronization."""
                )
                calc_method = calc_method.lower()

            with settings_col2:
                pace_predictor = st.radio(
                    "Predictor",
                    options=["Power", "Paul's"],
                    horizontal=True,
                    help="""**Power Law**: Fits a personalized fatigue curve to each athlete's test data. Uses formula: Watts = k × Distance^b. More accurate when athlete has multiple test distances.

**Paul's Law**: Traditional formula adding 6 seconds per 500m split for each doubling of distance. Simple and reliable with single test score."""
                )
                pace_predictor = "power_law" if pace_predictor == "Power" else "pauls_law"

            with settings_col3:
                erg_to_water = st.toggle(
                    "Erg-to-Water",
                    value=True,
                    help="""**Erg-to-Water Adjustment**: Convert erg scores to projected on-water times using BioRow/Kleshnev boat factors.

**Formula**: On-Water Time = Erg Time × Boat Factor × Tech Efficiency × (Race Dist / Erg Dist)

**Boat Factors**: 8+ (0.93) | 4x (0.96) | 4- (1.00) | 4+/2x (1.04) | 2- (1.08) | 1x (1.16)"""
                )

        # =========================================================================
        # AUTOFILL CONTROLS (in expander)
        # =========================================================================

        # Create lineup optimizer instance
        lineup_optimizer = LineupOptimizer(roster_manager, analyzer)

        # Get number of rowing seats from boat class
        boat_seats = {'1x': 1, '2x': 2, '2-': 2, '4x': 4, '4+': 4, '4-': 4, '8+': 8}
        num_rowing_seats = boat_seats.get(boat_class, 4)

        with st.expander("⚡ Autofill Lineup", expanded=False):
            autofill_cols = st.columns([2, 2, 2, 2, 2, 2])

            with autofill_cols[0]:
                autofill_gender = st.selectbox(
                    "Gender",
                    options=["Men's", "Women's", "Mixed"],
                    index=["Men's", "Women's", "Mixed"].index(st.session_state.autofill_gender),
                    key="autofill_gender_select"
                )
                st.session_state.autofill_gender = autofill_gender

            with autofill_cols[1]:
                autofill_target = st.selectbox(
                    "Target",
                    options=["Lineup A", "Lineup B", "Lineup C", "All (Top 3)"],
                    index=["Lineup A", "Lineup B", "Lineup C", "All (Top 3)"].index(st.session_state.autofill_target),
                    key="autofill_target_select"
                )
                st.session_state.autofill_target = autofill_target

            with autofill_cols[2]:
                autofill_use_constraints = st.checkbox(
                    "Use Event Constraints",
                    value=st.session_state.autofill_use_event_constraints,
                    key="autofill_use_constraints_cb",
                    help="When checked, applies gender/age constraints from checked events"
                )
                st.session_state.autofill_use_event_constraints = autofill_use_constraints

            # Compute constraints from checked events
            autofill_min_avg_age = 0
            autofill_gender_constraint = autofill_gender
            autofill_boat_constraint = None
            constraint_warning = None

            if autofill_use_constraints and st.session_state.autofill_checked_events:
                checked_genders = set()
                max_min_age = 0
                checked_boats = set()

                events = roster_manager.regatta_events.get(selected_regatta, [])
                for event in events:
                    if event.event_number in st.session_state.autofill_checked_events:
                        event_gender = parse_event_gender(event.event_name)
                        event_boat = parse_event_boat_class(event.event_name)
                        event_category = parse_event_category(event.event_name)

                        if event_gender:
                            checked_genders.add(event_gender)
                        if event_boat:
                            checked_boats.add(event_boat)
                        if event_category:
                            min_age = get_category_min_age(event_category)
                            max_min_age = max(max_min_age, min_age)

                # Determine effective gender constraint
                if checked_genders:
                    if len(checked_genders) == 1:
                        gender_code = list(checked_genders)[0]
                        if gender_code == 'M':
                            autofill_gender_constraint = "Men's"
                        elif gender_code == 'W':
                            autofill_gender_constraint = "Women's"
                        elif gender_code == 'Mix':
                            autofill_gender_constraint = "Mixed"
                    else:
                        # Multiple genders checked - require mixed
                        autofill_gender_constraint = "Mixed"

                autofill_min_avg_age = max_min_age
                # Store in session state so it's available when button is clicked
                st.session_state.autofill_computed_min_age = autofill_min_avg_age

                # Check for conflicting boat classes
                if len(checked_boats) > 1:
                    constraint_warning = f"Warning: Checked events have different boat classes: {', '.join(checked_boats)}"
                elif len(checked_boats) == 1:
                    autofill_boat_constraint = list(checked_boats)[0]
                    if autofill_boat_constraint != boat_class:
                        constraint_warning = f"Warning: Event boat class ({autofill_boat_constraint}) differs from selected ({boat_class})"

            # Show constraint info
            if autofill_use_constraints:
                with autofill_cols[3]:
                    if st.session_state.autofill_checked_events:
                        constraint_info = f"{autofill_gender_constraint}"
                        if autofill_min_avg_age > 0:
                            constraint_info += f", Age≥{autofill_min_avg_age}"
                        st.caption(f"Constraints: {constraint_info}")
                    else:
                        st.caption("No events checked")

            # Autofill buttons
            with autofill_cols[4]:
                autofill_raw_clicked = st.button("⚡ Autofill Raw", use_container_width=True,
                                                  help="Find fastest lineup by raw erg time")

            with autofill_cols[5]:
                # Button label and behavior changes based on whether event constraints are active
                if autofill_use_constraints and autofill_min_avg_age > 0:
                    autofill_adj_clicked = st.button("⚡ Autofill Category", use_container_width=True,
                                                      help=f"Find fastest lineup meeting category min age ({autofill_min_avg_age})")
                else:
                    autofill_adj_clicked = st.button("⚡ Autofill Adj.", use_container_width=True,
                                                      help="Find fastest lineup by handicap-adjusted time")

            # Show constraint warning
            if constraint_warning:
                st.warning(constraint_warning)

            # Toggle for showing lock/exclude controls
            st.session_state.show_autofill_controls = st.checkbox(
                "🔒 Enable Locks & Exclusions",
                value=st.session_state.show_autofill_controls,
                key="show_autofill_controls_cb",
                help="Show controls to lock seats (preserve during autofill) and exclude rowers from autofill"
            )

        # Handle autofill button clicks
        if autofill_raw_clicked or autofill_adj_clicked:
            # Get min_avg_age from session state (more reliable than local variable)
            effective_min_avg_age = st.session_state.get('autofill_computed_min_age', autofill_min_avg_age)

            # Determine optimization mode
            if autofill_raw_clicked:
                optimize_for = 'raw'
            elif autofill_use_constraints and effective_min_avg_age > 0:
                # With event constraints, optimize for fastest raw time that meets category
                optimize_for = 'category'
            else:
                optimize_for = 'adjusted'

            num_lineups = 3 if autofill_target == "All (Top 3)" else 1

            # Get effective constraints
            effective_gender = autofill_gender_constraint if autofill_use_constraints else autofill_gender

            # Helper to get locked rowers dict for a lineup slot
            def get_locked_rowers_for_slot(slot: str) -> dict:
                """Get dict of {seat_index: rower_name} for locked seats in a lineup."""
                locked_seats = st.session_state.get(f'locked_seats_{slot}', set())
                lineup = st.session_state.get(f'lineup_{slot}', [])
                locked_rowers = {}
                for seat_idx in locked_seats:
                    if seat_idx < len(lineup) and lineup[seat_idx]:
                        locked_rowers[seat_idx] = lineup[seat_idx]
                return locked_rowers

            # Determine which slot(s) we're filling and their locked rowers
            if autofill_target == "All (Top 3)":
                # Check if any lineup has locked seats
                locks_a = get_locked_rowers_for_slot('a')
                locks_b = get_locked_rowers_for_slot('b')
                locks_c = get_locked_rowers_for_slot('c')
                any_locks = bool(locks_a or locks_b or locks_c)

                if any_locks:
                    # If any lineup has locks, call separately for each so each respects its own locks
                    all_optimal = []
                    for slot, locked_rowers in [('a', locks_a), ('b', locks_b), ('c', locks_c)]:
                        slot_lineups = lineup_optimizer.find_optimal_lineups(
                            regatta=selected_regatta,
                            num_seats=num_rowing_seats,
                            boat_class=boat_class,
                            target_distance=target_distance,
                            calc_method=calc_method,
                            predictor=pace_predictor,
                            optimize_for=optimize_for,
                            gender=effective_gender,
                            min_avg_age=effective_min_avg_age,
                            num_results=1,
                            excluded_rowers=st.session_state.excluded_rowers,
                            locked_rowers=locked_rowers
                        )
                        if slot_lineups:
                            all_optimal.append(slot_lineups[0])
                        else:
                            all_optimal.append(None)
                    optimal_lineups = [x for x in all_optimal if x is not None]
                else:
                    # No locks - use single call to get top 3 different lineups
                    optimal_lineups = lineup_optimizer.find_optimal_lineups(
                        regatta=selected_regatta,
                        num_seats=num_rowing_seats,
                        boat_class=boat_class,
                        target_distance=target_distance,
                        calc_method=calc_method,
                        predictor=pace_predictor,
                        optimize_for=optimize_for,
                        gender=effective_gender,
                        min_avg_age=effective_min_avg_age,
                        num_results=3,
                        excluded_rowers=st.session_state.excluded_rowers,
                        locked_rowers=None
                    )
                    all_optimal = None  # Flag that we used single-call mode
            else:
                # Single lineup - get its locked rowers
                target_slot = autofill_target.split()[-1].lower()  # "Lineup A" -> "a"
                locked_rowers = get_locked_rowers_for_slot(target_slot)

                # Find optimal lineups
                optimal_lineups = lineup_optimizer.find_optimal_lineups(
                    regatta=selected_regatta,
                    num_seats=num_rowing_seats,
                    boat_class=boat_class,
                    target_distance=target_distance,
                    calc_method=calc_method,
                    predictor=pace_predictor,
                    optimize_for=optimize_for,
                    gender=effective_gender,
                    min_avg_age=effective_min_avg_age,
                    num_results=num_lineups,
                    excluded_rowers=st.session_state.excluded_rowers,
                    locked_rowers=locked_rowers
                )

            if not optimal_lineups:
                # Count eligible rowers for better error message
                gender_filter = None
                if effective_gender == "Men's":
                    gender_filter = 'M'
                elif effective_gender == "Women's":
                    gender_filter = 'W'
                eligible = lineup_optimizer.get_eligible_rowers(selected_regatta, gender_filter, 0)

                if len(eligible) < num_rowing_seats:
                    st.warning(f"Could not find valid lineup. Only {len(eligible)} eligible rowers with erg scores (need {num_rowing_seats}).")
                elif effective_gender == "Mixed":
                    # Check if we have enough of each gender for 50/50 split
                    half_seats = num_rowing_seats // 2
                    males = [r for r in eligible if r.gender == 'M']
                    females = [r for r in eligible if r.gender == 'F']
                    if len(males) < half_seats or len(females) < half_seats:
                        st.warning(f"Could not find valid Mixed lineup. Need {half_seats} of each gender, "
                                  f"but only have {len(males)} men and {len(females)} women with erg scores.")
                    elif optimize_for == 'category' and effective_min_avg_age > 0:
                        ages = sorted([r.age for r in eligible], reverse=True)
                        max_avg_age = sum(ages[:num_rowing_seats]) / num_rowing_seats if ages else 0
                        st.warning(f"Could not find Mixed lineup meeting category min age ({effective_min_avg_age}). "
                                  f"Max possible avg age: {max_avg_age:.1f}")
                    else:
                        st.warning(f"Could not find valid Mixed lineup from {len(eligible)} eligible rowers "
                                  f"({len(males)} men, {len(females)} women).")
                elif optimize_for == 'category' and effective_min_avg_age > 0:
                    # Calculate max possible avg age from eligible rowers
                    ages = sorted([r.age for r in eligible], reverse=True)
                    max_avg_age = sum(ages[:num_rowing_seats]) / num_rowing_seats if ages else 0
                    st.warning(f"Could not find lineup meeting category min age ({effective_min_avg_age}). "
                              f"Max possible avg age from eligible rowers: {max_avg_age:.1f}")
                else:
                    st.warning(f"Could not find valid lineup from {len(eligible)} eligible rowers.")
            else:
                # Assign lineups to slots
                if autofill_target == "All (Top 3)":
                    if all_optimal is not None:
                        # Per-slot mode (with locks) - all_optimal has results for [a, b, c] in order
                        # Each may be a result dict or None if it failed
                        for slot, lineup_data in zip(['a', 'b', 'c'], all_optimal):
                            if lineup_data:
                                rower_names = lineup_data['rowers']

                                # Resize lineup if needed
                                while len(rower_names) < num_rowing_seats:
                                    rower_names.append(None)

                                setattr_name = f'lineup_{slot}'
                                st.session_state[setattr_name] = rower_names[:num_rowing_seats]

                                # Format time for toast
                                if optimize_for == 'adjusted':
                                    time_val = lineup_data['adjusted_time']
                                    time_label = "adj"
                                else:
                                    time_val = lineup_data['raw_time']
                                    time_label = "raw"
                                time_str = format_time(time_val)
                                age_str = f"Avg age: {lineup_data['avg_age']:.1f}"
                                st.toast(f"Lineup {slot.upper()}: {time_str} {time_label} ({age_str})")
                    else:
                        # Single-call mode (no locks) - optimal_lineups has top 3 different lineups
                        for i, slot in enumerate(['a', 'b', 'c']):
                            if i < len(optimal_lineups):
                                lineup_data = optimal_lineups[i]
                                rower_names = lineup_data['rowers']

                                # Resize lineup if needed
                                while len(rower_names) < num_rowing_seats:
                                    rower_names.append(None)

                                setattr_name = f'lineup_{slot}'
                                st.session_state[setattr_name] = rower_names[:num_rowing_seats]

                                # Format time for toast
                                if optimize_for == 'adjusted':
                                    time_val = lineup_data['adjusted_time']
                                    time_label = "adj"
                                else:
                                    time_val = lineup_data['raw_time']
                                    time_label = "raw"
                                time_str = format_time(time_val)
                                age_str = f"Avg age: {lineup_data['avg_age']:.1f}"
                                st.toast(f"Lineup {slot.upper()}: {time_str} {time_label} ({age_str})")
                else:
                    # Single lineup target
                    lineup_slots = []
                    if autofill_target == "Lineup A":
                        lineup_slots = ['a']
                    elif autofill_target == "Lineup B":
                        lineup_slots = ['b']
                    elif autofill_target == "Lineup C":
                        lineup_slots = ['c']

                    for i, slot in enumerate(lineup_slots):
                        if i < len(optimal_lineups):
                            lineup_data = optimal_lineups[i]
                            rower_names = lineup_data['rowers']

                            # Resize lineup if needed
                            while len(rower_names) < num_rowing_seats:
                                rower_names.append(None)

                            setattr_name = f'lineup_{slot}'
                            st.session_state[setattr_name] = rower_names[:num_rowing_seats]

                            # Format time for toast
                            if optimize_for == 'adjusted':
                                time_val = lineup_data['adjusted_time']
                                time_label = "adj"
                            else:
                                time_val = lineup_data['raw_time']
                                time_label = "raw"
                            time_str = format_time(time_val)
                            age_str = f"Avg age: {lineup_data['avg_age']:.1f}"
                            st.toast(f"Lineup {slot.upper()}: {time_str} {time_label} ({age_str})")

                st.rerun()

        st.divider()
    else:
        # Dashboard mode: set default values for variables used later
        calc_method = "split"
        pace_predictor = "power_law"
        erg_to_water = False

    # =========================================================================
    # SIDEBAR: Roster with Sort, Search, Women/Men
    # =========================================================================

    with st.sidebar:
        st.header("Roster")

        # Sort selection
        sort_options = {
            "Name": "name",
            "Age (young)": "age_asc",
            "Age (old)": "age_desc",
            "1K (raw)": "erg_1k_raw",
            "1K (handicap)": "erg_1k_hcap",
            "5K (raw)": "erg_5k_raw",
            "5K (handicap)": "erg_5k_hcap",
            "Port Only": "port",
            "Starboard Only": "starboard"
        }
        sort_col1, sort_col2 = st.columns(2)
        with sort_col1:
            sort_mode = st.selectbox(
                "Sort by",
                options=list(sort_options.keys())
            )
            sort_mode = sort_options[sort_mode]

        # Secondary sort by events entered
        secondary_sort_options = {
            "None": "none",
            "Events (most)": "events_desc",
            "Events (least)": "events_asc"
        }
        with sort_col2:
            secondary_sort = st.selectbox(
                "Then by",
                options=list(secondary_sort_options.keys())
            )
            secondary_sort = secondary_sort_options[secondary_sort]

        # Helper to count events entered for a rower in selected regatta
        def count_events_entered(rower_name: str) -> int:
            try:
                if not has_events:
                    return 0
                regatta_name = selected_regatta.split("|")[0] if "|" in selected_regatta else selected_regatta
                regatta_lower = regatta_name.lower()
                count = 0
                for entry in st.session_state.event_entries:
                    entry_regatta = entry['regatta'].lower()
                    # Match if exact or partial match (one contains the other)
                    regatta_match = (entry_regatta == regatta_lower or
                                    entry_regatta in regatta_lower or
                                    regatta_lower in entry_regatta)
                    rowers_list = entry.get('rowers', [])
                    if not isinstance(rowers_list, list):
                        rowers_list = []
                    if regatta_match and is_name_in_list(rower_name, rowers_list):
                        count += 1
                return count
            except Exception as e:
                st.error(f"Error in count_events_entered: {e}")
                return 0

        # Search filter
        search_term = st.text_input("Search", key="search")

        # Get filtered rower list - show ALL rowers, not just those with scores
        if selected_regatta and selected_regatta != "__all__":
            # For regatta filter, show attending rowers (with or without scores)
            # Handle regatta|day format from events tab by extracting just the regatta name
            regatta_for_filter = selected_regatta.split("|")[0] if "|" in selected_regatta else selected_regatta

            # Try exact match first, then partial match for events tab regattas
            def is_attending_regatta(rower, regatta_name):
                # Direct match
                if rower.is_attending(regatta_name):
                    return True
                # For events tab regattas, try case-insensitive partial match on signup columns
                regatta_lower = regatta_name.lower()
                for signup_regatta, attending in rower.regatta_signups.items():
                    if attending and (regatta_lower in signup_regatta.lower() or signup_regatta.lower() in regatta_lower):
                        return True
                return False

            rower_names = [name for name, r in roster_manager.rowers.items()
                          if is_attending_regatta(r, regatta_for_filter)]
            rower_names = sorted(rower_names)
        else:
            rower_names = roster_manager.get_all_rowers()

        # Get rower objects
        rowers_list = [(name, roster_manager.get_rower(name)) for name in rower_names]
        rowers_list = [(n, r) for n, r in rowers_list if r is not None]

        # Helper function to calculate masters handicap
        def get_handicap_seconds(age: int, distance: int) -> float:
            """Calculate handicap: ((age - 27)^2) * K * distance_multiplier"""
            k_factor = 0.02
            distance_multiplier = distance / 1000
            return ((age - 27) ** 2) * k_factor * distance_multiplier

        # Helper to get projected time for a distance (uses Paul's Law if needed)
        def get_projected_time(rower, target_distance: int) -> tuple:
            """Returns (time_seconds, is_projected) tuple"""
            # First check if they have the exact distance
            score = rower.scores.get(target_distance)
            if score:
                return (score.time_seconds, False)

            # Otherwise project from closest available score
            closest_score = rower.get_closest_score(target_distance)
            if not closest_score:
                return (float('inf'), False)

            # Apply Paul's Law to project split
            projected_split = PhysicsEngine.pauls_law_projection(
                closest_score.split_500m, closest_score.distance, target_distance
            )
            # Convert split to total time
            projected_time = (projected_split / 500) * target_distance
            return (projected_time, True)

        # Helper to get raw total time for a distance
        def get_raw_time(rower, distance: int) -> float:
            time, _ = get_projected_time(rower, distance)
            return time

        # Helper to get handicap-adjusted time
        def get_adjusted_time(rower, distance: int) -> float:
            time, _ = get_projected_time(rower, distance)
            if time == float('inf'):
                return float('inf')
            handicap = get_handicap_seconds(rower.age, distance)
            return time - handicap  # Lower is better

        # Apply sorting
        show_erg_time = None  # Track if we should show erg times
        if sort_mode == 'name':
            rowers_list.sort(key=lambda x: x[0])
        elif sort_mode == 'age_asc':
            rowers_list.sort(key=lambda x: x[1].age)
        elif sort_mode == 'age_desc':
            rowers_list.sort(key=lambda x: x[1].age, reverse=True)
        elif sort_mode == 'erg_1k_raw':
            rowers_list.sort(key=lambda x: get_raw_time(x[1], 1000))
            show_erg_time = ('1k', 'raw')
        elif sort_mode == 'erg_1k_hcap':
            rowers_list.sort(key=lambda x: get_adjusted_time(x[1], 1000))
            show_erg_time = ('1k', 'hcap')
        elif sort_mode == 'erg_5k_raw':
            rowers_list.sort(key=lambda x: get_raw_time(x[1], 5000))
            show_erg_time = ('5k', 'raw')
        elif sort_mode == 'erg_5k_hcap':
            rowers_list.sort(key=lambda x: get_adjusted_time(x[1], 5000))
            show_erg_time = ('5k', 'hcap')
        elif sort_mode == 'port':
            rowers_list = [(n, r) for n, r in rowers_list if r.side_port]
            rowers_list.sort(key=lambda x: (0 if (x[1].side_port and not x[1].side_port) else 1, x[0]))
        elif sort_mode == 'starboard':
            rowers_list = [(n, r) for n, r in rowers_list if r.side_starboard]
            rowers_list.sort(key=lambda x: (0 if (x[1].side_starboard and not x[1].side_port) else 1, x[0]))

        # Apply secondary sort by events entered
        if secondary_sort == 'events_desc':
            rowers_list.sort(key=lambda x: count_events_entered(x[0]), reverse=True)
        elif secondary_sort == 'events_asc':
            rowers_list.sort(key=lambda x: count_events_entered(x[0]))

        # Apply search filter
        if search_term:
            search_lower = search_term.lower()
            rowers_list = [(n, r) for n, r in rowers_list if search_lower in n.lower()]

        # Split by gender
        women = [(n, r) for n, r in rowers_list if r.gender == 'F']
        men = [(n, r) for n, r in rowers_list if r.gender == 'M']

        # Helper to format time for display
        def format_erg_time(rower, distance, show_type):
            time, is_projected = get_projected_time(rower, distance)
            if time == float('inf'):
                return "N/A"
            if show_type == 'hcap':
                handicap = get_handicap_seconds(rower.age, distance)
                time = time - handicap
            mins = int(time // 60)
            secs = time % 60
            time_str = f"{mins}:{secs:04.1f}"
            # Add asterisk if projected
            return f"{time_str}*" if is_projected else time_str

        # Helper to format event count with color indicator
        def format_event_indicator(count: int) -> str:
            if count == 0:
                return "⚪0"
            elif count <= 2:
                return f"🟢{count}"
            elif count == 3:
                return f"🟡{count}"
            elif count == 4:
                return f"🟠{count}"
            else:
                return f"🔴{count}"

        # Check if autofill controls should be shown
        show_autofill_ui = st.session_state.get('show_autofill_controls', False)

        # Women section
        st.markdown(f"**Women ({len(women)})**")
        for name, rower in women:
            has_scores = bool(rower.scores)
            side = rower.side_preference_str()
            # Get event count if regatta has events
            event_count = count_events_entered(name) if has_events else 0
            event_indicator = f" {format_event_indicator(event_count)}" if has_events else ""

            # Check if rower is excluded (only show prefix when controls are visible)
            is_excluded = name in st.session_state.excluded_rowers
            exclude_prefix = "(X) " if (is_excluded and show_autofill_ui) else ""

            if not has_scores:
                display_text = f"{exclude_prefix}{name}{event_indicator} | (no scores)"
            elif show_erg_time:
                dist = 1000 if show_erg_time[0] == '1k' else 5000
                erg_time = format_erg_time(rower, dist, show_erg_time[1])
                display_text = f"{exclude_prefix}{name}{event_indicator} | {erg_time}"
            else:
                display_text = f"{exclude_prefix}{name}{event_indicator} | {rower.age} | {side}"

            is_selected = (st.session_state.selected_rower == name)
            btn_type = "primary" if is_selected else "secondary"

            # Use columns for button and exclude toggle (only when autofill controls enabled)
            if has_scores and show_autofill_ui:
                roster_cols = st.columns([6, 1], vertical_alignment="center")
                with roster_cols[0]:
                    if st.button(display_text, key=f"rower_{name}", use_container_width=True, type=btn_type):
                        if is_selected:
                            st.session_state.selected_rower = None
                        else:
                            st.session_state.selected_rower = name
                        st.rerun()
                with roster_cols[1]:
                    # Use checkbox for exclude toggle - checked means EXCLUDED
                    new_excluded = st.checkbox(
                        "X",
                        value=is_excluded,
                        key=f"exclude_{name}",
                        label_visibility="collapsed"
                    )
                    if new_excluded != is_excluded:
                        if new_excluded:
                            st.session_state.excluded_rowers.add(name)
                        else:
                            st.session_state.excluded_rowers.discard(name)
                        st.rerun()
            else:
                if st.button(display_text, key=f"rower_{name}", use_container_width=True, type=btn_type, disabled=not has_scores):
                    if is_selected:
                        st.session_state.selected_rower = None
                    else:
                        st.session_state.selected_rower = name
                    st.rerun()

        st.divider()

        # Men section
        st.markdown(f"**Men ({len(men)})**")
        for name, rower in men:
            has_scores = bool(rower.scores)
            side = rower.side_preference_str()
            # Get event count if regatta has events
            event_count = count_events_entered(name) if has_events else 0
            event_indicator = f" {format_event_indicator(event_count)}" if has_events else ""

            # Check if rower is excluded (only show prefix when controls are visible)
            is_excluded = name in st.session_state.excluded_rowers
            exclude_prefix = "(X) " if (is_excluded and show_autofill_ui) else ""

            if not has_scores:
                display_text = f"{exclude_prefix}{name}{event_indicator} | (no scores)"
            elif show_erg_time:
                dist = 1000 if show_erg_time[0] == '1k' else 5000
                erg_time = format_erg_time(rower, dist, show_erg_time[1])
                display_text = f"{exclude_prefix}{name}{event_indicator} | {erg_time}"
            else:
                display_text = f"{exclude_prefix}{name}{event_indicator} | {rower.age} | {side}"

            is_selected = (st.session_state.selected_rower == name)
            btn_type = "primary" if is_selected else "secondary"

            # Use columns for button and exclude toggle (only when autofill controls enabled)
            if has_scores and show_autofill_ui:
                roster_cols = st.columns([6, 1], vertical_alignment="center")
                with roster_cols[0]:
                    if st.button(display_text, key=f"rower_{name}", use_container_width=True, type=btn_type):
                        if is_selected:
                            st.session_state.selected_rower = None
                        else:
                            st.session_state.selected_rower = name
                        st.rerun()
                with roster_cols[1]:
                    # Use checkbox for exclude toggle - checked means EXCLUDED
                    new_excluded = st.checkbox(
                        "X",
                        value=is_excluded,
                        key=f"exclude_{name}",
                        label_visibility="collapsed"
                    )
                    if new_excluded != is_excluded:
                        if new_excluded:
                            st.session_state.excluded_rowers.add(name)
                        else:
                            st.session_state.excluded_rowers.discard(name)
                        st.rerun()
            else:
                if st.button(display_text, key=f"rower_{name}", use_container_width=True, type=btn_type, disabled=not has_scores):
                    if is_selected:
                        st.session_state.selected_rower = None
                    else:
                        st.session_state.selected_rower = name
                    st.rerun()

    # =========================================================================
    # MAIN AREA: Lineups or Dashboard (based on view mode)
    # =========================================================================

    if st.session_state.view_mode == 'dashboard':
        # ----- DASHBOARD VIEW -----
        render_dashboard(selected_regatta, roster_manager, format_event_time)
        st.stop()  # Don't render lineup view

    # ----- LINEUP VIEW -----

    # Helper to get US Rowing Masters age category (defined early for edit mode banner)
    def get_masters_category(avg_age: float) -> str:
        """Return US Rowing Masters category based on average age"""
        if avg_age < 27:
            return "AA"
        elif avg_age < 36:
            return "A"
        elif avg_age < 43:
            return "B"
        elif avg_age < 50:
            return "C"
        elif avg_age < 55:
            return "D"
        elif avg_age < 60:
            return "E"
        elif avg_age < 65:
            return "F"
        elif avg_age < 70:
            return "G"
        elif avg_age < 75:
            return "H"
        elif avg_age < 80:
            return "I"
        else:
            return "J"

    # Edit mode banner
    editing_entry = st.session_state.get('editing_entry')
    editing_event = st.session_state.get('editing_event')
    if editing_entry and editing_event:
        edit_col1, edit_col2, edit_col3 = st.columns([4, 1, 1])
        with edit_col1:
            st.warning(f"✏️ **Editing Entry {editing_entry['entry_number']}** for {editing_event.event_name}")
        with edit_col2:
            if st.button("💾 Save Changes", use_container_width=True, type="primary"):
                # Build updated entry from current lineup
                lineup = st.session_state.lineup_a
                cox_key = 'cox_a'
                is_coxed = '+' in boat_class
                rower_names = [r for r in lineup if r is not None]
                cox_name = st.session_state.get(cox_key) if is_coxed else None
                if cox_name:
                    rower_names.append(cox_name)

                if rower_names:
                    from datetime import datetime
                    # Calculate new stats
                    ages = [roster_manager.rowers[name].age for name in rower_names if name in roster_manager.rowers and roster_manager.rowers[name].age]
                    avg_age = sum(ages) / len(ages) if ages else 0
                    genders = [roster_manager.rowers[name].gender for name in rower_names if name in roster_manager.rowers]
                    lineup_gender = 'M' if all(g == 'M' for g in genders) else ('W' if all(g == 'W' for g in genders) else 'Mixed')

                    updated_entry = {
                        'regatta': editing_entry['regatta'],
                        'day': editing_entry['day'],
                        'event_number': editing_entry['event_number'],
                        'event_name': editing_entry['event_name'],
                        'event_time': editing_entry['event_time'],
                        'entry_number': editing_entry['entry_number'],
                        'boat_class': boat_class,
                        'category': f"{lineup_gender} {get_masters_category(avg_age) if avg_age >= 21 else 'AA'}",
                        'avg_age': round(avg_age, 1),
                        'rowers': rower_names,
                        'boat': st.session_state.get('boat_a', ''),  # Club boat assignment
                        'timestamp': datetime.now().isoformat()
                    }

                    # Update in Google Sheets
                    if update_entry_in_gsheet(editing_entry, updated_entry):
                        st.toast("Entry updated in Google Sheets")
                        # Update in session state
                        for i, e in enumerate(st.session_state.event_entries):
                            if (e['regatta'] == editing_entry['regatta'] and
                                e['day'] == editing_entry['day'] and
                                e['event_number'] == editing_entry['event_number'] and
                                e['entry_number'] == editing_entry['entry_number']):
                                st.session_state.event_entries[i] = updated_entry
                                break
                    else:
                        st.error("Failed to update entry")

                    # Clear edit mode
                    del st.session_state['editing_entry']
                    del st.session_state['editing_event']
                    st.rerun()
                else:
                    st.error("Lineup is empty - add rowers before saving")

        with edit_col3:
            if st.button("✖ Cancel", use_container_width=True):
                # Clear edit mode and reset lineup
                del st.session_state['editing_entry']
                del st.session_state['editing_event']
                st.session_state.lineup_a = [None] * num_seats
                st.session_state.cox_a = None
                st.session_state.boat_a = None
                st.toast("Edit cancelled")
                st.rerun()

    # Selection indicator (fixed height container to prevent page shift)
    sel_col1, sel_col2 = st.columns([4, 1])
    with sel_col1:
        if st.session_state.selected_rower:
            st.info(f"**Selected:** {st.session_state.selected_rower} — Click a seat to place")
        else:
            st.info("Select a rower from the sidebar, then click a seat to place them")
    with sel_col2:
        if st.session_state.selected_rower:
            if st.button("Clear Selection", use_container_width=True):
                st.session_state.selected_rower = None
                st.rerun()

    seat_labels = get_seat_labels(num_seats)

    # Helper to calculate lineup stats
    def get_lineup_stats(lineup_list, roster_mgr):
        """Calculate average age, category, and gender for a full lineup"""
        rower_names = [r for r in lineup_list if r is not None]
        if len(rower_names) != len(lineup_list):
            return None  # Not full
        ages = []
        genders = set()
        for name in rower_names:
            rower = roster_mgr.get_rower(name)
            if rower:
                ages.append(rower.age)
                if rower.gender:
                    g = rower.gender.upper()
                    # Convert 'F' (Female) to 'W' (Women) for rowing convention
                    if g == 'F':
                        g = 'W'
                    genders.add(g)
        if not ages:
            return None
        avg_age = sum(ages) / len(ages)
        category = get_masters_category(avg_age)
        # Determine gender: M, W, or Mix
        if len(genders) == 1:
            gender = genders.pop()
        elif len(genders) > 1:
            gender = 'Mix'
        else:
            gender = 'M'  # Default
        return (avg_age, category, gender)

    # Create columns based on whether events panel is shown
    if show_events_panel and has_events:
        lineup_cols = st.columns([1, 1, 1, 0.6])
        event_col = lineup_cols[3]
    else:
        lineup_cols = st.columns(3)
        event_col = None

    lineups_config = [
        ("Lineup A", "lineup_a", lineup_cols[0]),
        ("Lineup B", "lineup_b", lineup_cols[1]),
        ("Lineup C", "lineup_c", lineup_cols[2])
    ]

    # Store per-lineup tech efficiency values for use in analysis
    lineup_tech_efficiencies = {}

    for title, key, col in lineups_config:
        with col:
            lineup = st.session_state[key]

            # Per-lineup tech efficiency (only shown when erg-to-water is enabled)
            if erg_to_water:
                tech_key = f"tech_efficiency_{key}"
                # Initialize with default value
                if tech_key not in st.session_state:
                    st.session_state[tech_key] = 1.05
                lineup_tech = st.number_input(
                    f"{title} Tech Eff",
                    min_value=0.90,
                    max_value=1.20,
                    step=0.01,
                    format="%.2f",
                    key=tech_key,
                    help=f"Tech efficiency for {title}. 1.00 = National team | 1.05 = Excellent club | 1.10 = Intermediate club"
                )
                # Store the actual widget value for use in analysis
                lineup_tech_efficiencies[key] = lineup_tech
            else:
                lineup_tech_efficiencies[key] = 1.05

            stats = get_lineup_stats(lineup, roster_manager)
            if stats:
                avg_age, category, gender = stats
                gender_prefix = {"M": "Men's", "W": "Women's", "Mix": "Mixed"}.get(gender, "")
                st.markdown(f"**{title}** :gray[Avg: {avg_age:.1f} | Cat: {gender_prefix} {category}]")
            else:
                st.markdown(f"**{title}**")

            # Boat assignment dropdown (club boats) - displayed at top
            boat_key = f"boat_{key.split('_')[1]}"  # boat_a, boat_b, boat_c
            current_boat = st.session_state.get(boat_key)
            boat_options = ["(No boat assigned)"] + roster_manager.club_boats
            current_boat_idx = 0
            if current_boat and current_boat in roster_manager.club_boats:
                current_boat_idx = roster_manager.club_boats.index(current_boat) + 1

            boat_cols = st.columns([5, 1])
            with boat_cols[0]:
                selected_boat = st.selectbox(
                    "Boat",
                    options=boat_options,
                    index=current_boat_idx,
                    key=f"boat_select_{key}",
                    label_visibility="collapsed"
                )
                # Update session state if changed
                if selected_boat == "(No boat assigned)":
                    if st.session_state.get(boat_key) is not None:
                        st.session_state[boat_key] = None
                elif selected_boat != st.session_state.get(boat_key):
                    st.session_state[boat_key] = selected_boat
            with boat_cols[1]:
                if current_boat:
                    if st.button("❌", key=f"remove_boat_{key}"):
                        st.session_state[boat_key] = None
                        st.rerun()

            # Cox slot for coxed boats (4+, 8+)
            is_coxed_boat = '+' in boat_class
            cox_key = f"cox_{key.split('_')[1]}"  # cox_a, cox_b, cox_c
            if is_coxed_boat:
                cox_name = st.session_state.get(cox_key)
                cox_label = f"Cox: {cox_name}" if cox_name else "Cox: ..."

                cox_cols = st.columns([5, 1])
                with cox_cols[0]:
                    if st.button(cox_label, key=f"cox_{key}", use_container_width=True):
                        if st.session_state.selected_rower:
                            selected = st.session_state.selected_rower
                            # Cox can be same person in multiple lineups, just check not already a rower in this lineup
                            if selected in lineup:
                                st.warning(f"{selected} is already rowing in {title}!")
                            else:
                                st.session_state[cox_key] = selected
                                st.session_state.selected_rower = None
                                st.rerun()
                        elif cox_name:
                            st.session_state.selected_rower = cox_name
                            st.rerun()

                with cox_cols[1]:
                    if cox_name:
                        if st.button("❌", key=f"remove_cox_{key}"):
                            st.session_state[cox_key] = None
                            st.rerun()

            # Get the locked seats set for this lineup
            lineup_letter = key.split('_')[1]  # 'a', 'b', or 'c'
            locked_seats_key = f"locked_seats_{lineup_letter}"
            locked_seats = st.session_state.get(locked_seats_key, set())

            # Check if autofill controls should be shown
            show_lock_controls = st.session_state.get('show_autofill_controls', False)

            for i, label in enumerate(seat_labels):
                rower_name = lineup[i] if i < len(lineup) else None
                is_locked = i in locked_seats

                # Use secondary buttons for both, distinguish by text
                # Add lock icon prefix when locked (only if controls are visible)
                lock_prefix = "🔒 " if (is_locked and show_lock_controls) else ""
                if rower_name:
                    btn_label = f"{label}: {lock_prefix}{rower_name}"
                else:
                    btn_label = f"{label}: ..."

                # Use 5:1:1 columns when lock controls are shown, otherwise 5:1
                if show_lock_controls:
                    seat_cols = st.columns([5, 1, 1])
                else:
                    seat_cols = st.columns([5, 1])

                with seat_cols[0]:
                    if st.button(btn_label, key=f"seat_{key}_{i}", use_container_width=True):
                        # Only enforce lock restrictions when controls are visible
                        if is_locked and show_lock_controls:
                            st.warning(f"Seat {label} is locked. Unlock it first to modify.")
                        elif st.session_state.selected_rower:
                            selected = st.session_state.selected_rower
                            if selected in lineup:
                                # Rower is already in this lineup - move or swap
                                old_pos = lineup.index(selected)
                                # Check if old position is locked (only if controls visible)
                                if old_pos in locked_seats and show_lock_controls:
                                    st.warning(f"Cannot move {selected} - their current seat is locked.")
                                elif old_pos != i:
                                    if rower_name:
                                        # Swap: put current seat's rower in selected's old position
                                        st.session_state[key][old_pos] = rower_name
                                    else:
                                        # Move: clear old position
                                        st.session_state[key][old_pos] = None
                                    st.session_state[key][i] = selected
                                    st.session_state.selected_rower = None
                                    st.rerun()
                                else:
                                    st.session_state.selected_rower = None
                                    st.rerun()
                            else:
                                # Rower not in lineup yet - assign to seat
                                st.session_state[key][i] = selected
                                st.session_state.selected_rower = None
                                st.rerun()
                        elif rower_name:
                            st.session_state.selected_rower = rower_name
                            st.rerun()

                if show_lock_controls:
                    with seat_cols[1]:
                        # Lock toggle button - only show if seat has a rower
                        if rower_name:
                            lock_icon = "🔓" if is_locked else "🔒"
                            lock_help = "Unlock seat" if is_locked else "Lock seat (preserve during autofill)"
                            if st.button(lock_icon, key=f"lock_{key}_{i}", use_container_width=True, help=lock_help):
                                if is_locked:
                                    locked_seats.discard(i)
                                else:
                                    locked_seats.add(i)
                                st.session_state[locked_seats_key] = locked_seats
                                st.rerun()

                    with seat_cols[2]:
                        # Delete button - only show if seat has a rower AND is not locked
                        if rower_name and not is_locked:
                            if st.button("❌", key=f"remove_{key}_{i}"):
                                st.session_state[key][i] = None
                                st.rerun()
                else:
                    with seat_cols[1]:
                        # Delete button (no lock controls)
                        if rower_name:
                            if st.button("❌", key=f"remove_{key}_{i}"):
                                st.session_state[key][i] = None
                                st.rerun()

            # Copy buttons (also copy cox and boat for coxed boats)
            copy_cols = st.columns(2)
            if key == "lineup_a":
                if copy_cols[0].button("A→B", key="copy_a_b", use_container_width=True):
                    st.session_state.lineup_b = st.session_state.lineup_a.copy()
                    st.session_state.cox_b = st.session_state.cox_a
                    st.session_state.boat_b = st.session_state.boat_a
                    st.rerun()
                if copy_cols[1].button("A→C", key="copy_a_c", use_container_width=True):
                    st.session_state.lineup_c = st.session_state.lineup_a.copy()
                    st.session_state.cox_c = st.session_state.cox_a
                    st.session_state.boat_c = st.session_state.boat_a
                    st.rerun()
            elif key == "lineup_b":
                if copy_cols[0].button("B→A", key="copy_b_a", use_container_width=True):
                    st.session_state.lineup_a = st.session_state.lineup_b.copy()
                    st.session_state.cox_a = st.session_state.cox_b
                    st.session_state.boat_a = st.session_state.boat_b
                    st.rerun()
                if copy_cols[1].button("B→C", key="copy_b_c", use_container_width=True):
                    st.session_state.lineup_c = st.session_state.lineup_b.copy()
                    st.session_state.cox_c = st.session_state.cox_b
                    st.session_state.boat_c = st.session_state.boat_b
                    st.rerun()
            else:
                if copy_cols[0].button("C→A", key="copy_c_a", use_container_width=True):
                    st.session_state.lineup_a = st.session_state.lineup_c.copy()
                    st.session_state.cox_a = st.session_state.cox_c
                    st.session_state.boat_a = st.session_state.boat_c
                    st.rerun()
                if copy_cols[1].button("C→B", key="copy_c_b", use_container_width=True):
                    st.session_state.lineup_b = st.session_state.lineup_c.copy()
                    st.session_state.cox_b = st.session_state.cox_c
                    st.session_state.boat_b = st.session_state.boat_c
                    st.rerun()

            # Copy lineup names to clipboard button
            # Build rower list: seats first, then cox at end (for coxed boats)
            rower_names_list = [r for r in lineup if r is not None]
            cox_name_for_entry = st.session_state.get(cox_key) if is_coxed_boat else None
            if cox_name_for_entry:
                rower_names_list.append(cox_name_for_entry)
            if rower_names_list:
                # Find duplicate first names in the roster
                all_first_names = []
                for full_name in roster_manager.rowers.keys():
                    parts = full_name.split()
                    if parts:
                        all_first_names.append(parts[0])
                duplicate_first_names = set(n for n in all_first_names if all_first_names.count(n) > 1)

                # Format as "First" or "FirstL" (if duplicate) joined by dashes
                short_names = []
                for name in rower_names_list:
                    parts = name.split()
                    if len(parts) >= 2:
                        first_name = parts[0]
                        last_initial = parts[-1][0]
                        if first_name in duplicate_first_names:
                            short_names.append(f"{first_name}{last_initial}")
                        else:
                            short_names.append(first_name)
                    else:
                        short_names.append(name)
                names_text = "-".join(short_names)
                b64_names = base64.b64encode(names_text.encode('utf-8')).decode('ascii')
                btn_id = f"copyLineup_{key}"
                components.html(f"""
                    <html>
                    <head><style>
                        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                        html, body {{ height: 100%; overflow: hidden; margin: 0; padding: 0; }}
                        body {{ padding-left: 2px; padding-right: 2px; }}
                        button {{ display: block; width: 100%; }}
                    </style></head>
                    <body>
                    <button id="{btn_id}" data-text="{b64_names}" onclick="
                        var text = atob(this.getAttribute('data-text'));
                        navigator.clipboard.writeText(text).then(function() {{
                            document.getElementById('{btn_id}').innerText = 'Copied!';
                            document.getElementById('{btn_id}').style.background = '#28a745';
                            document.getElementById('{btn_id}').style.color = 'white';
                            document.getElementById('{btn_id}').style.borderColor = '#28a745';
                            setTimeout(function() {{
                                document.getElementById('{btn_id}').innerText = 'Copy Lineup';
                                document.getElementById('{btn_id}').style.background = 'white';
                                document.getElementById('{btn_id}').style.color = 'rgb(49, 51, 63)';
                                document.getElementById('{btn_id}').style.borderColor = 'rgba(49, 51, 63, 0.2)';
                            }}, 2000);
                        }}).catch(function(err) {{
                            document.getElementById('{btn_id}').innerText = 'Failed';
                            document.getElementById('{btn_id}').style.background = '#dc3545';
                        }});
                    " style="
                        padding: 0.25rem 0.75rem;
                        height: 100%;
                        font-family: 'Source Sans Pro', sans-serif;
                        font-size: 14px;
                        font-weight: 400;
                        line-height: 1.6;
                        cursor: pointer;
                        border-radius: 8px;
                        border: 1px solid rgba(49, 51, 63, 0.2);
                        background-color: white;
                        color: rgb(49, 51, 63);
                    "
                    onmouseover="this.style.borderColor='rgb(255, 75, 75)'; this.style.color='rgb(255, 75, 75)';"
                    onmouseout="this.style.borderColor='rgba(49, 51, 63, 0.2)'; this.style.color='rgb(49, 51, 63)';"
                    >Copy Lineup</button>
                    </body>
                    </html>
                """, height=36, scrolling=False)

            # "Enter Lineup into Event" buttons (only when events panel is active)
            # Check if all seats are filled (cox is optional for entry)
            seats_filled = len([r for r in lineup if r is not None])
            # Don't show "Enter into" when in edit mode - use Save Changes instead
            if show_events_panel and has_events and seats_filled == num_seats and not st.session_state.get('editing_entry'):
                # Get lineup stats for eligibility check
                stats = get_lineup_stats(lineup, roster_manager)
                if stats:
                    avg_age, category, lineup_gender = stats

                    # Get events to check eligibility against
                    available_events = roster_manager.regatta_events.get(selected_regatta, [])

                    # Find eligible events
                    eligible_events = []
                    for event in available_events:
                        # Only check targeted events if filter is on (will be checked when rendering)
                        if is_lineup_eligible_for_event(lineup_gender, avg_age, boat_class, event):
                            eligible_events.append(event)

                    # Show buttons for eligible events
                    # Prioritize: 1) Events checked for autofill, 2) Targeted events, 3) Other eligible
                    targeted_filter = st.session_state.get("targeted_events_filter", True)
                    autofill_checked = st.session_state.get('autofill_checked_events', set())

                    # Always include events checked for autofill (user explicitly selected them)
                    checked_events = [e for e in eligible_events if e.event_number in autofill_checked]
                    if targeted_filter:
                        other_events = [e for e in eligible_events if e.include and e.event_number not in autofill_checked]
                    else:
                        other_events = [e for e in eligible_events if e.event_number not in autofill_checked]

                    # Checked events first, then others
                    events_to_show = checked_events + other_events

                    if events_to_show:
                        st.markdown("**Enter into:**")
                        for event in events_to_show[:3]:  # Limit to 3 buttons to avoid clutter
                            # Create unique key for this button
                            btn_key = f"enter_{key}_{event.event_number}"
                            # Shorten event name for button
                            short_name = event.event_name[:20] + "..." if len(event.event_name) > 20 else event.event_name

                            # Check if this exact lineup is already entered in this event
                            current_rowers_set = set(rower_names_list)
                            already_entered = False
                            for existing in st.session_state.event_entries:
                                if (existing['event_number'] == event.event_number
                                    and existing['regatta'] == event.regatta
                                    and existing['day'] == event.day
                                    and set(existing.get('rowers', [])) == current_rowers_set):
                                    already_entered = True
                                    break

                            if already_entered:
                                st.button(f"✓ Entered: {short_name}", key=btn_key, use_container_width=True, disabled=True)
                            elif st.button(f"{format_event_time(event.event_time)} {short_name}", key=btn_key, use_container_width=True):
                                # Create entry
                                from datetime import datetime
                                # Count existing entries for this event to get entry number
                                existing_entries = [e for e in st.session_state.event_entries
                                                   if e['event_number'] == event.event_number
                                                   and e['regatta'] == event.regatta
                                                   and e['day'] == event.day]
                                entry_number = len(existing_entries) + 1

                                new_entry = {
                                    'regatta': event.regatta,
                                    'day': event.day,
                                    'event_number': event.event_number,
                                    'event_name': event.event_name,
                                    'event_time': event.event_time,
                                    'entry_number': entry_number,
                                    'boat_class': boat_class,
                                    'category': f"{lineup_gender} {category}",
                                    'avg_age': round(avg_age, 1),
                                    'rowers': rower_names_list.copy(),
                                    'boat': st.session_state.get(boat_key, ''),  # Club boat assignment
                                    'timestamp': datetime.now().isoformat()
                                }
                                # Save to Google Sheets
                                if save_entry_to_gsheet(new_entry):
                                    st.toast(f"Entry saved to Google Sheets")
                                st.session_state.event_entries.append(new_entry)
                                st.rerun()

    # Event panel (right column when shown)
    if event_col:
        with event_col:
            st.markdown("**Events**")

            # Filter checkbox
            show_targeted_only = st.checkbox("Targeted Only", value=True, key="targeted_events_filter")

            # Get events for selected regatta/day
            events = roster_manager.regatta_events.get(selected_regatta, [])

            # Filter if checkbox checked
            if show_targeted_only:
                events = [e for e in events if e.include]

            # Helper to normalize day strings for comparison (remove leading zeros)
            def normalize_day(day_str: str) -> str:
                """Normalize day string: 'Sunday, March 08, 2026' -> 'sunday, march 8, 2026'"""
                import re
                # Remove leading zeros from day numbers and lowercase
                normalized = re.sub(r'\b0(\d)', r'\1', day_str.lower())
                return normalized

            # Helper to check if entry matches event (handles format differences)
            def entry_matches_event(entry: dict, event) -> bool:
                if entry['event_number'] != event.event_number:
                    return False
                # Normalize day comparison
                if normalize_day(entry['day']) != normalize_day(event.day):
                    return False
                # Check regatta - exact match or partial match
                entry_regatta = entry['regatta'].lower()
                event_regatta = event.regatta.lower()
                if entry_regatta == event_regatta:
                    return True
                # Partial match - one contains the other
                if entry_regatta in event_regatta or event_regatta in entry_regatta:
                    return True
                return False

            # Helper to check if entry is missing cox for a coxed boat
            def is_missing_cox(entry: dict) -> bool:
                boat = entry.get('boat_class', '')
                if '+' not in boat:
                    return False  # Not a coxed boat
                rowers = entry.get('rowers', [])
                # 4+ needs 5 people (4 rowers + cox), 8+ needs 9 people (8 rowers + cox)
                expected_count = {'4+': 5, '8+': 9}.get(boat, 0)
                return len(rowers) < expected_count

            # Helper to check if entry is missing boat assignment
            def is_missing_boat(entry: dict) -> bool:
                boat = entry.get('boat', '')
                return not boat or boat.strip() == ''

            # Display event list with entry indicators and autofill checkboxes
            for event in events:
                priority_marker = "⭐ " if event.priority else ""

                # Check for entries in this event
                event_entries = [e for e in st.session_state.event_entries
                                if entry_matches_event(e, event)]

                # Build event label
                event_time_str = format_event_time(event.event_time)

                if event_entries:
                    # Show event with entry count as expander
                    entry_count = len(event_entries)
                    # Check if any entry is missing cox or boat
                    any_missing_cox = any(is_missing_cox(e) for e in event_entries)
                    any_missing_boat = any(is_missing_boat(e) for e in event_entries)
                    cox_warning = "📣 " if any_missing_cox else ""
                    boat_warning = "🚣 " if any_missing_boat else ""

                    # Checkbox for autofill (inline before expander)
                    is_checked = event.event_number in st.session_state.autofill_checked_events
                    chk_col, exp_col = st.columns([0.15, 9.85], gap="small")
                    with chk_col:
                        event_check = st.checkbox("", value=is_checked,
                                                   key=f"autofill_event_check_{event.event_number}")
                    with exp_col:
                        with st.expander(f"{event_time_str} {priority_marker}{cox_warning}{boat_warning}{event.event_name} [{entry_count}]"):
                            for entry_idx, entry in enumerate(event_entries):
                                avg_age_display = entry.get('avg_age', '-')
                                # Show warnings if this entry is missing cox or boat
                                entry_warnings = []
                                if is_missing_cox(entry):
                                    entry_warnings.append("📣 Needs Cox")
                                if is_missing_boat(entry):
                                    entry_warnings.append("🚣 Needs Boat")
                                warning_str = " ".join(entry_warnings)
                                boat_display = f" | Boat: {entry.get('boat')}" if entry.get('boat') else ""
                                st.markdown(f"**Entry {entry['entry_number']}** - {entry['boat_class']} {entry['category']} (Avg: {avg_age_display}){boat_display} {warning_str}")
                                # Show rowers
                                rower_list = ", ".join(entry['rowers'])
                                st.caption(rower_list)
                                # Edit and Remove buttons
                                btn_cols = st.columns(2)
                                with btn_cols[0]:
                                    if st.button("✏️", key=f"edit_{event.event_number}_{entry['entry_number']}_{entry_idx}", help="Edit in Lineup A"):
                                        # Load entry into Lineup A for editing
                                        entry_boat = entry.get('boat_class', '4+')
                                        entry_rowers = entry.get('rowers', [])
                                        is_coxed = '+' in entry_boat

                                        # Use pending flag for boat class (will be processed before widget renders on rerun)
                                        st.session_state.pending_boat_class = entry_boat

                                        # For coxed boats, last rower is cox
                                        boat_seats = {'1x': 1, '2x': 2, '2-': 2, '4x': 4, '4+': 4, '4-': 4, '8+': 8}
                                        expected_seats = boat_seats.get(entry_boat, 4)

                                        if is_coxed and entry_rowers:
                                            if len(entry_rowers) > expected_seats:
                                                # Has cox - split rowers and cox
                                                seat_rowers = entry_rowers[:expected_seats]
                                                cox = entry_rowers[expected_seats]
                                            else:
                                                # No cox assigned
                                                seat_rowers = entry_rowers
                                                cox = None
                                            st.session_state.lineup_a = seat_rowers + [None] * (expected_seats - len(seat_rowers))
                                            st.session_state.cox_a = cox
                                        else:
                                            # Non-coxed boat
                                            st.session_state.lineup_a = entry_rowers + [None] * (expected_seats - len(entry_rowers))
                                            st.session_state.cox_a = None

                                        # Restore club boat assignment (use pending flag for widget sync)
                                        st.session_state.pending_boat_a = entry.get('boat', None)

                                        # Store original entry for edit mode
                                        st.session_state.editing_entry = entry.copy()
                                        st.session_state.editing_event = event

                                        # Switch to lineup view
                                        st.session_state.view_mode = 'lineup'
                                        st.toast(f"Editing Entry {entry['entry_number']} for {event.event_name}")
                                        st.rerun()
                                with btn_cols[1]:
                                    if st.button("🗑️", key=f"remove_{event.event_number}_{entry['entry_number']}_{entry_idx}", help="Remove entry"):
                                        # Delete from Google Sheets
                                        if delete_entry_from_gsheet(entry):
                                            st.toast("Entry removed from Google Sheets")
                                        st.session_state.event_entries.remove(entry)
                                        # Clear edit mode if we just deleted the entry being edited
                                        editing = st.session_state.get('editing_entry')
                                        if editing and editing.get('entry_number') == entry.get('entry_number') and editing.get('event_number') == entry.get('event_number'):
                                            del st.session_state['editing_entry']
                                            if 'editing_event' in st.session_state:
                                                del st.session_state['editing_event']
                                        st.rerun()

                    # Handle checkbox state change (for events with entries)
                    if event_check and event.event_number not in st.session_state.autofill_checked_events:
                        st.session_state.autofill_checked_events.add(event.event_number)
                        st.rerun()
                    elif not event_check and event.event_number in st.session_state.autofill_checked_events:
                        st.session_state.autofill_checked_events.discard(event.event_number)
                        st.rerun()

                else:
                    # No entries - show checkbox with event name inline
                    is_checked = event.event_number in st.session_state.autofill_checked_events
                    event_label = f"{event_time_str} {priority_marker}{event.event_name}"
                    event_check = st.checkbox(
                        event_label,
                        value=is_checked,
                        key=f"autofill_event_check_{event.event_number}",
                    )
                    # Handle checkbox state change
                    if event_check and event.event_number not in st.session_state.autofill_checked_events:
                        st.session_state.autofill_checked_events.add(event.event_number)
                        st.rerun()
                    elif not event_check and event.event_number in st.session_state.autofill_checked_events:
                        st.session_state.autofill_checked_events.discard(event.event_number)
                        st.rerun()

            # Export button at bottom of event panel
            if st.session_state.event_entries:
                st.divider()
                # Create CSV data
                csv_lines = ["Regatta,Day,Event Number,Event Name,Event Time,Entry Number,Boat Class,Category,Avg Age,Lineup,Boat,Timestamp"]
                for entry in st.session_state.event_entries:
                    lineup_str = format_lineup_string(entry['rowers'], entry['boat_class'])
                    # Escape commas in fields
                    csv_lines.append(",".join([
                        f'"{entry["regatta"]}"',
                        f'"{entry["day"]}"',
                        str(entry['event_number']),
                        f'"{entry["event_name"]}"',
                        f'"{entry["event_time"]}"',
                        str(entry['entry_number']),
                        entry['boat_class'],
                        f'"{entry["category"]}"',
                        str(entry.get('avg_age', '')),
                        f'"{lineup_str}"',
                        f'"{entry.get("boat", "")}"',
                        entry['timestamp']
                    ]))
                csv_data = "\n".join(csv_lines)
                st.download_button(
                    label="Export CSV",
                    data=csv_data,
                    file_name="event_entries.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    # Analysis Results
    st.divider()
    st.subheader("Analysis Results")

    if st.session_state.get('analyze_clicked', False):
        st.session_state.analyze_clicked = False

        # Check for partially filled lineups
        partial_lineups = []
        for lineup_id, key in [("A", "lineup_a"), ("B", "lineup_b"), ("C", "lineup_c")]:
            lineup = st.session_state[key]
            filled_seats = [r for r in lineup if r is not None]
            # Partially filled = has some rowers but not all seats filled
            if filled_seats and len(filled_seats) < len(lineup):
                partial_lineups.append(lineup_id)

        if partial_lineups:
            lineup_names = ", ".join([f"Lineup {lid}" for lid in partial_lineups])
            st.error(f"{lineup_names} {'is' if len(partial_lineups) == 1 else 'are'} partially filled. Please fill all seats or clear the lineup before analyzing.")
            # Skip analysis when there are partial lineups
            st.session_state.analysis_skipped = True
        else:
            st.session_state.analysis_skipped = False

        results = []

        # Helper to determine lineup gender from rowers
        def get_lineup_gender(rower_names: List[str]) -> str:
            """Determine gender for lineup - 'M', 'W', or 'Mix'"""
            genders = set()
            for name in rower_names:
                rower = roster_manager.get_rower(name)
                if rower:
                    g = rower.gender.upper() if rower.gender else 'M'
                    # Convert 'F' (Female) to 'W' (Women) for rowing convention
                    if g == 'F':
                        g = 'W'
                    genders.add(g)
            if len(genders) == 1:
                return genders.pop()
            elif len(genders) > 1:
                return 'Mix'
            return 'M'  # Default

        # Only run analysis if no partial lineups were detected
        if not st.session_state.get('analysis_skipped', False):
            for lineup_id, key in [("A", "lineup_a"), ("B", "lineup_b"), ("C", "lineup_c")]:
                lineup = st.session_state[key]
                rower_names_in_lineup = [r for r in lineup if r is not None]

                if rower_names_in_lineup:
                    result = analyzer.analyze_lineup(rower_names_in_lineup, target_distance, boat_class, calc_method, pace_predictor)
                    result['lineup_id'] = lineup_id
                    # Store rower names for display formatting
                    result['rower_names'] = rower_names_in_lineup
                    result['lineup_display'] = format_lineup_display(lineup_id, rower_names_in_lineup, boat_class)
                    # Store lineup gender for erg-to-water conversion
                    result['lineup_gender'] = get_lineup_gender(rower_names_in_lineup)
                    # Store per-lineup tech efficiency (from widget value captured earlier)
                    result['tech_efficiency'] = lineup_tech_efficiencies.get(key, 1.05)
                    results.append(result)

        if results:
            # Sort by adjusted time
            def sort_key(r):
                if 'error' in r and 'avg_watts' not in r:
                    return float('inf')
                return r.get('adjusted_time', float('inf'))

            results.sort(key=sort_key)

            # Create results dataframe
            table_data = []
            for place, result in enumerate(results, 1):
                if 'error' in result and 'avg_watts' not in result:
                    table_data.append({
                        'Place': '-',
                        'Lineup': result.get('lineup_display', result['lineup_id']),
                        'Age': '-',
                        'Split': 'ERROR',
                        'Raw Time': result.get('error', 'Unknown'),
                        'Handicap': '-',
                        'Adjusted': '-',
                        'Avg W': '-',
                        'Port W': '-',
                        'Stbd W': '-',
                        '% Diff': '-'
                    })
                else:
                    port_w = f"{result['port_watts_total']:.0f}" if result.get('port_watts_total') else "-"
                    stbd_w = f"{result['starboard_watts_total']:.0f}" if result.get('starboard_watts_total') else "-"
                    pct_diff = f"{result['side_balance_pct']:.1f}%" if result.get('side_balance_pct') else "-"

                    # Get times - apply erg-to-water conversion if enabled
                    raw_time = result['raw_time']
                    adjusted_time = result['adjusted_time']
                    split_500m = result['boat_split_500m']

                    if erg_to_water:
                        tech_eff = result.get('tech_efficiency', 1.05)
                        dist_ratio = race_distance / target_distance

                        # Apply erg-to-water conversion with distance ratio
                        raw_time = apply_erg_to_water(result['raw_time'], boat_class, tech_eff) * dist_ratio
                        adjusted_time = apply_erg_to_water(result['adjusted_time'], boat_class, tech_eff) * dist_ratio
                        split_500m = apply_erg_to_water(result['boat_split_500m'], boat_class, tech_eff)

                    # Format age with masters category
                    avg_age = result.get('avg_age', 0)
                    masters_cat = get_masters_category(avg_age)
                    age_display = f"{avg_age:.1f} ({masters_cat})"

                    table_data.append({
                        'Place': str(place),
                        'Lineup': result.get('lineup_display', result['lineup_id']),
                        'Age': age_display,
                        'Split': format_split(split_500m),
                        'Raw Time': format_time(raw_time),
                        'Handicap': f"-{result['handicap_seconds']:.1f}s",
                        'Adjusted': format_time(adjusted_time),
                        'Avg W': f"{result['avg_watts']:.0f}",
                        'Port W': port_w,
                        'Stbd W': stbd_w,
                        '% Diff': pct_diff
                    })

            df = pd.DataFrame(table_data)

            # Show indicator when erg-to-water conversion is active
            if erg_to_water:
                # Check if lineups have different tech efficiencies
                lineup_tech_effs = set(r.get('tech_efficiency', 1.05) for r in results)
                tech_eff_value = list(lineup_tech_effs)[0] if len(lineup_tech_effs) == 1 else 1.05
                tech_eff_str = f"{tech_eff_value:.2f}" if len(lineup_tech_effs) <= 1 else "varies"
                boat_factor = get_boat_factor(boat_class)
                caption = f"*On-Water Projection Mode* | {boat_class} Factor: {boat_factor:.2f} | Tech Eff: {tech_eff_str}"
                if race_distance != target_distance:
                    caption += f" | Race: {race_distance}m (from {target_distance}m)"
                st.caption(caption)

            st.dataframe(df, use_container_width=True, hide_index=True)

            # Show detailed projections for each lineup
            with st.expander("View Detailed Projections"):
                for result in results:
                    if 'projections' in result:
                        st.markdown(f"**Lineup {result.get('lineup_display', result['lineup_id'])}**")
                        proj_data = []

                        proj_tech_eff = result.get('tech_efficiency', 1.05)

                        for proj in result['projections']:
                            if 'error' not in proj:
                                # Format source distance - handle short distances
                                src_dist = proj.get('source_distance', 0)
                                if src_dist >= 1000:
                                    src_str = f"{src_dist//1000}K"
                                else:
                                    src_str = f"{src_dist}m"

                                # Format prediction method indicator
                                method = proj.get('projection_method', 'pauls_law')
                                power_law_points = proj.get('power_law_points')
                                if method == 'actual':
                                    method_str = "Actual"
                                elif method == 'power_law' and power_law_points:
                                    # Format the two distances used
                                    def fmt_dist(d):
                                        return f"{d//1000}K" if d >= 1000 else f"{d}m"
                                    dists = sorted([p[0] for p in power_law_points])
                                    method_str = f"Power({fmt_dist(dists[0])},{fmt_dist(dists[1])})"
                                elif method == 'power_law':
                                    method_str = "Power"
                                else:
                                    method_str = "Paul's"

                                # Get splits - apply erg-to-water conversion if enabled
                                source_split = proj.get('source_split', 0)
                                projected_split = proj.get('projected_split', 0)

                                if erg_to_water and source_split > 0:
                                    source_split = apply_erg_to_water(source_split, boat_class, proj_tech_eff)
                                if erg_to_water and projected_split > 0:
                                    projected_split = apply_erg_to_water(projected_split, boat_class, proj_tech_eff)

                                proj_data.append({
                                    'Seat': proj.get('seat', '-'),
                                    'Side': proj.get('seat_side', '-'),
                                    'Rower': proj['rower'],
                                    'Age': proj.get('age', '-'),
                                    'Pref': proj.get('side', '-'),
                                    'Source': src_str,
                                    'Source Split': format_split(source_split),
                                    'Projected Split': format_split(projected_split),
                                    'Watts': f"{proj.get('projected_watts', 0):.0f}",
                                    'Method': method_str
                                })
                            else:
                                proj_data.append({
                                    'Seat': proj.get('seat', '-'),
                                    'Side': '-',
                                    'Rower': proj['rower'],
                                    'Age': proj.get('age', '-'),
                                    'Pref': proj.get('side', '-'),
                                    'Source': '-',
                                    'Source Split': '-',
                                    'Projected Split': '-',
                                    'Watts': proj.get('error', 'Error'),
                                    'Method': '-'
                                })

                        if proj_data:
                            st.table(pd.DataFrame(proj_data))

            # Export options
            st.divider()
            export_col1, export_col2 = st.columns([1, 5])

            with export_col1:
                # Excel download
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    # Summary sheet
                    df.to_excel(writer, sheet_name='Summary', index=False)

                    # Detailed projections for each lineup
                    for result in results:
                        if 'projections' in result:
                            lineup_id = result['lineup_id']
                            lineup_display = result.get('lineup_display', lineup_id)
                            avg_age = result.get('avg_age', 0)

                            export_tech_eff = result.get('tech_efficiency', 1.05)

                            proj_rows = []
                            for proj in result['projections']:
                                if 'error' not in proj:
                                    # Get splits - apply erg-to-water if enabled
                                    source_split = proj.get('source_split', 0)
                                    projected_split = proj.get('projected_split', 0)

                                    if erg_to_water:
                                        if source_split > 0:
                                            source_split = apply_erg_to_water(source_split, boat_class, export_tech_eff)
                                        if projected_split > 0:
                                            projected_split = apply_erg_to_water(projected_split, boat_class, export_tech_eff)

                                    # Format method string
                                    method = proj.get('projection_method', 'pauls_law')
                                    power_law_points = proj.get('power_law_points')
                                    if method == 'actual':
                                        method_str = "Actual"
                                    elif method == 'power_law' and power_law_points:
                                        def fmt_dist(d):
                                            return f"{d//1000}K" if d >= 1000 else f"{d}m"
                                        dists = sorted([p[0] for p in power_law_points])
                                        method_str = f"Power({fmt_dist(dists[0])},{fmt_dist(dists[1])})"
                                    elif method == 'power_law':
                                        method_str = "Power"
                                    else:
                                        method_str = "Paul's"

                                    proj_rows.append({
                                        'Seat': proj.get('seat', '-'),
                                        'Rower': proj['rower'],
                                        'Age': proj.get('age', '-'),
                                        'Source Split': format_split(source_split),
                                        'Projected Split': format_split(projected_split),
                                        'Watts': proj.get('projected_watts', 0),
                                        'Method': method_str
                                    })
                            if proj_rows:
                                # Create DataFrame with average age header
                                proj_df = pd.DataFrame(proj_rows)
                                # Add lineup name, average age, Masters category, and analysis settings as first row
                                masters_cat = get_masters_category(avg_age)
                                calc_display = "Watts" if calc_method == "watts" else "Split"
                                pred_display = "Power Law" if pace_predictor == "power_law" else "Paul's Law"
                                header_info = f'{lineup_display} | Avg Age: {avg_age:.1f} | Cat: {masters_cat} | Calc: {calc_display} | Pred: {pred_display}'
                                if erg_to_water:
                                    boat_factor = get_boat_factor(boat_class)
                                    header_info += f' | On-Water ({boat_factor:.2f})'
                                else:
                                    header_info += ' | Erg (Raw)'
                                header_df = pd.DataFrame([{'Seat': header_info, 'Rower': '', 'Age': '', 'Source Split': '', 'Projected Split': '', 'Watts': '', 'Method': ''}])
                                combined_df = pd.concat([header_df, proj_df], ignore_index=True)
                                combined_df.to_excel(
                                    writer, sheet_name=f'Lineup {lineup_id}', index=False
                                )

                excel_buffer.seek(0)

                # Copy to clipboard - create tab-separated text for easy pasting
                # Add analysis settings header
                calc_display = "Watts" if calc_method == "watts" else "Split"
                pred_display = "Power Law" if pace_predictor == "power_law" else "Paul's Law"
                header_line = f"LINEUP ANALYSIS | Calc: {calc_display} | Predictor: {pred_display}"
                if erg_to_water:
                    boat_factor = get_boat_factor(boat_class)
                    header_line += f" | {boat_class} Factor: {boat_factor:.2f}"
                else:
                    header_line += " | Erg (Raw)"
                clipboard_text = header_line + "\n\n" + df.to_csv(sep='\t', index=False)

                # Add detailed projections
                for result in results:
                    if 'projections' in result:
                        avg_age = result.get('avg_age', 0)
                        masters_cat = get_masters_category(avg_age)
                        clip_tech_eff = result.get('tech_efficiency', 1.05)

                        lineup_display = result.get('lineup_display', result['lineup_id'])
                        header_info = f"{lineup_display} Details (Avg Age: {avg_age:.1f} | Cat: {masters_cat})"
                        if erg_to_water:
                            lineup_factor = get_boat_factor(boat_class)
                            header_info += f" [On-Water: {lineup_factor:.2f} x {clip_tech_eff:.2f}]"
                        else:
                            header_info += " [Erg]"
                        clipboard_text += f"\n\n{header_info}:\n"

                        proj_rows = []
                        for proj in result['projections']:
                            if 'error' not in proj:
                                # Get split - apply erg-to-water if enabled
                                projected_split = proj.get('projected_split', 0)
                                if erg_to_water and projected_split > 0:
                                    projected_split = apply_erg_to_water(projected_split, boat_class, clip_tech_eff)

                                # Format method string
                                method = proj.get('projection_method', 'pauls_law')
                                power_law_points = proj.get('power_law_points')
                                if method == 'actual':
                                    method_str = "Actual"
                                elif method == 'power_law' and power_law_points:
                                    def fmt_dist(d):
                                        return f"{d//1000}K" if d >= 1000 else f"{d}m"
                                    dists = sorted([p[0] for p in power_law_points])
                                    method_str = f"Power({fmt_dist(dists[0])},{fmt_dist(dists[1])})"
                                elif method == 'power_law':
                                    method_str = "Power"
                                else:
                                    method_str = "Paul's"

                                proj_rows.append({
                                    'Seat': proj.get('seat', '-'),
                                    'Rower': proj['rower'],
                                    'Projected Split': format_split(projected_split),
                                    'Watts': f"{proj.get('projected_watts', 0):.0f}",
                                    'Method': method_str
                                })
                        if proj_rows:
                            clipboard_text += pd.DataFrame(proj_rows).to_csv(sep='\t', index=False)

                # Encode text as base64 to avoid any escaping issues
                b64_text = base64.b64encode(clipboard_text.encode('utf-8')).decode('ascii')

                # Copy to clipboard using components.html (executes JavaScript properly)
                components.html(f"""
                    <html>
                    <head><style>
                        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                        html, body {{ height: 100%; overflow: hidden; margin: 0; padding: 0; }}
                        body {{ padding-left: 2px; padding-right: 14px; }}
                        button {{ display: block; width: 100%; }}
                    </style></head>
                    <body>
                    <button id="copyBtn" data-text="{b64_text}" onclick="
                        var text = atob(this.getAttribute('data-text'));
                        navigator.clipboard.writeText(text).then(function() {{
                            document.getElementById('copyBtn').innerText = 'Copied!';
                            document.getElementById('copyBtn').style.background = '#28a745';
                            document.getElementById('copyBtn').style.color = 'white';
                            document.getElementById('copyBtn').style.borderColor = '#28a745';
                            setTimeout(function() {{
                                document.getElementById('copyBtn').innerText = 'Copy to Clipboard';
                                document.getElementById('copyBtn').style.background = 'white';
                                document.getElementById('copyBtn').style.color = 'rgb(49, 51, 63)';
                                document.getElementById('copyBtn').style.borderColor = 'rgba(49, 51, 63, 0.2)';
                            }}, 2000);
                        }}).catch(function(err) {{
                            document.getElementById('copyBtn').innerText = 'Failed';
                            document.getElementById('copyBtn').style.background = '#dc3545';
                        }});
                    " style="
                        padding: 0.25rem 0.75rem;
                        height: 100%;
                        font-family: 'Source Sans Pro', sans-serif;
                        font-size: 16px;
                        font-weight: 400;
                        line-height: 1.6;
                        cursor: pointer;
                        border-radius: 8px;
                        border: 1px solid rgba(49, 51, 63, 0.2);
                        background-color: white;
                        color: rgb(49, 51, 63);
                    "
                    onmouseover="this.style.borderColor='rgb(255, 75, 75)'; this.style.color='rgb(255, 75, 75)';"
                    onmouseout="this.style.borderColor='rgba(49, 51, 63, 0.2)'; this.style.color='rgb(49, 51, 63)';"
                    >Copy to Clipboard</button>
                    </body>
                    </html>
                """, height=36, scrolling=False)

                # Update filename to indicate on-water mode
                filename_suffix = "_onwater" if erg_to_water else ""
                st.download_button(
                    label="Download Excel",
                    data=excel_buffer,
                    file_name=f"lineup_analysis_{boat_class}_{target_distance}m{filename_suffix}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

        else:
            # Only show this message if analysis wasn't skipped due to partial lineups
            if not st.session_state.get('analysis_skipped', False):
                st.info("No lineups to analyze. Add rowers to at least one lineup.")
    else:
        st.info("Click 'Analyze All Lineups' in the sidebar to see results.")


if __name__ == "__main__":
    main()
