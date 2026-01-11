#!/usr/bin/env python3
"""
Rowing Lineup Analyzer - Interactive TUI
Analyzes potential boat lineups using Paul's Law and boat physics
"""

import pandas as pd
import numpy as np
import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import re
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header, Footer, Static, Button, Label, Input,
    Select, DataTable, Rule
)
from textual.binding import Binding
from textual.reactive import reactive
from textual import events
from textual.message import Message
from rich.text import Text
from rich.table import Table
from rich.panel import Panel


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

    def __repr__(self):
        mins = int(self.split_500m // 60)
        secs = self.split_500m % 60
        return f"{mins}:{secs:04.1f} ({self.distance}m)"

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
    pref_order: str = ""  # Raw preference order string from spreadsheet
    scores: Dict[int, Score] = field(default_factory=dict)
    regatta_signups: Dict[str, bool] = field(default_factory=dict)

    def add_score(self, score: Score):
        """Add score, keeping best time per distance"""
        if score.distance not in self.scores:
            self.scores[score.distance] = score
        elif score.time_seconds < self.scores[score.distance].time_seconds:
            self.scores[score.distance] = score

    def get_closest_score(self, target_distance: int) -> Optional[Score]:
        """Get the score closest to target distance for projection"""
        if not self.scores:
            return None
        closest_dist = min(self.scores.keys(),
                           key=lambda d: abs(d - target_distance))
        return self.scores[closest_dist]

    def side_preference_str(self) -> str:
        """Return side preference as a string like 'P', 'S', 'PS', 'X', etc."""
        parts = []
        if self.side_port:
            parts.append('P')
        if self.side_starboard:
            parts.append('S')
        if self.side_coxswain:
            parts.append('X')
        return ''.join(parts) if parts else '-'

    def primary_side(self) -> str:
        """Return primary side based on preference order or capabilities.
        Ignores X (sculling) - only considers P (port) and S (starboard)."""
        # Check pref_order first - first P or S in the string is preferred
        # Skip X entirely as it indicates sculling, not sweep side preference
        if self.pref_order:
            for char in self.pref_order.upper():
                if char == 'X':
                    continue  # Ignore sculling indicator
                if char == 'P' and self.side_port:
                    return 'P'
                elif char == 'S' and self.side_starboard:
                    return 'S'
        # Fall back to capabilities (ignoring sculling)
        if self.side_port and not self.side_starboard:
            return 'P'
        elif self.side_starboard and not self.side_port:
            return 'S'
        # Both sides - no clear preference
        elif self.side_port and self.side_starboard:
            return 'P'  # Default to port if both with no stated preference
        return 'S'

    def preferred_side(self) -> str:
        """Return primary side preference for power balance calculation"""
        if self.side_port and not self.side_starboard:
            return 'P'
        elif self.side_starboard and not self.side_port:
            return 'S'
        return 'Both'

    def is_attending(self, regatta_name: str) -> bool:
        return self.regatta_signups.get(regatta_name, False)

    def scores_summary(self) -> str:
        """Return formatted summary of available scores"""
        if not self.scores:
            return "No scores"
        return ", ".join([f"{d//1000}K" for d in sorted(self.scores.keys())])


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

        # If already numeric, return as-is
        if isinstance(time_val, (int, float)):
            return float(time_val)

        time_str = str(time_val).strip()
        if not time_str or time_str.lower() == 'nan':
            return None

        # Try parsing as float first
        try:
            return float(time_str)
        except ValueError:
            pass

        # Pattern: MM:SS.d or M:SS.d
        patterns = [
            r'^(\d{1,2}):(\d{2})\.(\d+)$',  # 3:18.5 or 17:39.2
            r'^(\d{1,2}):(\d{2}\.\d+)$',     # 3:18.5
            r'^(\d{1,2}):(\d{2})$',          # 3:18
            r'^0?(\d{1,2}):(\d{2}):(\d{2})$', # 00:03:18 (hour:min:sec)
        ]

        for pattern in patterns:
            match = re.match(pattern, time_str)
            if match:
                groups = match.groups()
                if len(groups) == 3 and ':' in time_str and time_str.count(':') == 1:
                    # MM:SS.d format
                    mins = int(groups[0])
                    secs = int(groups[1])
                    decimal = float(f"0.{groups[2]}")
                    return mins * 60 + secs + decimal
                elif len(groups) == 3:
                    # H:MM:SS format
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

        # Handle formats like "01:45.9" or "1:45.9"
        match = re.match(r'^0?(\d{1,2}):(\d{2})\.?(\d*)$', split_str)
        if match:
            mins = int(match.group(1))
            secs = int(match.group(2))
            decimal = float(f"0.{match.group(3)}") if match.group(3) else 0
            return mins * 60 + secs + decimal

        # Try as float (already in seconds)
        try:
            return float(split_str)
        except ValueError:
            pass

        return None


# =============================================================================
# PHYSICS ENGINE
# =============================================================================

class PhysicsEngine:
    """Rowing physics calculations using Paul's Law"""

    @staticmethod
    def calculate_500m_split(total_time: float, distance: int) -> float:
        """Calculate 500m split from total time and distance"""
        return (total_time / distance) * 500

    @staticmethod
    def split_to_watts(split_seconds: float) -> float:
        """Convert 500m split to watts using Concept2 formula"""
        # Watts = 2.80 / (pace_per_500 / 500)^3
        pace_ratio = split_seconds / 500
        return 2.80 / (pace_ratio ** 3)

    @staticmethod
    def watts_to_split(watts: float) -> float:
        """Convert watts back to 500m split"""
        if watts <= 0:
            return float('inf')
        return 500 * (2.80 / watts) ** (1/3)

    @staticmethod
    def pauls_law_projection(known_split: float, known_dist: int, target_dist: int) -> float:
        """
        Apply Paul's Law to project split at different distance.
        Paul's Law: Projected_Split = Known_Split + 5 * log2(Target_Dist / Known_Dist)
        """
        if known_dist == target_dist:
            return known_split
        if known_dist <= 0 or target_dist <= 0:
            return known_split

        distance_ratio = target_dist / known_dist
        split_adjustment = 5 * math.log2(distance_ratio)
        return known_split + split_adjustment

    @staticmethod
    def calculate_boat_split(watts_list: List[float]) -> float:
        """
        Calculate boat split by averaging watts (not splits), then converting back.
        This is the correct physics approach.
        """
        if not watts_list:
            return 0
        avg_watts = statistics.mean(watts_list)
        return PhysicsEngine.watts_to_split(avg_watts)


# =============================================================================
# DATA LOADING
# =============================================================================

class RosterManager:
    """Manages loading and accessing rower data from Excel"""

    # Boat class K factors for age handicap
    K_FACTORS = {
        '1x': 0.02,
        '2x': 0.02,
        '2-': 0.02,
        '4x': 0.02,
        '4+': 0.02,
        '4-': 0.02,
        '8+': 0.02
    }

    def __init__(self):
        self.rowers: Dict[str, Rower] = {}
        self.regattas: List[str] = []
        self.regatta_display_names: Dict[str, str] = {}  # col_name -> clean display name
        self.load_log: List[str] = []

    def log(self, msg: str):
        self.load_log.append(msg)

    def load_from_excel(self, filepath: str) -> bool:
        """Load all data from the Excel spreadsheet"""
        try:
            self.log(f"Loading from: {filepath}")
            xl = pd.ExcelFile(filepath)

            # Load roster first
            self._load_roster(xl)

            # Load regatta signups
            self._load_regatta_signups(xl)

            # Load score sheets
            self._load_score_sheets(xl)

            self.log(f"Loaded {len(self.rowers)} rowers total")
            return True

        except Exception as e:
            self.log(f"ERROR: {e}")
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
            # Combine First Name and Last Name
            first = str(row.get('First Name', '')).strip()
            last = str(row.get('Last Name', '')).strip()

            if not first or first == 'nan':
                continue

            name = f"{first} {last}".strip()

            # Get age
            age = 27  # Default
            if 'Age' in row and pd.notna(row['Age']):
                try:
                    age = int(row['Age'])
                except (ValueError, TypeError):
                    pass

            # Get gender
            gender = 'M'
            for col in ['GR', 'Gender', 'Sex']:
                if col in row and pd.notna(row[col]):
                    gender = str(row[col]).strip().upper()
                    if gender in ['MALE', 'M']:
                        gender = 'M'
                    elif gender in ['FEMALE', 'F']:
                        gender = 'F'
                    break

            # Get side preferences (P, S, X as boolean columns)
            side_port = bool(row.get('P', True)) if pd.notna(row.get('P')) else True
            side_starboard = bool(row.get('S', True)) if pd.notna(row.get('S')) else True
            side_cox = bool(row.get('X', False)) if pd.notna(row.get('X')) else False

            # Get preference order string (e.g., "PX....S", "SXP", "no pref")
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

        # Log ALL columns for debugging
        self.log(f"All columns in Regatta Sign Ups: {list(df.columns)}")

        name_col = 'Name'

        # Columns to completely ignore
        exclude_keywords = ['name', 'email', 'age', 'cat', 'category', 'notes',
                           'housing', 'feedback', 'request', 'saturday', 'sunday',
                           'friday', 'thursday', 'monday', 'tuesday', 'wednesday',
                           'unnamed']

        regatta_keywords = ['classic', 'regatta', 'regionals', 'head', 'sprints',
                           'rebellion', 'cascadia', 'rowfest', 'tail', 'bridge',
                           'opening day', 'lake', 'dog', 'charles']

        # Patterns to identify signup columns
        # Date pattern (e.g., "3/27/25", "10/15/2025")
        date_pattern = re.compile(r'\d{1,2}/\d{1,2}/\d{2,4}')
        # Location pattern (e.g., "San Diego, CA", "Seattle, WA", "Portland, OR")
        location_pattern = re.compile(r',\s*[A-Z]{2}\b')
        # Day pattern at start (e.g., "SUNDAY\n...", "SATURDAY\n...")
        day_pattern = re.compile(r'^(SUNDAY|SATURDAY|FRIDAY|THURSDAY|MONDAY|TUESDAY|WEDNESDAY)\n', re.IGNORECASE)

        # First pass: find all potential regatta columns grouped by base name
        regatta_candidates = {}  # base_name -> list of (col_name, is_signup_col, has_yes)
        for col in df.columns:
            col_str = str(col).lower()

            # Skip excluded columns (but NOT columns with day\n prefix - those are signup columns)
            if any(kw in col_str for kw in exclude_keywords):
                # Exception: if it starts with a day name, it's a signup column, don't skip
                if not day_pattern.match(str(col)):
                    continue
            if col_str.strip() == '' or col_str == 'nan':
                continue

            # Check if this looks like a regatta column
            if any(word in col_str for word in regatta_keywords):
                # Check if column header has a day prefix, date, OR location (indicates signup column)
                has_day_prefix = bool(day_pattern.match(str(col)))
                has_date = bool(date_pattern.search(str(col)))
                has_location = bool(location_pattern.search(str(col)))
                is_signup_col = has_day_prefix or has_date or has_location

                # Normalize the name (remove day prefix, date, location, .1 suffix, asterisk, date ranges)
                base_name = str(col)
                base_name = re.sub(r'^(SUNDAY|SATURDAY|FRIDAY|THURSDAY|MONDAY|TUESDAY|WEDNESDAY)\n', '', base_name, flags=re.IGNORECASE)
                base_name = re.sub(r'\.\d+$', '', base_name)
                base_name = re.sub(r',?\s*\d{1,2}/\d{1,2}/\d{2,4}(-\d{1,2}/\d{1,2}/\d{2,4})?', '', base_name)  # dates and date ranges
                base_name = re.sub(r'\n\d{1,2}/\d{1,2}/\d{2,4}(-\d{1,2}/\d{1,2}/\d{2,4})?', '', base_name)  # newline + dates
                base_name = re.sub(r',\s*[A-Z]{2}\b', '', base_name)  # Remove ", CA", ", WA", etc.
                base_name = re.sub(r'\*$', '', base_name).strip()

                # Check if this column has Yes values
                col_values = df[col].dropna().astype(str).str.lower().unique()
                has_yes = any('yes' in str(v) for v in col_values)

                if base_name not in regatta_candidates:
                    regatta_candidates[base_name] = []
                regatta_candidates[base_name].append((col, is_signup_col, has_yes))
                self.log(f"  Candidate: '{col}' -> base: '{base_name}', is_signup_col: {is_signup_col}, has_yes: {has_yes}")

        # Second pass: for each regatta, prefer column with date/location in header
        regatta_cols = []
        regatta_display_names = {}  # Map column name to clean display name
        for base_name, candidates in regatta_candidates.items():
            # Priority: columns with date/location in header (these have the Yes/No data)
            signup_cols = [(c, hy) for c, is_signup, hy in candidates if is_signup]
            if signup_cols:
                selected = signup_cols[0][0]
            else:
                # Fall back to columns with Yes values
                yes_cols = [c for c, is_signup, hy in candidates if hy]
                if yes_cols:
                    selected = yes_cols[0]
                else:
                    selected = candidates[0][0]

            regatta_cols.append(selected)
            regatta_display_names[selected] = base_name

        self.regattas = regatta_cols
        self.regatta_display_names = regatta_display_names
        self.log(f"Selected {len(regatta_cols)} regatta columns: {regatta_cols}")

        # First, log sample values for each regatta column to debug
        self.log("Sample values per regatta column:")
        for regatta in regatta_cols:
            unique_vals = df[regatta].dropna().unique()[:5]  # First 5 unique non-null values
            self.log(f"  {regatta}: {list(unique_vals)}")

        # Process each row
        signups_loaded = 0
        regatta_signup_counts = {r: 0 for r in regatta_cols}  # Initialize all to 0
        for _, row in df.iterrows():
            name = str(row.get(name_col, '')).strip()

            # Skip header/info rows
            if not name or name == 'nan' or name in ['Dates', 'Registration Opens', 'Regatta Master']:
                continue

            # Match to roster
            if name not in self.rowers:
                # Try matching by first+last name pattern
                continue

            rower = self.rowers[name]

            for regatta in regatta_cols:
                val = row.get(regatta, '')
                val_str = str(val).strip().lower() if pd.notna(val) else ''
                # More flexible detection - any non-empty value that's not 'no' or 'n'
                is_attending = bool(val_str) and val_str not in ['no', 'n', 'false', '0', '', 'nan']
                rower.regatta_signups[regatta] = is_attending
                if is_attending:
                    signups_loaded += 1
                    regatta_signup_counts[regatta] += 1

        self.log(f"Loaded {signups_loaded} regatta signups")
        for regatta in regatta_cols:
            self.log(f"  {regatta}: {regatta_signup_counts[regatta]} rowers")

    def _load_score_sheets(self, xl: pd.ExcelFile):
        """Load scores from score sheets (1K, 5K, etc.)"""
        score_sheets = []

        for sheet in xl.sheet_names:
            # Match patterns like "2025 1K Avgs", "2025 5Ks", etc.
            match = re.search(r'(\d+)\s*[Kk]', sheet)
            if match:
                distance = int(match.group(1)) * 1000
                score_sheets.append((sheet, distance))

        self.log(f"Found score sheets: {[(s, d) for s, d in score_sheets]}")

        for sheet_name, distance in score_sheets:
            self._load_scores_from_sheet(xl, sheet_name, distance)

    def _load_scores_from_sheet(self, xl: pd.ExcelFile, sheet_name: str, distance: int):
        """Load scores from a specific score sheet"""
        df = xl.parse(sheet_name)

        scores_loaded = 0
        for _, row in df.iterrows():
            # Get name
            name = str(row.get('Name', '')).strip()
            if not name or name == 'nan' or name not in self.rowers:
                continue

            rower = self.rowers[name]

            # Try to get time in seconds - prioritize 'Time in Seconds' column
            time_seconds = None

            # Check for 'Time in Seconds' column first (most reliable)
            for col in ['Time in Seconds', 'Time in Secs', 'Total Time']:
                if col in row.index and pd.notna(row[col]):
                    val = row[col]
                    if isinstance(val, (int, float)) and val > 0:
                        time_seconds = float(val)
                        break

            # If no time found, try parsing time string columns
            if not time_seconds:
                for col in ['1K Time', '5K Time', 'Time']:
                    if col in row.index and pd.notna(row[col]):
                        time_seconds = TimeParser.parse(row[col])
                        if time_seconds:
                            break

            # Get split if available (for 5K sheets with Avg. Split column)
            split_500m = None
            for col in ['Avg. Split', 'Avg Split', 'Split', '500m Split']:
                if col in row.index and pd.notna(row[col]):
                    split_500m = TimeParser.parse_split(row[col])
                    if split_500m and split_500m > 60:  # Valid split should be > 1 minute
                        break
                    else:
                        split_500m = None

            # Calculate split from total time if not found
            if time_seconds and not split_500m:
                split_500m = PhysicsEngine.calculate_500m_split(time_seconds, distance)
            elif split_500m and not time_seconds:
                time_seconds = (split_500m / 500) * distance

            # Skip if we couldn't get valid data
            if not time_seconds or not split_500m:
                continue

            # Sanity check: split should be reasonable (between 1:20 and 3:00 for most rowers)
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

        self.log(f"Loaded {scores_loaded} scores from '{sheet_name}'")

    def get_rower(self, name: str) -> Optional[Rower]:
        return self.rowers.get(name)

    def get_rowers_with_scores(self) -> List[str]:
        """Get list of rowers who have at least one score"""
        return sorted([name for name, r in self.rowers.items() if r.scores])

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
                       boat_class: str = '4+') -> Dict[str, Any]:
        """Analyze a lineup and return comprehensive results.

        For sweep boats (2-, 4+, 4-, 8+), seat positions alternate sides:
        - Even seats (0, 2, 4, 6) = Starboard (Stroke, 6, 4, 2)
        - Odd seats (1, 3, 5, 7) = Port (7, 5, 3, Bow)

        For sculling boats (1x, 2x, 4x), there is no port/starboard.
        """

        if not rower_names:
            return {'error': 'No rowers in lineup'}

        # Check if this is a sculling boat (no port/starboard)
        is_sculling = 'x' in boat_class.lower()

        rowers = []
        projections = []
        port_watts = []
        starboard_watts = []
        all_watts = []

        for seat_idx, name in enumerate(rower_names):
            rower = self.roster.get_rower(name)
            if not rower:
                projections.append({
                    'rower': name,
                    'seat': seat_idx,
                    'error': 'Rower not found'
                })
                continue

            rowers.append(rower)
            score = rower.get_closest_score(target_distance)

            if not score:
                projections.append({
                    'rower': name,
                    'seat': seat_idx,
                    'age': rower.age,
                    'side': rower.side_preference_str(),
                    'error': 'No scores available'
                })
                continue

            # Project split to target distance using Paul's Law
            projected_split = PhysicsEngine.pauls_law_projection(
                score.split_500m, score.distance, target_distance
            )
            projected_watts = PhysicsEngine.split_to_watts(projected_split)

            # Track side power based on SEAT POSITION (only for sweep boats)
            # Even seats (0, 2, 4, 6) = Starboard, Odd seats (1, 3, 5, 7) = Port
            seat_side = None
            if not is_sculling:
                seat_side = 'S' if seat_idx % 2 == 0 else 'P'
                if seat_side == 'P':
                    port_watts.append(projected_watts)
                else:
                    starboard_watts.append(projected_watts)

            all_watts.append(projected_watts)

            projections.append({
                'rower': name,
                'seat': seat_idx,
                'seat_side': seat_side,
                'age': rower.age,
                'side': rower.side_preference_str(),
                'source_distance': score.distance,
                'source_split': score.split_500m,
                'source_sheet': score.source_sheet,
                'projected_split': projected_split,
                'projected_watts': projected_watts
            })

        if not all_watts:
            return {
                'error': 'No valid projections',
                'projections': projections
            }

        # Calculate boat metrics
        avg_watts = statistics.mean(all_watts)
        boat_split = PhysicsEngine.watts_to_split(avg_watts)
        raw_time = (boat_split / 500) * target_distance

        # Side balance
        side_balance_pct = None
        if port_watts and starboard_watts:
            total_port = sum(port_watts)
            total_starboard = sum(starboard_watts)
            total_power = total_port + total_starboard
            side_balance_pct = abs(total_port - total_starboard) / total_power * 100

        # Age handicap: (avg_age - 27)² × K × distance_multiplier
        # USRowing uses distance/1000 as multiplier (1 for 1K, 2 for 2K, 5 for 5K)
        avg_age = statistics.mean([r.age for r in rowers]) if rowers else 27
        k_factor = self.K_FACTORS.get(boat_class, 0.02)
        distance_multiplier = target_distance / 1000
        handicap_seconds = ((avg_age - 27) ** 2) * k_factor * distance_multiplier
        adjusted_time = raw_time - handicap_seconds  # Subtract for faster effective time

        return {
            'rower_count': len(rowers),
            'projections': projections,
            'target_distance': target_distance,
            'boat_class': boat_class,
            'avg_watts': avg_watts,
            'boat_split_500m': boat_split,
            'raw_time': raw_time,
            'avg_age': avg_age,
            'handicap_seconds': handicap_seconds,
            'adjusted_time': adjusted_time,
            'side_balance_pct': side_balance_pct,
            'port_watts_total': sum(port_watts) if port_watts else None,
            'starboard_watts_total': sum(starboard_watts) if starboard_watts else None
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


# =============================================================================
# TUI COMPONENTS
# =============================================================================

class RowerListItem(Static):
    """Clickable rower item in roster list"""

    class Selected(Message):
        """Message when rower is selected"""
        def __init__(self, rower_name: str):
            self.rower_name = rower_name
            super().__init__()

    def __init__(self, name: str, rower: Rower):
        super().__init__()
        self.rower_name = name
        self.rower = rower

    def compose(self) -> ComposeResult:
        scores = self.rower.scores_summary()
        side = self.rower.side_preference_str()
        yield Static(f"{self.rower_name} | Age:{self.rower.age} | {side} | {scores}")

    def on_click(self, event: events.Click) -> None:
        self.post_message(self.Selected(self.rower_name))


class LineupRowerItem(Static):
    """Clickable rower/seat in a lineup"""

    class RemoveRower(Message):
        """Message when rower should be removed (right-click)"""
        def __init__(self, lineup_id: str, rower_name: str, seat: int):
            self.lineup_id = lineup_id
            self.rower_name = rower_name
            self.seat = seat
            super().__init__()

    class SeatClicked(Message):
        """Message when empty seat is left-clicked"""
        def __init__(self, lineup_id: str, seat: int):
            self.lineup_id = lineup_id
            self.seat = seat
            super().__init__()

    class RowerClicked(Message):
        """Message when filled seat is left-clicked (select or swap)"""
        def __init__(self, lineup_id: str, rower_name: str, seat: int):
            self.lineup_id = lineup_id
            self.rower_name = rower_name
            self.seat = seat
            super().__init__()

    def __init__(self, lineup_id: str, seat: int, seat_label: str, rower_name: Optional[str] = None):
        super().__init__()
        self.lineup_id = lineup_id
        self.seat = seat
        self.seat_label = seat_label
        self.rower_name = rower_name

    def compose(self) -> ComposeResult:
        if self.rower_name:
            yield Static(f"  {self.seat_label}: [bold]{self.rower_name}[/]")
        else:
            yield Static(f"  {self.seat_label}: [dim]-- click to place --[/dim]")

    def on_click(self, event: events.Click) -> None:
        if event.button == 3:  # Right-click
            if self.rower_name:
                self.post_message(self.RemoveRower(self.lineup_id, self.rower_name, self.seat))
        else:  # Left-click
            if self.rower_name:
                self.post_message(self.RowerClicked(self.lineup_id, self.rower_name, self.seat))
            else:
                self.post_message(self.SeatClicked(self.lineup_id, self.seat))


class LineupPanel(Static):
    """Panel for a single lineup (A, B, or C)"""

    SEAT_LABELS = {
        1: ["Stroke"],
        2: ["Stroke", "Bow"],
        4: ["Stroke", "3", "2", "Bow"],
        8: ["Stroke", "7", "6", "5", "4", "3", "2", "Bow"],
    }

    def __init__(self, lineup_id: str, title: str):
        super().__init__()
        self.lineup_id = lineup_id
        self.title = title
        self._rowers: List[Optional[str]] = []
        self._num_seats = 4  # Default for 4+

    def compose(self) -> ComposeResult:
        if self.title:
            yield Static(f"[bold cyan]{self.title}[/]", id=f"title-{self.lineup_id}")
        yield Vertical(id=f"seats-{self.lineup_id}")

    def on_mount(self) -> None:
        self._init_seats()

    def set_num_seats(self, num_seats: int):
        """Set number of seats and reset lineup"""
        if num_seats != self._num_seats:
            self._num_seats = num_seats
            self._rowers = [None] * num_seats
            self._update_display()

    def _init_seats(self):
        """Initialize seat display"""
        self._rowers = [None] * self._num_seats
        self._update_display()

    def add_rower(self, name: str) -> bool:
        """Add rower to first empty seat"""
        if name in self._rowers:
            return False  # Already in lineup
        for i in range(len(self._rowers)):
            if self._rowers[i] is None:
                self._rowers[i] = name
                self._update_display()
                return True
        return False  # No empty seats

    def remove_rower_at_seat(self, seat: int):
        """Remove rower at specific seat"""
        if 0 <= seat < len(self._rowers):
            self._rowers[seat] = None
            self._update_display()

    def set_rower_at_seat(self, seat: int, name: str):
        """Place a rower at a specific seat"""
        if 0 <= seat < len(self._rowers):
            self._rowers[seat] = name
            self._update_display()

    def remove_rower(self, name: str):
        """Remove rower by name"""
        if name in self._rowers:
            idx = self._rowers.index(name)
            self._rowers[idx] = None
            self._update_display()

    def clear(self):
        """Clear all rowers from lineup"""
        self._rowers = [None] * self._num_seats
        self._update_display()

    def get_rowers(self) -> List[str]:
        """Return list of rowers (non-empty seats only)"""
        return [r for r in self._rowers if r is not None]

    def get_all_seats(self) -> List[Optional[str]]:
        """Return all seats including empty ones"""
        return self._rowers.copy()

    def set_rowers(self, rowers: List[Optional[str]]):
        """Set rowers directly (for copy operation)"""
        self._rowers = rowers[:self._num_seats]
        # Pad with None if needed
        while len(self._rowers) < self._num_seats:
            self._rowers.append(None)
        self._update_display()

    def _update_display(self):
        try:
            container = self.query_one(f"#seats-{self.lineup_id}", Vertical)
            container.remove_children()

            labels = self.SEAT_LABELS.get(self._num_seats, [str(i) for i in range(1, self._num_seats + 1)])

            for i, label in enumerate(labels):
                rower = self._rowers[i] if i < len(self._rowers) else None
                item = LineupRowerItem(self.lineup_id, i, label, rower)
                container.mount(item)
        except Exception:
            pass  # Widget not ready yet


class ResultsPanel(Static):
    """Panel showing analysis results"""

    def compose(self) -> ComposeResult:
        yield Static("[bold]Analysis Results[/]", id="results-title")
        yield DataTable(id="results-table")


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class RowingAnalyzerApp(App):
    """Main TUI Application"""

    CSS = """
    Screen {
        background: $surface;
        overflow: auto;
    }

    #main-container {
        height: 1fr;
        width: 100%;
        min-height: 30;
    }

    #left-panel {
        width: 35%;
        min-width: 40;
        height: 100%;
        border: solid $primary;
        padding: 1;
    }

    #right-panel {
        width: 65%;
        height: 1fr;
        padding: 1;
    }

    #roster-scroll {
        height: 1fr;
        border: solid $primary-darken-2;
        margin-top: 1;
    }

    .rower-item {
        padding: 0 1;
        margin: 0;
        height: auto;
    }

    .rower-item:hover {
        background: $primary;
    }

    .lineup-box {
        border: solid $accent;
        padding: 0 1;
        margin: 0 1 0 0;
        height: auto;
        max-height: 14;
        width: 1fr;
    }

    .lineup-header {
        height: 1;
        margin: 0;
    }

    .action-hint {
        margin-left: 1;
        height: 3;
        content-align: center middle;
    }

    .rower-item.selected {
        background: $warning;
        color: $text;
    }

    .lineup-content {
        padding: 0 1;
    }

    #config-section {
        border: solid $secondary;
        padding: 1;
        margin-bottom: 0;
        height: auto;
        max-height: 8;
    }

    #config-section Horizontal {
        height: auto;
    }

    #config-section Vertical {
        width: 1fr;
        height: auto;
        padding: 0 1 0 0;
    }

    #config-section Select {
        width: 100%;
        margin: 0;
    }

    #config-section Static {
        height: 1;
    }

    #lineups-section {
        height: auto;
        width: 100%;
    }

    #lineup-boxes {
        height: auto;
        width: 100%;
    }

    #bottom-panel {
        height: auto;
        min-height: 8;
        max-height: 15;
        border: solid $success;
        padding: 1;
        margin-top: 1;
    }

    Button {
        margin: 0 1;
    }

    Input, Select {
        margin: 1 0;
    }

    #status-bar {
        dock: bottom;
        height: 1;
        background: $primary-darken-2;
        padding: 0 1;
    }

    #active-lineup-indicator {
        color: $success;
        text-style: bold;
    }

    .copy-btn {
        min-width: 4;
        width: auto;
        height: 1;
        margin: 0 0 0 1;
        border: none;
        background: $primary-darken-2;
    }

    .copy-btn:hover {
        background: $primary;
    }

    #action-buttons {
        height: 3;
        padding: 0;
        margin: 1 0;
    }

    #action-buttons Button {
        margin: 0 1 0 0;
    }

    LineupRowerItem {
        height: auto;
        padding: 0;
    }

    LineupRowerItem:hover {
        background: $error-darken-2;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("a", "analyze", "Analyze"),
        Binding("c", "clear_lineups", "Clear"),
        Binding("escape", "clear_selection", "Deselect"),
    ]

    BOAT_SEATS = {
        '1x': 1, '2x': 2, '2-': 2,
        '4x': 4, '4+': 4, '4-': 4,
        '8+': 8,
    }

    def __init__(self):
        super().__init__()
        self.roster_manager = RosterManager()
        self.analyzer: Optional[BoatAnalyzer] = None
        self.target_distance = 1000
        self.boat_class = '4+'
        self.selected_regatta: Optional[str] = None
        self.sort_mode = 'name'
        self.selected_rower: Optional[str] = None  # Currently selected rower for placement
        self.selected_rower_source: Optional[Tuple[str, int]] = None  # (lineup_id, seat) if from lineup

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal(id="main-container"):
            # Left panel - Roster
            with Vertical(id="left-panel"):
                yield Static("[bold]Roster[/]")
                yield Select([], prompt="Select Regatta...", id="regatta-select")
                yield Select([
                    ("Sort: Name", "name"),
                    ("Sort: Age (young first)", "age_asc"),
                    ("Sort: Age (old first)", "age_desc"),
                    ("Sort: Gender (F first)", "gender_f"),
                    ("Sort: Gender (M first)", "gender_m"),
                    ("Sort: Port", "port"),
                    ("Sort: Starboard", "starboard"),
                ], value="name", id="sort-select")
                yield Input(placeholder="Search rowers...", id="search-input")
                yield Static("", id="roster-status")
                yield Static("[dim]Click rower to select, click seat to place[/dim]", id="roster-hint")
                yield ScrollableContainer(id="roster-scroll")

            # Right panel - Config and Lineups
            with Vertical(id="right-panel"):
                # Configuration
                with Vertical(id="config-section"):
                    with Horizontal():
                        with Vertical():
                            yield Static("[bold]Race Distance:[/]")
                            yield Select([
                                ("1K (1000m)", 1000),
                                ("2K (2000m)", 2000),
                                ("5K (5000m)", 5000),
                            ], value=1000, id="distance-select")
                        with Vertical():
                            yield Static("[bold]Boat Class:[/]")
                            yield Select([
                                ("1x - Single", "1x"),
                                ("2x - Double", "2x"),
                                ("2- - Pair", "2-"),
                                ("4x - Quad", "4x"),
                                ("4+ - Four", "4+"),
                                ("4- - Straight Four", "4-"),
                                ("8+ - Eight", "8+"),
                            ], value="4+", id="boat-select")

                # Action buttons
                with Horizontal(id="action-buttons"):
                    yield Button("Analyze All", id="analyze-btn", variant="primary")
                    yield Button("Clear All", id="clear-btn", variant="error")
                    yield Static(" [dim]| right-click to remove a rower[/dim]", classes="action-hint")

                # Lineups
                with Vertical(id="lineups-section"):
                    with Horizontal(id="lineup-boxes"):
                        with Vertical(classes="lineup-box"):
                            with Horizontal(classes="lineup-header"):
                                yield Static("[bold cyan]Lineup A[/]")
                                yield Button("A>B", id="copy-a-to-b", classes="copy-btn")
                                yield Button("A>C", id="copy-a-to-c", classes="copy-btn")
                            yield LineupPanel("A", "")
                        with Vertical(classes="lineup-box"):
                            with Horizontal(classes="lineup-header"):
                                yield Static("[bold cyan]Lineup B[/]")
                                yield Button("B>A", id="copy-b-to-a", classes="copy-btn")
                                yield Button("B>C", id="copy-b-to-c", classes="copy-btn")
                            yield LineupPanel("B", "")
                        with Vertical(classes="lineup-box"):
                            with Horizontal(classes="lineup-header"):
                                yield Static("[bold cyan]Lineup C[/]")
                                yield Button("C>A", id="copy-c-to-a", classes="copy-btn")
                                yield Button("C>B", id="copy-c-to-b", classes="copy-btn")
                            yield LineupPanel("C", "")

        # Bottom panel - Results
        with Vertical(id="bottom-panel"):
            yield Static("[bold]Analysis Results[/]")
            yield DataTable(id="results-table")

        yield Static(f"Distance: 1000m | Boat: 4+", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        """Initialize on mount"""
        self.load_data()
        self._update_status_bar()

    def load_data(self):
        """Load data from Excel file"""
        try:
            filepath = Path("2026 WRC Racing Spreadsheet.xlsx")
            if not filepath.exists():
                self.notify("Excel file not found!", severity="error")
                return

            success = self.roster_manager.load_from_excel(str(filepath))

            # Write load log to file for debugging
            with open("load_log.txt", "w") as f:
                f.write("=== Data Load Log ===\n")
                for msg in self.roster_manager.load_log:
                    f.write(msg + "\n")
                f.write("=====================\n")

            if success:
                self.analyzer = BoatAnalyzer(self.roster_manager)

                # Populate regatta dropdown with clean display names
                regattas = self.roster_manager.regattas
                display_names = self.roster_manager.regatta_display_names
                if regattas:
                    select = self.query_one("#regatta-select", Select)
                    # Use display name for label, actual column name for value
                    options = [("All Rowers (no filter)", "__all__")]
                    for col in regattas:
                        display = display_names.get(col, col)
                        options.append((display, col))
                    select.set_options(options)

                # Show all rowers with scores initially
                self._populate_roster(None)

                self.notify(f"Loaded {len(self.roster_manager.rowers)} rowers", severity="information")
            else:
                self.notify("Failed to load data", severity="error")

        except Exception as e:
            self.notify(f"Error: {e}", severity="error")

    def _populate_roster(self, regatta: Optional[str]):
        """Populate roster list with rowers"""
        roster_container = self.query_one("#roster-scroll", ScrollableContainer)

        # Clear existing items
        roster_container.remove_children()

        # Get rowers
        if regatta and regatta != "__all__":
            rower_names = self.roster_manager.get_attending_rowers(regatta)
            base_status = f"{len(rower_names)} attending"
        else:
            rower_names = self.roster_manager.get_all_rowers_with_scores()
            base_status = f"{len(rower_names)} with scores"

        # Get rower objects for sorting/filtering
        rowers_list = [(name, self.roster_manager.get_rower(name)) for name in rower_names]
        rowers_list = [(n, r) for n, r in rowers_list if r is not None]

        # Apply sort/filter
        if self.sort_mode == 'name':
            rowers_list.sort(key=lambda x: x[0])
        elif self.sort_mode == 'age_asc':
            rowers_list.sort(key=lambda x: x[1].age)
        elif self.sort_mode == 'age_desc':
            rowers_list.sort(key=lambda x: x[1].age, reverse=True)
        elif self.sort_mode == 'gender_f':
            rowers_list.sort(key=lambda x: (0 if x[1].gender == 'F' else 1, x[0]))
        elif self.sort_mode == 'gender_m':
            rowers_list.sort(key=lambda x: (0 if x[1].gender == 'M' else 1, x[0]))
        elif self.sort_mode == 'port':
            # Filter to only rowers who CAN row port
            rowers_list = [(n, r) for n, r in rowers_list if r.side_port]
            # Sort: exclusive port (no starboard) first, then both-side rowers
            # Within each group, sort by name
            rowers_list.sort(key=lambda x: (
                0 if (x[1].side_port and not x[1].side_starboard) else 1,
                x[0]
            ))
        elif self.sort_mode == 'starboard':
            # Filter to only rowers who CAN row starboard
            rowers_list = [(n, r) for n, r in rowers_list if r.side_starboard]
            # Sort: exclusive starboard (no port) first, then both-side rowers
            # Within each group, sort by name
            rowers_list.sort(key=lambda x: (
                0 if (x[1].side_starboard and not x[1].side_port) else 1,
                x[0]
            ))

        status_text = f"{len(rowers_list)} rowers ({base_status})"
        self.query_one("#roster-status", Static).update(status_text)

        # Add rower items
        for name, rower in rowers_list:
            item = RowerListItem(name, rower)
            item.add_class("rower-item")
            roster_container.mount(item)

    def on_rower_list_item_selected(self, message: RowerListItem.Selected) -> None:
        """Handle rower selection from roster - select rower for placement"""
        if self.selected_rower == message.rower_name and self.selected_rower_source is None:
            # Clicking same rower again deselects them
            self._clear_selection()
            self.notify("Selection cleared")
        else:
            self._select_rower(message.rower_name, source=None)

    def _select_rower(self, rower_name: str, source: Optional[Tuple[str, int]] = None):
        """Select a rower for placement. Source is (lineup_id, seat) if from a lineup."""
        # Clear previous selection highlighting
        for item in self.query(RowerListItem):
            item.remove_class("selected")

        # Set new selection
        self.selected_rower = rower_name
        self.selected_rower_source = source

        # Highlight the selected rower in roster if present
        for item in self.query(RowerListItem):
            if item.rower_name == rower_name:
                item.add_class("selected")
                break

        source_text = f" from Lineup {source[0]}" if source else ""
        self.notify(f"Selected {rower_name}{source_text} - click a seat to place")

    def _clear_selection(self):
        """Clear the current rower selection"""
        self.selected_rower = None
        self.selected_rower_source = None
        for item in self.query(RowerListItem):
            item.remove_class("selected")

    def action_clear_selection(self) -> None:
        """Action to clear rower selection (Escape key)"""
        if self.selected_rower:
            self._clear_selection()
            self.notify("Selection cleared")

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select changes"""
        if event.select.id == "regatta-select":
            value = str(event.value) if event.value else None
            if value == "__all__":
                self.selected_regatta = None
            else:
                self.selected_regatta = value

            self._populate_roster(self.selected_regatta)
            self._update_status_bar()

        elif event.select.id == "sort-select":
            self.sort_mode = str(event.value)
            self._populate_roster(self.selected_regatta)

        elif event.select.id == "distance-select":
            self.target_distance = int(event.value)
            self._update_status_bar()

        elif event.select.id == "boat-select":
            self.boat_class = str(event.value)
            num_seats = self.BOAT_SEATS.get(self.boat_class, 4)
            # Update all lineup panels with new seat count
            for panel in self.query(LineupPanel):
                panel.set_num_seats(num_seats)
            self._update_status_bar()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes"""
        if event.input.id == "search-input":
            self._filter_roster(event.value)

    def _filter_roster(self, search_term: str):
        """Filter roster by search term"""
        search_lower = search_term.lower()
        for item in self.query(RowerListItem):
            if search_lower in item.rower_name.lower():
                item.display = True
            else:
                item.display = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        if event.button.id == "analyze-btn":
            self.action_analyze()
        elif event.button.id == "clear-btn":
            self.action_clear_lineups()
        elif event.button.id.startswith("copy-"):
            # Parse copy button id: copy-X-to-Y
            parts = event.button.id.split("-")
            if len(parts) == 4:
                source = parts[1].upper()
                dest = parts[3].upper()
                self._copy_lineup(source, dest)

    def _copy_lineup(self, source: str, dest: str):
        """Copy lineup from source to destination"""
        source_panel = None
        dest_panel = None
        for panel in self.query(LineupPanel):
            if panel.lineup_id == source:
                source_panel = panel
            elif panel.lineup_id == dest:
                dest_panel = panel

        if source_panel and dest_panel:
            rowers = source_panel.get_all_seats()
            dest_panel.set_rowers(rowers)
            self.notify(f"Copied Lineup {source} to Lineup {dest}")

    def on_lineup_rower_item_remove_rower(self, message: LineupRowerItem.RemoveRower) -> None:
        """Handle removing a rower from a lineup"""
        for panel in self.query(LineupPanel):
            if panel.lineup_id == message.lineup_id:
                panel.remove_rower_at_seat(message.seat)
                self.notify(f"Removed {message.rower_name} from Lineup {message.lineup_id}")
                break

    def on_lineup_rower_item_seat_clicked(self, message: LineupRowerItem.SeatClicked) -> None:
        """Handle clicking an empty seat - place selected rower there"""
        if not self.selected_rower:
            self.notify("Select a rower first, then click a seat", severity="warning")
            return

        for panel in self.query(LineupPanel):
            if panel.lineup_id == message.lineup_id:
                # Check if rower already in this lineup (unless moving within same lineup)
                is_same_lineup = (self.selected_rower_source and
                                  self.selected_rower_source[0] == message.lineup_id)
                current_rowers = panel.get_rowers()
                if self.selected_rower in current_rowers and not is_same_lineup:
                    self.notify(f"[bold]{self.selected_rower}[/] is already in Lineup {message.lineup_id}!", severity="error")
                    return

                # If moving from another seat in a lineup, clear the source seat
                if self.selected_rower_source:
                    src_lineup_id, src_seat = self.selected_rower_source
                    for src_panel in self.query(LineupPanel):
                        if src_panel.lineup_id == src_lineup_id:
                            src_panel.remove_rower_at_seat(src_seat)
                            break
                    # Update source to new location (moving within lineups)
                    self.selected_rower_source = (message.lineup_id, message.seat)

                # Place rower at the specific seat
                panel.set_rower_at_seat(message.seat, self.selected_rower)
                self.notify(f"Placed {self.selected_rower} in Lineup {message.lineup_id}")

                # Don't set source if placing from roster - allows placing in multiple lineups
                # Source only gets set above when moving FROM a lineup
                break

    def on_lineup_rower_item_rower_clicked(self, message: LineupRowerItem.RowerClicked) -> None:
        """Handle left-clicking a rower in a lineup - select or swap"""
        if self.selected_rower and self.selected_rower != message.rower_name:
            # Swap: there's a selected rower and we clicked a different rower
            for panel in self.query(LineupPanel):
                if panel.lineup_id == message.lineup_id:
                    # Check if selected rower already in this lineup at a different seat
                    existing_rowers = panel.get_all_seats()

                    if self.selected_rower_source:
                        # Swapping from lineup to lineup
                        src_lineup_id, src_seat = self.selected_rower_source

                        if src_lineup_id == message.lineup_id:
                            # Same lineup - just swap seats
                            panel.set_rower_at_seat(src_seat, message.rower_name)
                            panel.set_rower_at_seat(message.seat, self.selected_rower)
                            self.notify(f"Swapped {self.selected_rower} and {message.rower_name}")
                        else:
                            # Different lineups - swap between lineups
                            for src_panel in self.query(LineupPanel):
                                if src_panel.lineup_id == src_lineup_id:
                                    src_panel.set_rower_at_seat(src_seat, message.rower_name)
                                    break
                            panel.set_rower_at_seat(message.seat, self.selected_rower)
                            self.notify(f"Swapped {self.selected_rower} and {message.rower_name}")
                        # Update source to new location when moving between lineups
                        self.selected_rower_source = (message.lineup_id, message.seat)
                    else:
                        # Placing from roster - just replace the clicked rower
                        # DON'T set source - allows placing in multiple lineups
                        panel.set_rower_at_seat(message.seat, self.selected_rower)
                        self.notify(f"Replaced {message.rower_name} with {self.selected_rower}")
                    break
        else:
            # No swap - just select this rower from the lineup
            self._select_rower(message.rower_name, source=(message.lineup_id, message.seat))

    def action_analyze(self) -> None:
        """Analyze all lineups"""
        if not self.analyzer:
            self.notify("Data not loaded", severity="error")
            return

        table = self.query_one("#results-table", DataTable)
        table.clear(columns=True)

        # Add columns
        table.add_column("Place", width=6)
        table.add_column("Lineup", width=8)
        table.add_column("Rowers", width=7)
        table.add_column("Split", width=8)
        table.add_column("Raw Time", width=9)
        table.add_column("Handicap", width=9)
        table.add_column("Adjusted", width=9)
        table.add_column("Avg W", width=7)
        table.add_column("Port W", width=7)
        table.add_column("Stbd W", width=7)
        table.add_column("% Diff", width=7)

        # Collect all results first
        results = []
        for lineup_id in ['A', 'B', 'C']:
            for panel in self.query(LineupPanel):
                if panel.lineup_id == lineup_id:
                    rowers = panel.get_rowers()
                    if not rowers:
                        continue

                    result = self.analyzer.analyze_lineup(
                        rowers, self.target_distance, self.boat_class
                    )
                    result['lineup_id'] = lineup_id
                    result['rower_list'] = rowers
                    results.append(result)
                    break

        # Sort by adjusted time (fastest first), errors go to end
        def sort_key(r):
            if 'error' in r and 'avg_watts' not in r:
                return float('inf')
            return r.get('adjusted_time', float('inf'))

        results.sort(key=sort_key)

        # Add rows with place
        for place, result in enumerate(results, 1):
            lineup_id = result['lineup_id']
            rowers = result['rower_list']

            if 'error' in result and 'avg_watts' not in result:
                table.add_row(
                    "-",
                    f"[bold]{lineup_id}[/]",
                    str(len(rowers)),
                    "ERROR",
                    result.get('error', 'Unknown'),
                    "-", "-", "-", "-", "-", "-"
                )
            else:
                place_str = f"[bold green]{place}[/]" if place == 1 else str(place)
                port_w = f"{result['port_watts_total']:.0f}" if result.get('port_watts_total') else "-"
                stbd_w = f"{result['starboard_watts_total']:.0f}" if result.get('starboard_watts_total') else "-"
                pct_diff = f"{result['side_balance_pct']:.1f}%" if result.get('side_balance_pct') else "-"
                table.add_row(
                    place_str,
                    f"[bold]{lineup_id}[/]",
                    str(result['rower_count']),
                    format_split(result['boat_split_500m']),
                    format_time(result['raw_time']),
                    f"-{result['handicap_seconds']:.1f}s",
                    format_time(result['adjusted_time']),
                    f"{result['avg_watts']:.0f}",
                    port_w,
                    stbd_w,
                    pct_diff
                )

        self.notify("Analysis complete", severity="information")

    def action_clear_lineups(self) -> None:
        """Clear all lineups"""
        for panel in self.query(LineupPanel):
            panel.clear()
        self.notify("All lineups cleared", severity="warning")

    def action_remove_last(self) -> None:
        """Remove last rower from active lineup"""
        for panel in self.query(LineupPanel):
            if panel.lineup_id == self.active_lineup:
                rowers = panel.get_rowers()
                if rowers:
                    removed = rowers[-1]
                    panel.remove_rower(removed)
                    self.notify(f"Removed {removed}")
                break

    def _update_status_bar(self) -> None:
        """Update the status bar"""
        regatta = self.selected_regatta or "All"
        status = self.query_one("#status-bar", Static)
        status.update(
            f"Distance: {self.target_distance}m | "
            f"Boat: {self.boat_class} | "
            f"Regatta: {regatta}"
        )

    def action_quit(self) -> None:
        self.exit()


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Main entry point"""
    app = RowingAnalyzerApp()
    app.run()


if __name__ == "__main__":
    main()
