#!/usr/bin/env python3
"""
Lineup Comparison - Streamlit Web App
Analyzes potential boat lineups using Paul's Law and boat physics
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
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
        """Return side preference as a string like 'P', 'S', 'PS', 'X', etc."""
        parts = []
        if self.side_port:
            parts.append('P')
        if self.side_starboard:
            parts.append('S')
        if self.side_coxswain:
            parts.append('X')
        return ''.join(parts) if parts else '-'

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
    """Rowing physics calculations using Paul's Law"""

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

            side_port = bool(row.get('P', True)) if pd.notna(row.get('P')) else True
            side_starboard = bool(row.get('S', True)) if pd.notna(row.get('S')) else True
            side_cox = bool(row.get('X', False)) if pd.notna(row.get('X')) else False

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
        """Load scores from score sheets (1K, 5K, etc.)"""
        score_sheets = []

        for sheet in xl.sheet_names:
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
                       boat_class: str = '4+', calc_method: str = 'watts') -> Dict[str, Any]:
        """Analyze a lineup and return comprehensive results.

        calc_method: 'watts' (average watts, convert to split) or 'split' (average splits directly)
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

            projected_split = PhysicsEngine.pauls_law_projection(
                score.split_500m, score.distance, target_distance
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
# STREAMLIT APP
# =============================================================================

def check_password():
    """Returns True if the user has the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
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
            st.title("Lineup Comparison")
        st.text_input(
            "Enter password to access the app:",
            type="password",
            on_change=password_entered,
            key="password"
        )
        st.info("Contact your team leadership for access credentials.")
        return False

    if not st.session_state["password_correct"]:
        col_logo, col_title = st.columns([1, 10])
        with col_logo:
            st.image("wrc-badge-red.png", width=60)
        with col_title:
            st.title("Lineup Comparison")
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


@st.cache_resource(ttl=300)  # Cache for 5 minutes, then refresh from Google Sheets
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


def main():
    st.set_page_config(
        page_title="Lineup Comparison",
        page_icon="wrc-badge-red.png",
        layout="wide"
    )

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

    # Debug: show load log in expander
    with st.expander("Debug: Data Load Log", expanded=False):
        for log_entry in roster_manager.load_log:
            st.text(log_entry)

        # Show rowers without scores
        rowers_without_scores = [name for name, r in roster_manager.rowers.items() if not r.scores]
        if rowers_without_scores:
            st.text(f"\nRowers WITHOUT scores ({len(rowers_without_scores)}):")
            for name in sorted(rowers_without_scores):
                st.text(f"  - {name}")

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

    # Initialize selected rower state
    if 'selected_rower' not in st.session_state:
        st.session_state.selected_rower = None

    # =========================================================================
    # HEADER: Logo, Title, and Configuration
    # =========================================================================

    header_cols = st.columns([1, 4, 2, 2, 2, 2, 2, 2])

    with header_cols[0]:
        st.image("wrc-badge-red.png", width=50)

    with header_cols[1]:
        st.markdown("### Lineup Comparison")

    # Regatta selection
    regatta_options = {"All Rowers": "__all__"}
    for col in roster_manager.regattas:
        display = roster_manager.regatta_display_names.get(col, col)
        regatta_options[display] = col

    with header_cols[2]:
        selected_regatta_display = st.selectbox(
            "Regatta",
            options=list(regatta_options.keys()),
            label_visibility="collapsed",
            key=f"regatta_select_{st.session_state.cache_version}"  # Reset on refresh
        )
        selected_regatta = regatta_options.get(selected_regatta_display, "__all__")

    # Distance selection
    distance_options = {"1K": 1000, "2K": 2000, "5K": 5000}
    with header_cols[3]:
        selected_distance_display = st.selectbox(
            "Distance",
            options=list(distance_options.keys()),
            label_visibility="collapsed"
        )
        target_distance = distance_options[selected_distance_display]

    # Boat class selection
    boat_options = {"1x": "1x", "2x": "2x", "2-": "2-", "4x": "4x", "4+": "4+", "4-": "4-", "8+": "8+"}
    with header_cols[4]:
        boat_class = st.selectbox(
            "Boat",
            options=list(boat_options.keys()),
            index=4,  # Default to 4+
            label_visibility="collapsed"
        )

    # Calculate number of seats
    boat_seats = {'1x': 1, '2x': 2, '2-': 2, '4x': 4, '4+': 4, '4-': 4, '8+': 8}
    num_seats = boat_seats.get(boat_class, 4)

    # Resize lineups if boat size changed
    for lineup_key in ['lineup_a', 'lineup_b', 'lineup_c']:
        current = st.session_state[lineup_key]
        if len(current) != num_seats:
            new_lineup = [None] * num_seats
            for i in range(min(len(current), num_seats)):
                new_lineup[i] = current[i]
            st.session_state[lineup_key] = new_lineup

    with header_cols[5]:
        if st.button("Analyze", type="primary", use_container_width=True):
            st.session_state.analyze_clicked = True

    with header_cols[6]:
        if st.button("Clear All", type="secondary", use_container_width=True):
            st.session_state.lineup_a = [None] * num_seats
            st.session_state.lineup_b = [None] * num_seats
            st.session_state.lineup_c = [None] * num_seats
            st.session_state.selected_rower = None
            st.rerun()

    with header_cols[7]:
        if st.button("Reload", type="secondary", use_container_width=True, help=f"Reload data from Google Sheets (loaded: {st.session_state.get('last_load_time', '?')})"):
            st.session_state.cache_version += 1
            st.rerun()

    # Calculation method toggle
    calc_col1, calc_col2 = st.columns([1, 5])
    with calc_col1:
        calc_method = st.radio(
            "Calculation",
            options=["Split", "Watts"],
            horizontal=True,
            help="""**Split method**: Averages splits directly. More conservative, may better reflect real-world crew dynamics with varied abilities.

**Watts method**: Averages power (watts) then converts to split. Faster prediction, assumes perfect synchronization."""
        )
        calc_method = calc_method.lower()

    st.divider()

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
        sort_mode = st.selectbox(
            "Sort by",
            options=list(sort_options.keys())
        )
        sort_mode = sort_options[sort_mode]

        # Search filter
        search_term = st.text_input("Search...", key="search")

        # Get filtered rower list - show ALL rowers, not just those with scores
        if selected_regatta and selected_regatta != "__all__":
            # For regatta filter, show attending rowers (with or without scores)
            rower_names = [name for name, r in roster_manager.rowers.items()
                          if r.is_attending(selected_regatta)]
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

        # Women section
        st.markdown(f"**Women ({len(women)})**")
        for name, rower in women:
            has_scores = bool(rower.scores)
            side = rower.side_preference_str()
            if not has_scores:
                display_text = f"{name} | (no scores)"
            elif show_erg_time:
                dist = 1000 if show_erg_time[0] == '1k' else 5000
                erg_time = format_erg_time(rower, dist, show_erg_time[1])
                display_text = f"{name} | {erg_time}"
            else:
                display_text = f"{name} | {rower.age} | {side}"

            is_selected = (st.session_state.selected_rower == name)
            btn_type = "primary" if is_selected else "secondary"

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
            if not has_scores:
                display_text = f"{name} | (no scores)"
            elif show_erg_time:
                dist = 1000 if show_erg_time[0] == '1k' else 5000
                erg_time = format_erg_time(rower, dist, show_erg_time[1])
                display_text = f"{name} | {erg_time}"
            else:
                display_text = f"{name} | {rower.age} | {side}"

            is_selected = (st.session_state.selected_rower == name)
            btn_type = "primary" if is_selected else "secondary"

            if st.button(display_text, key=f"rower_{name}", use_container_width=True, type=btn_type, disabled=not has_scores):
                if is_selected:
                    st.session_state.selected_rower = None
                else:
                    st.session_state.selected_rower = name
                st.rerun()

    # =========================================================================
    # MAIN AREA: Lineups
    # =========================================================================

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

    # Helper to get US Rowing Masters age category
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

    # Helper to calculate lineup stats
    def get_lineup_stats(lineup_list, roster_mgr):
        """Calculate average age and category for a full lineup"""
        rower_names = [r for r in lineup_list if r is not None]
        if len(rower_names) != len(lineup_list):
            return None  # Not full
        ages = []
        for name in rower_names:
            rower = roster_mgr.get_rower(name)
            if rower:
                ages.append(rower.age)
        if not ages:
            return None
        avg_age = sum(ages) / len(ages)
        category = get_masters_category(avg_age)
        return (avg_age, category)

    # Three columns for lineups
    lineup_cols = st.columns(3)

    lineups_config = [
        ("Lineup A", "lineup_a", lineup_cols[0]),
        ("Lineup B", "lineup_b", lineup_cols[1]),
        ("Lineup C", "lineup_c", lineup_cols[2])
    ]

    for title, key, col in lineups_config:
        with col:
            lineup = st.session_state[key]
            stats = get_lineup_stats(lineup, roster_manager)
            if stats:
                avg_age, category = stats
                st.markdown(f"**{title}** :gray[Avg: {avg_age:.1f} | Cat: {category}]")
            else:
                st.markdown(f"**{title}**")

            for i, label in enumerate(seat_labels):
                rower_name = lineup[i] if i < len(lineup) else None

                # Use secondary buttons for both, distinguish by text
                if rower_name:
                    btn_label = f"{label}: {rower_name}"
                else:
                    btn_label = f"{label}: ..."

                seat_cols = st.columns([5, 1])
                with seat_cols[0]:
                    if st.button(btn_label, key=f"seat_{key}_{i}", use_container_width=True):
                        if st.session_state.selected_rower:
                            selected = st.session_state.selected_rower
                            if selected in lineup and lineup.index(selected) != i:
                                st.warning(f"{selected} is already in {title}!")
                            else:
                                st.session_state[key][i] = selected
                                st.session_state.selected_rower = None
                                st.rerun()
                        elif rower_name:
                            st.session_state.selected_rower = rower_name
                            st.rerun()

                with seat_cols[1]:
                    if rower_name:
                        if st.button("❌", key=f"remove_{key}_{i}"):
                            st.session_state[key][i] = None
                            st.rerun()

            # Copy buttons
            copy_cols = st.columns(2)
            if key == "lineup_a":
                if copy_cols[0].button("A→B", key="copy_a_b", use_container_width=True):
                    st.session_state.lineup_b = st.session_state.lineup_a.copy()
                    st.rerun()
                if copy_cols[1].button("A→C", key="copy_a_c", use_container_width=True):
                    st.session_state.lineup_c = st.session_state.lineup_a.copy()
                    st.rerun()
            elif key == "lineup_b":
                if copy_cols[0].button("B→A", key="copy_b_a", use_container_width=True):
                    st.session_state.lineup_a = st.session_state.lineup_b.copy()
                    st.rerun()
                if copy_cols[1].button("B→C", key="copy_b_c", use_container_width=True):
                    st.session_state.lineup_c = st.session_state.lineup_b.copy()
                    st.rerun()
            else:
                if copy_cols[0].button("C→A", key="copy_c_a", use_container_width=True):
                    st.session_state.lineup_a = st.session_state.lineup_c.copy()
                    st.rerun()
                if copy_cols[1].button("C→B", key="copy_c_b", use_container_width=True):
                    st.session_state.lineup_b = st.session_state.lineup_c.copy()
                    st.rerun()

    # Analysis Results
    st.divider()
    st.subheader("Analysis Results")

    if st.session_state.get('analyze_clicked', False):
        st.session_state.analyze_clicked = False

        results = []

        for lineup_id, key in [("A", "lineup_a"), ("B", "lineup_b"), ("C", "lineup_c")]:
            lineup = st.session_state[key]
            rower_names_in_lineup = [r for r in lineup if r is not None]

            if rower_names_in_lineup:
                result = analyzer.analyze_lineup(rower_names_in_lineup, target_distance, boat_class, calc_method)
                result['lineup_id'] = lineup_id
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
                        'Lineup': result['lineup_id'],
                        'Rowers': '-',
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

                    table_data.append({
                        'Place': f"🥇 {place}" if place == 1 else str(place),
                        'Lineup': result['lineup_id'],
                        'Rowers': str(result['rower_count']),
                        'Split': format_split(result['boat_split_500m']),
                        'Raw Time': format_time(result['raw_time']),
                        'Handicap': f"-{result['handicap_seconds']:.1f}s",
                        'Adjusted': format_time(result['adjusted_time']),
                        'Avg W': f"{result['avg_watts']:.0f}",
                        'Port W': port_w,
                        'Stbd W': stbd_w,
                        '% Diff': pct_diff
                    })

            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Show detailed projections for each lineup
            with st.expander("View Detailed Projections"):
                for result in results:
                    if 'projections' in result:
                        st.markdown(f"**Lineup {result['lineup_id']}**")
                        proj_data = []
                        for proj in result['projections']:
                            if 'error' not in proj:
                                proj_data.append({
                                    'Seat': proj.get('seat', '-'),
                                    'Side': proj.get('seat_side', '-'),
                                    'Rower': proj['rower'],
                                    'Age': proj.get('age', '-'),
                                    'Pref': proj.get('side', '-'),
                                    'Source': f"{proj.get('source_distance', 0)//1000}K",
                                    'Source Split': format_split(proj.get('source_split', 0)),
                                    'Projected Split': format_split(proj.get('projected_split', 0)),
                                    'Watts': f"{proj.get('projected_watts', 0):.0f}"
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
                                    'Watts': proj.get('error', 'Error')
                                })

                        if proj_data:
                            st.dataframe(pd.DataFrame(proj_data), use_container_width=True, hide_index=True)
        else:
            st.info("No lineups to analyze. Add rowers to at least one lineup.")
    else:
        st.info("Click 'Analyze All Lineups' in the sidebar to see results.")


if __name__ == "__main__":
    main()
