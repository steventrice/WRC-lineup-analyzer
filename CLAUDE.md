# WRC Lineup Analyzer

A Streamlit-based rowing lineup analysis tool for masters rowing clubs.

## Tech Stack
- **Python** with Streamlit for the web interface
- **Google Sheets** integration for data persistence (roster, erg scores, event entries)
- **Pandas** for data manipulation
- Single file app: `rowing_analyzer_web.py` (~6800 lines)

## Core Features

### Lineup Sandbox
- Three lineup columns (A, B, C) for building and comparing crews
- **Auto-analysis**: Results appear automatically when a lineup is full (no button click needed)
- Inline analysis cards show raw and adjusted placement badges with fade-in animation
- Supports boat classes: 1x, 2x, 2-, 4x, 4+, 4-, 8+

### Analysis Engine
- **Paul's Law**: +5 seconds per doubling of distance for pace prediction
- **Power Law**: Personalized fatigue curves when athlete has multiple test distances
- **Age handicap**: US Rowing masters handicap factors
- **Erg-to-water conversion**: BioRow/Kleshnev boat factors for on-water predictions
- **% of GMS**: Compare lineup's raw time against Gold Medal Standards (loaded from separate Google Sheet)
- Port/starboard wattage balance calculations

### Regatta Management
- Event scheduling with eligibility filtering (gender, age category, boat class)
- **"Race This Lineup"** popover for entering lineups into events
- Hot-seating conflict detection (rowers in back-to-back events)
- Dashboard view showing all entries, conflicts, and timeline

### Autofill Optimizer
- Finds fastest available crew by raw or adjusted erg time
- Seat locking (preserve specific assignments)
- Rower exclusion (skip unavailable athletes)
- Event constraint mode (auto-apply gender/age rules)

## Key Classes

- `RosterManager`: Loads/manages athletes, erg scores, regattas, events from Google Sheets
- `BoatAnalyzer`: Calculates lineup projections, handicaps, and comparisons
- `LineupOptimizer`: Autofill algorithm for optimal crew selection

## Data Flow

1. **Google Sheets** → `RosterManager` loads roster, erg scores, club boats, regattas, events
2. **GMS Sheet** → `RosterManager.load_gms_from_google_sheets()` loads Gold Medal Standards (separate sheet via `gms_spreadsheet_id` secret)
3. User builds lineups in **Sandbox** columns
4. `BoatAnalyzer.analyze_lineup()` calculates projections when lineup is full
5. Entries saved back to **Google Sheets** via `save_entry_to_gsheet()`

## UI Layout (Sandbox Mode)

```
┌─────────────────────────────────────────────────────────┐
│ [Regatta ▼] [Distance ▼] [Boat ▼] [Clear All] [Reload] │
├──────────────┬──────────────┬──────────────┬───────────┤
│  Lineup A    │  Lineup B    │  Lineup C    │  Sidebar  │
│  [Seats...]  │  [Seats...]  │  [Seats...]  │  Roster   │
│  [Copy]      │  [Copy]      │  [Copy]      │  Autofill │
│  [Race This] │  [Race This] │  [Race This] │  Help     │
├──────────────┼──────────────┼──────────────┤           │
│  Analysis A  │  Analysis B  │  Analysis C  │           │
│  [1st raw]   │  [2nd raw]   │  [3rd raw]   │           │
│  [1st adj.]  │  [2nd adj.]  │  [3rd adj.]  │           │
│  Split/Time  │  Split/Time  │  Split/Time  │           │
└──────────────┴──────────────┴──────────────┴───────────┘
```

## Recent Changes (Jan 2026)

- **% of GMS**: Analysis cards now show lineup's raw time as percentage of Gold Medal Standard (matches regatta, boat class, gender, age category)
- **Inline auto-analysis**: Removed Analyze button; results appear automatically below each full lineup
- **Placement badges**: Show both raw and adjusted finish positions
- **Race This Lineup**: Single popover button replaced multiple event entry buttons
- Events panel moved to dialog for cleaner layout
- Autofill exclusion and seat locking features
