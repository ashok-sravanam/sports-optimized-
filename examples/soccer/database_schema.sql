-- Soccer Tactical Analysis Database Schema
-- PostgreSQL Database Schema for tracking players, formations, events, and positions

-- Teams and Matches
CREATE TABLE teams (
    team_id SERIAL PRIMARY KEY,
    team_name VARCHAR(100) NOT NULL,
    team_color VARCHAR(7), -- Hex color code
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE matches (
    match_id SERIAL PRIMARY KEY,
    team_a_id INT REFERENCES teams(team_id),
    team_b_id INT REFERENCES teams(team_id),
    match_date TIMESTAMP,
    venue VARCHAR(100),
    competition VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Players
CREATE TABLE players (
    player_id SERIAL PRIMARY KEY,
    team_id INT REFERENCES teams(team_id),
    jersey_number INT NOT NULL,
    name VARCHAR(100),
    position VARCHAR(20), -- GK, CB, LB, RB, CDM, CM, CAM, LW, RW, ST, etc.
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(team_id, jersey_number) -- Ensure unique jersey numbers per team
);

-- Formations (Coach Input: Initial Formation)
CREATE TABLE formations (
    formation_id SERIAL PRIMARY KEY,
    match_id INT REFERENCES matches(match_id),
    team_id INT REFERENCES teams(team_id),
    formation_type VARCHAR(10), -- "4-3-3", "4-4-2", "3-5-2", etc.
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE formation_positions (
    formation_id INT REFERENCES formations(formation_id),
    player_id INT REFERENCES players(player_id),
    jersey_number INT NOT NULL,
    tactical_position VARCHAR(20), -- LW, RCB, CDM, etc.
    x_start FLOAT, -- Starting position on tactical board (0-1 normalized)
    y_start FLOAT, -- Starting position on tactical board (0-1 normalized)
    PRIMARY KEY (formation_id, player_id)
);

-- Match Events
CREATE TABLE events (
    event_id SERIAL PRIMARY KEY,
    match_id INT REFERENCES matches(match_id),
    event_type VARCHAR(20), -- 'GOAL', 'SUBSTITUTION', 'YELLOW_CARD', 'RED_CARD', 'ASSIST'
    team_id INT REFERENCES teams(team_id),
    timestamp TIME,
    player_id INT REFERENCES players(player_id),
    jersey_number INT,
    details JSONB, -- {assisted_by, goal_type, card_reason, etc}
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE substitutions (
    sub_id SERIAL PRIMARY KEY,
    match_id INT REFERENCES matches(match_id),
    team_id INT REFERENCES teams(team_id),
    player_out INT REFERENCES players(player_id),
    player_in INT REFERENCES players(player_id),
    jersey_out INT,
    jersey_in INT,
    timestamp TIME,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tracked Positions (linked to jersey numbers)
CREATE TABLE tracked_positions (
    position_id SERIAL PRIMARY KEY,
    frame_id INT,
    match_id INT REFERENCES matches(match_id),
    timestamp FLOAT, -- Video timestamp in seconds
    jersey_number INT,
    team_id INT REFERENCES teams(team_id),
    x FLOAT, -- Homography transformed x coordinate
    y FLOAT, -- Homography transformed y coordinate
    confidence FLOAT, -- Detection confidence
    tracker_id INT, -- Original tracker ID from computer vision
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_tracked_positions_match_timestamp ON tracked_positions(match_id, timestamp);
CREATE INDEX idx_tracked_positions_jersey_team ON tracked_positions(jersey_number, team_id);
CREATE INDEX idx_events_match_timestamp ON events(match_id, timestamp);
CREATE INDEX idx_players_team_jersey ON players(team_id, jersey_number);

-- Sample data for testing
INSERT INTO teams (team_name, team_color) VALUES 
('Real Madrid', '#FFFFFF'),
('Getafe', '#0066CC'),
('PSG', '#004170'),
('Liverpool', '#C8102E');

-- Sample match
INSERT INTO matches (team_a_id, team_b_id, match_date, venue, competition) VALUES 
(1, 2, '2024-01-15 20:00:00', 'Santiago Bernabeu', 'La Liga'),
(3, 4, '2024-01-20 15:00:00', 'Parc des Princes', 'Champions League');

-- Sample players for Real Madrid
INSERT INTO players (team_id, jersey_number, name, position) VALUES 
(1, 1, 'Courtois', 'GK'),
(1, 2, 'Carvajal', 'RB'),
(1, 4, 'Ramos', 'CB'),
(1, 5, 'Varane', 'CB'),
(1, 12, 'Marcelo', 'LB'),
(1, 8, 'Kroos', 'CM'),
(1, 10, 'Modric', 'CM'),
(1, 14, 'Casemiro', 'CDM'),
(1, 7, 'Hazard', 'LW'),
(1, 9, 'Benzema', 'ST'),
(1, 11, 'Vinicius', 'RW');

-- Sample players for Getafe
INSERT INTO players (team_id, jersey_number, name, position) VALUES 
(2, 1, 'Soria', 'GK'),
(2, 2, 'Dakonam', 'RB'),
(2, 4, 'Cabrera', 'CB'),
(2, 6, 'Djene', 'CB'),
(2, 17, 'Olivera', 'LB'),
(2, 8, 'Arambarri', 'CM'),
(2, 20, 'Maksimovic', 'CM'),
(2, 15, 'Cucurella', 'LW'),
(2, 7, 'Mata', 'ST'),
(2, 11, 'Unal', 'ST'),
(2, 19, 'Nyom', 'RW');
