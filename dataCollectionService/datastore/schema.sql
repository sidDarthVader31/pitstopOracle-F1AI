
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE driver (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    full_name VARCHAR(100) NOT NULL,
    nationality VARCHAR(50),
    date_of_birth DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE constructor (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    base_country VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE season (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    year INT NOT NULL,
    regulation_era VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE car (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    constructor_id UUID REFERENCES constructors(id),
    season_id UUID REFERENCES seasons(id),
    model_name VARCHAR(100),
    spec_details JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE driver_season (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    driver_id UUID REFERENCES drivers(id),
    season_id UUID REFERENCES seasons(id),
    constructor_id UUID REFERENCES constructors(id),
    car_id UUID REFERENCES cars(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE race (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    season_id UUID REFERENCES seasons(id),
    name VARCHAR(100),
    location VARCHAR(100),
    date DATE,
    weather JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE qualifying_result (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    race_id UUID REFERENCES races(id),
    driver_id UUID REFERENCES drivers(id),
    position INT,
    q1_time VARCHAR(20),
    q2_time VARCHAR(20),
    q3_time VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE race_result (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    race_id UUID REFERENCES races(id),
    driver_id UUID REFERENCES drivers(id),
    constructor_id UUID REFERENCES constructors(id),
    position INT,
    points INT,
    time VARCHAR(20),
    status VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Auto-update `updated_at`
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
   NEW.updated_at = NOW();
   RETURN NEW;
END;
$$ language 'plpgsql';

DO $$
DECLARE tbl TEXT;
BEGIN
  FOR tbl IN
    SELECT table_name FROM information_schema.tables
    WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
  LOOP
    EXECUTE format('
      CREATE TRIGGER trigger_%I_updated_at
      BEFORE UPDATE ON %I
      FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();', tbl, tbl);
  END LOOP;
END;
$$;
