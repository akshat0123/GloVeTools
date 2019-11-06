CREATE DATABASE glove_20k;
CREATE USER glove WITH ENCRYPTED PASSWORD 'gl0v3';

\c glove_20k;

CREATE TABLE embeddings (
    key VARCHAR(1000) PRIMARY KEY,
    embedding JSONB
);

CREATE TABLE distances (
    term_a VARCHAR(1000),
    term_b VARCHAR(1000),
    distance FLOAT
);

ALTER TABLE embeddings OWNER TO glove;
ALTER TABLE distances OWNER TO glove;
GRANT ALL PRIVILEGES ON DATABASE glove_20k TO glove;
GRANT ALL PRIVILEGES ON embeddings TO glove;
GRANT ALL PRIVILEGES ON distances TO glove;
