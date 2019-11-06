\COPY distances FROM data/glove.840B.300d.first20k.trimmed.distances.csv DELIMITER '|' ESCAPE '\' CSV;

ALTER TABLE distances ADD PRIMARY KEY(term_a, term_b);

CREATE INDEX term_reversal ON distances(term_b, term_a);

CREATE INDEX distance_btree ON distances USING btree (distance);

CREATE VIEW clusters AS (
    SELECT 
        term_a, term_b, distance 
    FROM 
        distances 
    UNION 
    SELECT 
        term_b as term_a, term_a as term_b, distance 
    FROM 
        distances
);
