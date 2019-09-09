\COPY distances FROM data/glove.6B.300d.first20k.distances.csv DELIMITER '|' ESCAPE '\' CSV;
ALTER TABLE distances ADD PRIMARY KEY(term_a, term_b);
CREATE INDEX distance_btree ON distances USING btree (distance);

