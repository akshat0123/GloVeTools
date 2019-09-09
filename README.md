# GloVeTools
Class containing helper methods and PostgreSQL database storage for pre-trained
GloVe embeddings

### Prerequisites
Place one of the GloVe data text files in the `data` directory and update the
`distances\_path` field in the `glove.json` metadata file

### Running 
1) Run the `glove.sql` script (as a Postgres admin)
2) Run the `glove.py` script
3) Run the `load_distances.sql' script (as a Postgres admin)
