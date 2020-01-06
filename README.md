# GloVeTools
Class containing helper methods and PostgreSQL database storage for pre-trained
GloVe embeddings

### Prerequisites
Place one of the GloVe data text files in the `data` directory and update the
`path`, and `vocab_size` fields in the `glove.json` metadata file

### Running 
[Optional] Run `trim.py` to trim the starting dataset
1) Run the `glove.sql` script (as a Postgres admin)
2) Run the `glove.py` script
