# ConfDef

## Paper

The full published ConDef paper can be found [here](https://link.springer.com/chapter/10.1007/978-3-031-10464-0_41).

## Setup and Installation

The first step is to downlaod the repository.


Change into the folder using the `cd` command


The next step is to install all the dependencies. To do so we recommend creating and sourcing a new python virtual environment like so:

```bash
python3 -m venv env
source ./env/bin/activate
```

Next, install all the dependencies via

```bash
pip install -r requirements.txt
```

Nextly, install a dump of wikipedia [here](https://dumps.wikimedia.org/).

Finally, run the data pre-procesing via

```bash
python3 scrape.py
```

## Usage

To train the model, simply run `main.py` via

```bash
python3 main.py
```

To validate the model, simply check to make sure the model id is correct and run `validate_rouge.py` via

```bash
python3 execute.py
```

