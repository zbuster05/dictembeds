# Dictembed

## Overview
Dictembed is a project meant to push the limits of automated lexicography with a process able to generate context aware descriptions of a given term within a text. It's been trained on a novel Wikipedia dataset. For more information, refer to our paper (insert link or something idfk).

## Setup and Installation

The first step is to clone the following repository.

```bash
git clone --recursive <repositor url>
```

Change into the folder like so:

```
cd dictembed
```

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

To execute the model, simply run `execute.py` via

```bash
python3 execute.py
```
