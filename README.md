# wizualizacja-projekt

This project contains:
- a Jupyter notebook: `plotly-demo.ipynb`
- a Dash dashboard app: `plotly_demo.py`
- an exercise dashboard (Darts time series): `exercise.py`

## 1) Install uv

If you do not have uv yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify:

```bash
uv --version
```

## 2) Create environment and install dependencies

From the project directory:

```bash
uv sync
```

This creates `.venv` and installs dependencies from `pyproject.toml`.

## A) Run Notebook

After `uv sync`:

1. Open `plotly-demo.ipynb` in VS Code.
2. Click the kernel picker in the notebook top-right corner.
3. Select the uv environment kernel.


## B) Run Dash App

```bash
uv run python plotly_demo.py
```

Open the URL shown in terminal (usually `http://127.0.0.1:8050`).

## C) Run Exercise Dashboard

```bash
uv run python exercise.py
```

Open the URL shown in terminal (usually `http://127.0.0.1:8050`).
