import warnings

import numpy as np
import openml
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore")

RANDOM_STATE = 1234


def load_data() -> pd.DataFrame:
	ds = openml.datasets.get_dataset(39)
	df, _, _, _ = ds.get_data()
	# Keep the same class filtering as in the notebook.
	df = df[df["class"].isin(["cp", "im", "pp", "imU"])].copy()
	return df


def train_model(df: pd.DataFrame):
	df_model = df.drop(columns=["lip", "chg"]).copy()
	X = df_model.drop(columns=["class"]).copy()
	y = df_model["class"].copy()

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=0.3,
		random_state=RANDOM_STATE,
		shuffle=True,
		stratify=y,
	)

	rfc = RandomForestClassifier(
		n_estimators=20,
		random_state=RANDOM_STATE,
		n_jobs=-1,
		max_features=2,
	)
	rfc.fit(X_train, y_train)
	y_pred = rfc.predict(X_test)

	return X_train, X_test, y_train, y_test, y_pred, rfc


def make_confusion_matrix_figure(y_test: pd.Series, y_pred: np.ndarray) -> go.Figure:
	class_labels = sorted(y_test.unique())
	cm = confusion_matrix(y_test, y_pred, labels=class_labels)

	fig = ff.create_annotated_heatmap(
		z=cm,
		x=class_labels,
		y=class_labels,
		colorscale="Blues",
		showscale=True,
		annotation_text=cm,
	)
	fig.update_layout(
		title="Confusion Matrix",
		xaxis_title="Predicted",
		yaxis_title="True",
		template="plotly_white",
		width=700,
		height=520,
	)
	return fig


def pick_tree_feature_pair(tree, X_train: pd.DataFrame, numeric_cols: list[str]) -> tuple[str, str]:
	importances = pd.Series(tree.feature_importances_, index=X_train.columns)
	ranked = [c for c in importances.sort_values(ascending=False).index if c in numeric_cols]

	pair = []
	for col in ranked:
		if col not in pair:
			pair.append(col)
		if len(pair) == 2:
			break

	if len(pair) < 2:
		for col in numeric_cols:
			if col not in pair:
				pair.append(col)
			if len(pair) == 2:
				break

	return pair[0], pair[1]


def to_class_index(pred, class_to_idx: dict) -> int:
	if pred in class_to_idx:
		return class_to_idx[pred]
	idx = int(pred)
	if 0 <= idx < len(class_to_idx):
		return idx
	raise ValueError(f"Cannot map prediction {pred!r} to class index")


def to_class_label(pred, classes: list[str], class_to_idx: dict) -> str:
	return classes[to_class_index(pred, class_to_idx)]


def build_discrete_colorscale(colors: list[str]) -> list[list]:
	n = len(colors)
	if n == 1:
		return [[0.0, colors[0]], [1.0, colors[0]]]

	colorscale = []
	for i, color in enumerate(colors):
		left = i / n
		right = (i + 1) / n
		colorscale.append([left, color])
		colorscale.append([right, color])
	return colorscale


def build_tree_payload(rfc, X_train: pd.DataFrame, y_train: pd.Series) -> list[dict]:
	classes = list(rfc.classes_)
	class_to_idx = {c: i for i, c in enumerate(classes)}
	numeric_cols = X_train.select_dtypes(include="number").columns.tolist()

	max_trees_to_show = min(30, len(rfc.estimators_))
	payload = []

	for tree in rfc.estimators_[:max_trees_to_show]:
		feature_1, feature_2 = pick_tree_feature_pair(tree, X_train, numeric_cols)

		step = 0.02
		x_min, x_max = X_train[feature_1].min() - 0.05, X_train[feature_1].max() + 0.05
		y_min, y_max = X_train[feature_2].min() - 0.05, X_train[feature_2].max() + 0.05
		xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

		base_row = X_train.median(numeric_only=True)
		grid_full = pd.DataFrame(np.tile(base_row.values, (xx.size, 1)), columns=X_train.columns)
		grid_full[feature_1] = xx.ravel()
		grid_full[feature_2] = yy.ravel()

		tree_pred = tree.predict(grid_full)
		z = np.array([to_class_index(p, class_to_idx) for p in tree_pred], dtype=int).reshape(xx.shape)
		z_class = np.array(classes, dtype=object)[z]

		# Predict points on the same 2D slice used by the background (other features fixed to medians).
		slice_points = pd.DataFrame(np.tile(base_row.values, (len(X_train), 1)), columns=X_train.columns)
		slice_points[feature_1] = X_train[feature_1].values
		slice_points[feature_2] = X_train[feature_2].values
		train_pred = np.array(
			[to_class_label(p, classes, class_to_idx) for p in tree.predict(slice_points)],
			dtype=object,
		)

		points = []
		for cls in classes:
			mask = y_train == cls
			points.append(
				{
					"x": X_train.loc[mask, feature_1],
					"y": X_train.loc[mask, feature_2],
					"true": cls,
					"pred": train_pred[mask],
				}
			)

		payload.append(
			{
				"f1": feature_1,
				"f2": feature_2,
				"xgrid": xx[0],
				"ygrid": yy[:, 0],
				"z": z,
				"z_class": z_class,
				"xrange": [float(x_min), float(x_max)],
				"yrange": [float(y_min), float(y_max)],
				"points": points,
			}
		)

	return payload


DF = load_data()
AXIS_COLS = DF.select_dtypes(include="number").columns.tolist()
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, Y_PRED, RFC = train_model(DF)
TREE_PAYLOAD = build_tree_payload(RFC, X_TRAIN, Y_TRAIN)
CLASS_VALUES = list(RFC.classes_)
CLASS_COLOR_MAP = {
	cls: px.colors.qualitative.Bold[i % len(px.colors.qualitative.Bold)]
	for i, cls in enumerate(CLASS_VALUES)
}
BOUNDARY_COLORSCALE = build_discrete_colorscale([CLASS_COLOR_MAP[cls] for cls in CLASS_VALUES])
REPORT = classification_report(Y_TEST, Y_PRED)
TEST_ACC = RFC.score(X_TEST, Y_TEST)


app = Dash(__name__)
app.title = "Notebook-based Plotly Dashboard"

app.layout = html.Div(
	[
		html.H1("Plotly + Dash tutorial dashboard"),
		html.P("Built from your notebook workflow (OpenML dataset 39)."),
		dcc.Tabs(
			[
				dcc.Tab(
					label="Exploration",
					children=[
						html.H3("Interactive Scatter"),
						html.Div(
							[
								html.Div(
									[
										html.Label("X axis"),
										dcc.Dropdown(
											id="scatter-x",
											options=[{"label": c, "value": c} for c in AXIS_COLS],
											value=AXIS_COLS[0],
											clearable=False,
										),
									],
									style={"width": "48%", "display": "inline-block"},
								),
								html.Div(
									[
										html.Label("Y axis"),
										dcc.Dropdown(
											id="scatter-y",
											options=[{"label": c, "value": c} for c in AXIS_COLS],
											value=AXIS_COLS[1] if len(AXIS_COLS) > 1 else AXIS_COLS[0],
											clearable=False,
										),
									],
									style={"width": "48%", "display": "inline-block", "float": "right"},
								),
							]
						),
						dcc.Graph(id="scatter-graph"),
						html.H3("Interactive Histogram"),
						dcc.Dropdown(
							id="hist-col",
							options=[{"label": c, "value": c} for c in AXIS_COLS],
							value=AXIS_COLS[0],
							clearable=False,
						),
						dcc.Graph(id="hist-graph"),
					],
				),
				dcc.Tab(
					label="Model",
					children=[
						html.H3(f"Random Forest test accuracy: {TEST_ACC:.4f}"),
						html.Pre(REPORT, style={"background": "#f6f8fa", "padding": "12px"}),
						dcc.Graph(figure=make_confusion_matrix_figure(Y_TEST, Y_PRED)),
					],
				),
				dcc.Tab(
					label="Tree Boundaries",
					children=[
						html.P("Slider shows decision boundary of each tree using tree-specific top 2 features."),
						dcc.Slider(
							id="tree-slider",
							min=1,
							max=len(TREE_PAYLOAD),
							step=1,
							value=1,
							marks={i: str(i) for i in range(1, len(TREE_PAYLOAD) + 1)},
						),
						dcc.Graph(id="tree-graph"),
					],
				),
			]
		),
	],
	style={"maxWidth": "1300px", "margin": "0 auto", "padding": "16px"},
)


@app.callback(Output("scatter-graph", "figure"), Input("scatter-x", "value"), Input("scatter-y", "value"))
def update_scatter(x_col: str, y_col: str):
	fig = px.scatter(
		DF,
		x=x_col,
		y=y_col,
		color="class",
		symbol="class",
		hover_data=[c for c in ["lip", "chg", "aac", "alm1", "alm2"] if c in DF.columns],
		title=f"Scatter plot: {y_col} vs {x_col}",
		template="plotly_white",
	)
	return fig


@app.callback(Output("hist-graph", "figure"), Input("hist-col", "value"))
def update_hist(col: str):
	fig = px.histogram(
		DF,
		x=col,
		color="class",
		barmode="overlay",
		nbins=30,
		opacity=0.6,
		title=f"Interactive Histogram of {col}",
		template="plotly_white",
	)
	return fig


@app.callback(Output("tree-graph", "figure"), Input("tree-slider", "value"))
def update_tree_graph(tree_idx: int):
	idx = max(1, min(tree_idx, len(TREE_PAYLOAD))) - 1
	payload = TREE_PAYLOAD[idx]

	fig = go.Figure()
	fig.add_trace(
		go.Heatmap(
			x=payload["xgrid"],
			y=payload["ygrid"],
			z=payload["z"],
			hoverinfo="skip",
			zmin=-0.5,
			zmax=len(CLASS_VALUES) - 0.5,
			zsmooth=False,
			colorscale=BOUNDARY_COLORSCALE,
			opacity=0.55,
			showscale=False,
			name="Decision plane",
		)
	)

	for pts in payload["points"]:
		fig.add_trace(
			go.Scatter(
				x=pts["x"],
				y=pts["y"],
				customdata=np.array(pts["pred"], dtype=object),
				mode="markers",
				name=f"{pts['true']}",
				marker=dict(
					size=7,
					color=CLASS_COLOR_MAP[pts["true"]],
					line=dict(width=0.5, color="white"),
				),
				hovertemplate=(
					f"True class: {pts['true']}<br>"
					"Tree predicts: %{customdata}<br>"
					f"{payload['f1']}: %{{x:.3f}}<br>{payload['f2']}: %{{y:.3f}}<extra></extra>"
				),
			)
		)

	fig.update_layout(
		title=f"Decision boundary of tree {idx+1}/{len(TREE_PAYLOAD)} on {payload['f1']}/{payload['f2']}",
		xaxis=dict(title=payload["f1"], range=payload["xrange"]),
		yaxis=dict(title=payload["f2"], range=payload["yrange"]),
		template="plotly_white",
		width=980,
		height=680,
	)
	return fig


if __name__ == "__main__":
	app.run(debug=True)
