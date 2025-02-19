#!/usr/bin/env python
# coding: utf-8

# Import necessary packages
import os
import glob
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px  # for interactive visualizations
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output  # Dash 2.x syntax
from dash.dependencies import Input, Output
import plotly.express as px


# Define path
data_dir = 'data'  # update with your actual path

excel_files = glob.glob(os.path.join(data_dir, '*.xlsx'))
main_file = glob.glob(os.path.join(data_dir, '*2026*'))

# Select useful columns
columns_to_keep = ['Title translated', 'Abstract', 'Funding amount in USD', 'Start Year', 'End Year', 'Funder', 'Country of standardized research organization', 'Funder Country', 'Fields of Research (ANZSRC 2020)']

all_data = pd.read_excel(main_file[0], skiprows=1)

# Select columns
all_data = all_data.loc[:, [col for col in columns_to_keep if col in all_data.columns]]

# all_data['Start Year'] = pd.to_datetime(all_data['Start Year'], errors='coerce')
all_data["Start Year"] = pd.to_datetime(
    all_data["Start Year"].astype(str),
    format="%Y",   # Only the 4-digit year
    errors="coerce"
)
all_data['Year'] = all_data['Start Year'].dt.year

# Remove duplicates, standardize text, etc.
all_data = all_data.drop_duplicates()

# Rename columns
all_data = all_data.rename(columns={'Title translated': 'Title'})
all_data = all_data.rename(columns={'Fields of Research (ANZSRC 2020)': 'Fields of research'})
all_data = all_data.rename(columns={'Funding amount in USD': 'Funding (USD)'})
all_data = all_data.rename(columns={'Country of standardized research organization': 'Research Country'})

# Clean up Fields of Research column (remove numeric codes, "Physical Sciences", semicolons, etc)
all_data['Fields of research'] = all_data['Fields of research'].str.replace('Physical Sciences;', '', regex=False)
all_data['Fields of research'] = all_data['Fields of research'].str.replace(r'\d+', '', regex=True)
all_data['Fields of research'] = all_data['Fields of research'].str.replace(r'^\s*;\s*', '', regex=True)
# Clean up extra whitespace
all_data['Fields of research'] = all_data['Fields of research'].str.replace(r'\s+', ' ', regex=True).str.strip()

# Clean up Funder column
all_data['Funder'] = all_data['Funder'].replace(r'^Directorate for.*', 'NSF', regex=True)
all_data['Funder'] = all_data['Funder'].replace({
    "UK Research and Innovation": "UKRI",
    "Swiss National Science Foundation": "SNSF",
    "Federal Ministry of Education and Research": "BMBF",
    "Australian Research Council": "ARC",
    "European Research Council": "ERC"
})

# Clean up Funder column
all_data['Funder'] = all_data['Funder'].replace({
    "Directorate for *": "NSF",
    "UK Research and Innovation": "UKRI",
    "Swiss National Science Foundation": "SNSF",
    "Federal Ministry of Education and Research": "BMBF",
    "Australian Research Council": "ARC",
    "European Research Council": "ERC"
})

all_data = all_data[all_data['Funder'] != 'United States Department of Defense']
all_data = all_data[all_data['Funder'] != 'National Natural Science Foundation of China']

# Replace Belgium with EU in Funder country
all_data['Funder Country'] = all_data['Funder Country'].replace({
    "Belgium": "EU"
})

# Filter out rows where funding is 0
all_data = all_data[all_data["Funding (USD)"] != 0]

# Remove rows with any NaN
# all_data = all_data.dropna()

# Replace NaNs in Abstract (BMBF)
all_data['Abstract'] = all_data['Abstract'].fillna('Not provided')

# Remove ERC/EC duplicates and keep only ERC rows
def filter_funders(group):
    # Check if both "ERC" and "European Commission" are in the "Funder" column of this group
    funders = group["Funder"].values
    if "ERC" in funders and "European Commission" in funders:
        # Keep only rows where "Funder" is "ERC"
        return group[group["Funder"] == "ERC"]
    else:
        # Otherwise, keep the group as is
        return group

# Combine EC and ERC
all_data['Funder'] = all_data['Funder'].replace({'ERC': 'EC/ERC', 'European Commission': 'EC/ERC'})

all_data = all_data.groupby("Title", group_keys=False).apply(filter_funders)

# Remove other duplicates
all_data = all_data.drop_duplicates(subset="Title", keep="first")

# Convert funding and years to integers
all_data['Funding (USD)'] = all_data['Funding (USD)'].round(0).astype(int)
# all_data['Start Year'] = all_data['Start Year'].round(0).astype(int)
# all_data['End Year'] = all_data['End Year'].round(0).astype(int)

# Remove rows with less than 50k funding
all_data = all_data[all_data["Funding (USD)"] > 50000]

# Sort everything by $$
all_data = all_data.sort_values(by="Funding (USD)", ascending=False)

# Split the "Fields of research" into lists
all_data["Fields of research"] = all_data["Fields of research"].str.split(";")

# Explode the list so each subfield is its own row
all_data_exploded = all_data.explode("Fields of research")

# Clean up whitespace in the subfield names
all_data_exploded["Fields of research"] = all_data_exploded["Fields of research"].str.strip()

subfields_to_keep = [
    "Atomic, Molecular and Optical Physics",
    "Classical Physics",
    "Condensed Matter Physics",
    "Mathematical Physics",
    "Nuclear and Plasma Physics",
    "Particle and High Energy Physics",
    "Quantum Physics",
    "Astronomical Sciences",
    "Space Sciences",
    "Synchrotrons and Accelerators"
]

# Filter to keep only desired subfields
filtered_data = all_data_exploded[all_data_exploded["Fields of research"].isin(subfields_to_keep)].copy()

rename_dict = {
    "Quantum Physics": "Quantum",
    "Condensed Matter Physics": "CondMat",
    "Atomic, Molecular and Optical Physics": "AMO",
    "Astronomical Sciences": "Astronomy",
    "Space Sciences": "Astronomy",
    "Nuclear and Plasma Physics": "Nuclear & Plasma"
}

# Replace subfield
filtered_data["Fields of research"] = filtered_data["Fields of research"].replace(rename_dict)

unique_subfields = sorted(filtered_data["Fields of research"].unique())


# Set plot theme
import plotly.io as pio

# Custom template
custom_template = pio.templates["plotly_white"]

# Update fonts
custom_template.layout.font = dict(
    family="Arial",  # Replace with your desired font
    size=12,               # Set the default font size
    color="black"          # Set the default font color
)

pio.templates.default = custom_template

app = dash.Dash(__name__)

app.layout = html.Div([
    # Year range slider at the top (or inside a tab—your choice)
    html.Div([
        html.H3("Select Year Range"),
        dcc.RangeSlider(
            id='year-range-slider',
            min=all_data['Year'].min(),
            max=all_data['Year'].max(),
            value=[all_data['Year'].min(), all_data['Year'].max()],
            marks={str(y): str(y) for y in sorted(all_data['Year'].unique())},
            step=None
        )
    ], style={'padding': '20px', 'width': '80%', 'margin': 'auto'}),

    dcc.Tabs([
        dcc.Tab(label="Overview", children=[
            html.Div([
                html.H3("Aggregated Funding by Funder"),
                dcc.Graph(id="funder-fig")
            ], style={'padding': '20px'})
        ]),
        dcc.Tab(label="Breakdown by Subfield & Funder", children=[
            html.Div([
                html.H3("Breakdown of Research Funding by Subfield for Every Funder"),
                dcc.Graph(id="breakdown-fig"),
                html.H3("Aggregated Funding by Research Subfield"),
                dcc.Graph(id="sub-fig")
            ], style={'padding': '20px'})
        ]),
        dcc.Tab(label="Geographical Analysis", children=[
            html.Div([
                html.H3("Aggregated Funding by Research Country"),
                dcc.Graph(id="country-fig"),
                html.H3("Aggregated Funding by Research Country (Map)"),
                dcc.Graph(id="choropleth-fig"),
            ], style={'padding': '20px'})
        ]),
        dcc.Tab(label="Top Grants by Subfield", children=[
            html.Div([
                html.H3("Top 30 Grants by Subfield"),
                html.Label("Select Subfield:"),
                dcc.Dropdown(
                    id="subfield-dropdown",
                    options=[{'label': sub, 'value': sub} for sub in unique_subfields],
                    value=unique_subfields[0],
                    clearable=False
                ),
                dcc.Graph(id="top20-table", style={'height': '1000px'})
            ], style={'padding': '20px'})
        ])
    ])
])

# Callback for the Main Figures

# Define subfields to keep and rename dict outside callbacks so they are easily reused
subfields_to_keep = [
    "Atomic, Molecular and Optical Physics",
    "Classical Physics",
    "Condensed Matter Physics",
    "Mathematical Physics",
    "Nuclear and Plasma Physics",
    "Particle and High Energy Physics",
    "Quantum Physics",
    "Astronomical Sciences",
    "Space Sciences",
    "Synchrotrons and Accelerators"
]

rename_dict = {
    "Quantum Physics": "Quantum",
    "Condensed Matter Physics": "CondMat",
    "Atomic, Molecular and Optical Physics": "AMO",
    "Astronomical Sciences": "Astronomy",
    "Space Sciences": "Astronomy",
    "Nuclear and Plasma Physics": "Nuclear & Plasma"
}

@app.callback(
    [
        Output("funder-fig", "figure"),
        Output("sub-fig", "figure"),
        Output("breakdown-fig", "figure"),
        Output("country-fig", "figure"),
        Output("choropleth-fig", "figure")
    ],
    Input("year-range-slider", "value")
)
def update_main_figures(year_range):
    """Filter data by the selected year range, then create and return all five figures."""
    start_year, end_year = year_range

    # 1) Filter the master DataFrame by the chosen years
    df_year_filtered = all_data[
        (all_data["Year"] >= start_year) & (all_data["Year"] <= end_year)
    ].copy()

    # -- FIGURE 1: Funder Bar Chart --
    agg_funding_funder = (
        df_year_filtered.groupby("Funder", as_index=False)["Funding (USD)"].sum()
    )
    fig_funder = px.bar(
        agg_funding_funder,
        x="Funder",
        y="Funding (USD)",
        text="Funding (USD)",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_funder.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    fig_funder.update_layout(
        autosize=False, width=800, height=600,
        plot_bgcolor='white', paper_bgcolor='white'
    )

    # 2) For subfield-based figures, explode the "Fields of research"
    df_year_filtered["Fields of research"] = df_year_filtered["Fields of research"].apply(
        lambda x: x if isinstance(x, list) else str(x).split(";")
    )
    all_data_exploded = df_year_filtered.explode("Fields of research")
    all_data_exploded["Fields of research"] = all_data_exploded["Fields of research"].str.strip()

    # Filter subfields
    filtered_data_sub = all_data_exploded[all_data_exploded["Fields of research"].isin(subfields_to_keep)].copy()
    filtered_data_sub["Fields of research"] = filtered_data_sub["Fields of research"].replace(rename_dict)

    # -- FIGURE 2: Aggregated Funding by Research Subfield (Bar + Pie) --
    agg_funding_subfield = (
        filtered_data_sub.groupby("Fields of research", as_index=False)["Funding (USD)"].sum()
    )
    total_funding = agg_funding_subfield["Funding (USD)"].sum()
    agg_funding_subfield["CustomLabel"] = agg_funding_subfield.apply(
        lambda row: f"{row['Fields of research']} ({row['Funding (USD)'] / total_funding * 100:.1f}%)", axis=1
    )

    fig_sub = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'xy'}, {'type': 'domain'}]]
    )
    # Bar
    fig_sub.add_trace(
        go.Bar(
            x=agg_funding_subfield["Fields of research"],
            y=agg_funding_subfield["Funding (USD)"],
            marker_color=px.colors.qualitative.Pastel[0],
            showlegend=False
        ),
        row=1, col=1
    )
    # Pie
    fig_sub.add_trace(
        go.Pie(
            labels=agg_funding_subfield["CustomLabel"],
            values=agg_funding_subfield["Funding (USD)"],
            marker_colors=px.colors.qualitative.Pastel,
            textinfo='none',
            showlegend=True
        ),
        row=1, col=2
    )
    fig_sub.update_layout(
        title_x=0.5,
        template="plotly_white",
        width=1400,
        height=600,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # -- FIGURE 3: Grouped Bar (Funding by Subfield & Funder) --
    agg_funding_subfield_funder = (
        filtered_data_sub.groupby(["Fields of research", "Funder"], as_index=False)["Funding (USD)"].sum()
    )
    fig_breakdown = px.bar(
        agg_funding_subfield_funder,
        x="Fields of research",
        y="Funding (USD)",
        color="Funder",
        barmode="group",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    # Remove bar text
    for trace in fig_breakdown.data:
        trace.text = None
        trace.texttemplate = None
        trace.textposition = None
    fig_breakdown.update_layout(
        autosize=False,
        width=1300,
        height=600,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # -- FIGURE 4: Bar Chart by Research Country --
    # Explode "Research Country" if it isn't already
    df_year_filtered["Research Country"] = df_year_filtered["Research Country"].apply(
        lambda x: x if isinstance(x, list) else str(x).split(";")
    )
    all_data_exploded_country = df_year_filtered.explode("Research Country")
    all_data_exploded_country["Research Country"] = all_data_exploded_country["Research Country"].str.strip()

    agg_funding_country = (
        all_data_exploded_country.groupby("Research Country", as_index=False)["Funding (USD)"].sum()
    )
    fig_country = px.bar(
        agg_funding_country,
        x="Research Country",
        y="Funding (USD)",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_country.update_traces(texttemplate=None)
    fig_country.update_layout(
        autosize=False,
        width=800,
        height=600,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # -- FIGURE 5: Choropleth --
    fig_choropleth = px.choropleth(
        agg_funding_country,
        locations="Research Country",
        locationmode="country names",
        color="Funding (USD)",
        hover_name="Research Country",
        color_continuous_scale="Blues"
    )
    fig_choropleth.update_layout(
        width=1200,
        height=800,
        margin=dict(l=50, r=50, t=100, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    fig_choropleth.update_geos(projection_type="robinson")

    return fig_funder, fig_sub, fig_breakdown, fig_country, fig_choropleth

# Callback for the table

@app.callback(
    Output("top20-table", "figure"),
    [
        Input("subfield-dropdown", "value"),
        Input("year-range-slider", "value")
    ]
)
def update_top_grants_table(selected_subfield, year_range):
    """Filter the data by (subfield + year range) and return a Table figure."""
    start_year, end_year = year_range

    # Filter main data
    df_year_filtered = all_data[
        (all_data["Year"] >= start_year) & (all_data["Year"] <= end_year)
    ].copy()

    # Explode "Fields of research" if needed
    df_year_filtered["Fields of research"] = df_year_filtered["Fields of research"].apply(
        lambda x: x if isinstance(x, list) else str(x).split(";")
    )
    df_exploded_sub = df_year_filtered.explode("Fields of research")
    df_exploded_sub["Fields of research"] = df_exploded_sub["Fields of research"].str.strip()

    # Apply the same subfields filtering + rename
    df_exploded_sub = df_exploded_sub[df_exploded_sub["Fields of research"].isin(subfields_to_keep)]
    df_exploded_sub["Fields of research"] = df_exploded_sub["Fields of research"].replace(rename_dict)

    # Now filter for the chosen subfield
    df_selected = df_exploded_sub[df_exploded_sub["Fields of research"] == selected_subfield].copy()

    # Sort by funding in descending order and take top 30
    df_top30 = df_selected.sort_values("Funding (USD)", ascending=False).head(30)

    # Build the table
    table_fig = go.Figure(data=[go.Table(
        header=dict(
            values=["<b>Title</b>", "<b>Abstract</b>", "<b>Funder</b>", "<b>Funding (USD)</b>"],
            fill_color="paleturquoise",
            align="left",
            font=dict(size=18)
        ),
        cells=dict(
            values=[
                df_top30["Title"],
                df_top30["Abstract"],
                df_top30["Funder"],
                df_top30["Funding (USD)"]
            ],
            fill_color="lavender",
            align="left",
            font=dict(size=16)
        )
    )])

    table_fig.update_layout(
        title_text=f"Top 30 Grants for Subfield: {selected_subfield} ({start_year}-{end_year})",
        title_x=0.5,
        template="plotly_white",
        height=1000
    )
    return table_fig

# ----------------------
# Run the app
# ----------------------
if __name__ == '__main__':
    # For local development, you might want to use debug mode.
    app.run_server(debug=True)
