#!/usr/bin/env python
# coding: utf-8

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


# Define path
data_dir = 'data'  # update with your actual path

excel_files = glob.glob(os.path.join(data_dir, '*.xlsx'))

# Select useful columns
columns_to_keep = ['Title', 'Abstract', 'Funding amount in USD', 'Start Year', 'End Year', 'Funder', 'Country of standardized research organization', 'Funder Country', 'Fields of Research (ANZSRC 2020)']


# Empty list
df_list = []

for file in excel_files:
    # Read Excel file into a DataFrame
    df = pd.read_excel(file, skiprows=1)

    # Select columns
    df = df.loc[:, [col for col in columns_to_keep if col in df.columns]]

    if 'Start date' in df.columns:
        df['Start date'] = pd.to_datetime(df['Start date'], errors='coerce')

    if 'End date' in df.columns:
        df['End date'] = pd.to_datetime(df['End date'], errors='coerce')

    # Remove duplicates, standardize text, etc.
    df = df.drop_duplicates()

    # Append to the list
    df_list.append(df)

# Concatenate all data
all_data = pd.concat(df_list, ignore_index=True)

# Rename columns
all_data = all_data.rename(columns={'Fields of Research (ANZSRC 2020)': 'Fields of research'})
all_data = all_data.rename(columns={'Funding amount in USD': 'Funding (USD)'})
all_data = all_data.rename(columns={'Start date': 'Start Date'})
all_data = all_data.rename(columns={'Country of standardized research organization': 'Research Country'})


# Clean up Fields of Research column (remove numeric codes, "Physical Sciences", semicolons, etc)
all_data['Fields of research'] = all_data['Fields of research'].str.replace('Physical Sciences;', '', regex=False)
all_data['Fields of research'] = all_data['Fields of research'].str.replace(r'\d+', '', regex=True)
all_data['Fields of research'] = all_data['Fields of research'].str.replace(r'^\s*;\s*', '', regex=True)
# Clean up extra whitespace
all_data['Fields of research'] = all_data['Fields of research'].str.replace(r'\s+', ' ', regex=True).str.strip()


# Clean up Funder column
all_data['Funder'] = all_data['Funder'].replace({
    "Directorate for Mathematical & Physical Sciences": "NSF",
    "European Research Council": "ERC"
})

# Replace Belgium with EU in Funder country
all_data['Funder Country'] = all_data['Funder Country'].replace({
    "Belgium": "EU"
})


# Filter out rows where funding is 0
all_data = all_data[all_data["Funding (USD)"] != 0]

# Remove rows with any NaN
all_data = all_data.dropna()


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

all_data = all_data.groupby("Title", group_keys=False).apply(filter_funders)

# Remove other duplicates
all_data = all_data.drop_duplicates(subset="Title", keep="first")


# Convert funding and years to integers
all_data['Funding (USD)'] = all_data['Funding (USD)'].round(0).astype(int)
all_data['Start Year'] = all_data['Start Year'].round(0).astype(int)
all_data['End Year'] = all_data['End Year'].round(0).astype(int)

# Remove rows with less than 50k funding
all_data = all_data[all_data["Funding (USD)"] > 50000]

# Sort everything by $$
all_data = all_data.sort_values(by="Funding (USD)", ascending=False)


# Save as CSV
all_data.to_csv('data/all_data.csv', index=False)

# Save as Excel
all_data.to_excel('data/all_data.xlsx', index=False)

# Aggregate funding data

agg_funding_funder = (
    all_data.groupby("Funder", as_index=False)["Funding (USD)"]
    .sum()
)

# Bar chart for aggregated funding by funder
fig_funder = px.bar(
    agg_funding_funder,
    x="Funder",
    y="Funding (USD)",
    text="Funding (USD)",
    template="plotly_white",
    color_discrete_sequence=px.colors.qualitative.Pastel  # Use a pastel color palette
)
fig_funder.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
fig_funder.update_layout(autosize=False, width=800, height=600,
                   plot_bgcolor='white', paper_bgcolor='white')
# fig_funder.show()


# Ensure that the "Fields of research" column is split into lists.
all_data["Fields of research"] = all_data["Fields of research"].str.split(";")

# Explode the list
all_data_exploded = all_data.explode("Fields of research")

# Clean up extra whitespace.
all_data_exploded["Fields of research"] = all_data_exploded["Fields of research"].str.strip()

# Define subfields to keep
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

# Filter the exploded df and include only the desired subfields
filtered_data = all_data_exploded[all_data_exploded["Fields of research"].isin(subfields_to_keep)]

rename_dict = {
    "Quantum Physics": "Quantum",
    "Condensed Matter Physics": "CondMat",
    "Atomic, Molecular and Optical Physics": "AMO",
    "Astronomical Sciences": "Astronomy",
    "Space Sciences": "Astronomy",
    "Nuclear and Plasma Physics": "Nuclear & Plasma"
}

filtered_data["Fields of research"] = filtered_data["Fields of research"].replace(rename_dict)

# Aggregated Funding by Research Subfield
agg_funding_subfield = (
    filtered_data.groupby("Fields of research", as_index=False)["Funding (USD)"]
    .sum()
)


# Compute custom labels for pie chart
total_funding = agg_funding_subfield["Funding (USD)"].sum()
agg_funding_subfield["CustomLabel"] = agg_funding_subfield.apply(
    lambda row: f"{row['Fields of research']} ({row['Funding (USD)'] / total_funding * 100:.1f}%)", axis=1
)

fig_sub = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'xy'}, {'type': 'domain'}]]
)

# Set showlegend=False to remove the extra legend entry for the bar chart
bar_trace = go.Bar(
    x=agg_funding_subfield["Fields of research"],
    y=agg_funding_subfield["Funding (USD)"],
    marker_color=px.colors.qualitative.Pastel[0],
    showlegend=False  # This prevents an extra legend entry for the bar chart
)
fig_sub.add_trace(bar_trace, row=1, col=1)

# Use the custom labels for the pie chart so that the legend shows the subfield plus percentage
pie_trace = go.Pie(
    labels=agg_funding_subfield["CustomLabel"],
    values=agg_funding_subfield["Funding (USD)"],
    marker_colors=px.colors.qualitative.Pastel,
    textinfo='none',  # Do not display any text on the pie slices.
    showlegend=True   # The legend will display the custom labels.
)
fig_sub.add_trace(pie_trace, row=1, col=2)

fig_sub.update_layout(
    title_text="Aggregated Funding by Research Subfield",
    title_x=0.5,  # Center the title
    template="plotly_white",
    width=1400,
    height=600,
    plot_bgcolor='white',
    paper_bgcolor='white'
)

# fig_sub.show()

# Group by both "Fields of research" and "Funder" and sum the funding amounts
agg_funding_subfield_funder = (
    filtered_data.groupby(["Fields of research", "Funder"], as_index=False)["Funding (USD)"]
    .sum()
)

# Grouped bar chart showingm the funding breakdown by funder (for each subfield)
fig_breakdown = px.bar(
    agg_funding_subfield_funder,
    x="Fields of research",
    y="Funding (USD)",
    color="Funder",
    barmode="group",
    template="plotly_white",
    color_discrete_sequence=px.colors.qualitative.Pastel
)

# Remove text annotations from the top of bars
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

# fig_breakdown.show()

# pivot = agg_funding_subfield_funder.pivot(index="Funder",
#                                            columns="Fields of research",
#                                            values="Funding (USD)").fillna(0)

# # Compute percentages per funder
# pivot_percent = pivot.div(pivot.sum(axis=1), axis=0)

# plt.figure(figsize=(12, 8))
# sns.heatmap(pivot_percent, annot=True, cmap="YlGnBu", fmt=".1%")
# plt.title("Percentage Distribution of Funding by Research Subfield per Funder")
# plt.xlabel("Fields of research")
# plt.ylabel("Funder")
# plt.show()


all_data["Research Country"] = all_data["Research Country"].str.split(";")

# Explode the list so that each country appears in its own row
all_data_exploded_country = all_data.explode("Research Country")

# Clean up extra whitespace
all_data_exploded_country["Research Country"] = all_data_exploded_country["Research Country"].str.strip()


# Aggregating funding by Country
agg_funding_country = (
    all_data_exploded_country
    .groupby("Research Country", as_index=False)["Funding (USD)"]
    .sum()
)


fig_country = px.bar(
    agg_funding_country,
    x="Research Country",
    y="Funding (USD)",
    template="plotly_white",
    color_discrete_sequence=px.colors.qualitative.Pastel
)

# Remove any text annotations on the bars
fig_country.update_traces(texttemplate=None)

fig_country.update_layout(
    autosize=False,
    width=800,
    height=600,
    plot_bgcolor='white',
    paper_bgcolor='white'
)

# fig_country.show()

# Split the "Research Country" column into lists
filtered_data["Research Country"] = filtered_data["Research Country"].str.split(";")

# Explode the df so that each country appears in its own row
filtered_data_exploded_country = filtered_data.explode("Research Country")

# Clean up extra whitespace from each country name
filtered_data_exploded_country["Research Country"] = filtered_data_exploded_country["Research Country"].str.strip()

# Extract a sorted list of unique subfields from the "Fields of research" column
unique_subfields = sorted(filtered_data_exploded_country["Fields of research"].unique())
n_subfields = len(unique_subfields)

# Decide grid dimensions for subplots
ncols = 3
nrows = math.ceil(n_subfields / ncols)


# Create a subplot grid with one bar chart per subfield
fig_breakdown_country = make_subplots(
    rows=nrows, cols=ncols,
    subplot_titles=unique_subfields,
    horizontal_spacing=0.1, vertical_spacing=0.15
)

# Loop over each subfield to create a corresponding bar chart
for i, subfield in enumerate(unique_subfields):
    # Filter the data for the current subfield.
    df_sub = filtered_data_exploded_country[filtered_data_exploded_country["Fields of research"] == subfield]

    # Aggregate funding by "Research Country" for this subfield
    agg_country = df_sub.groupby("Research Country", as_index=False)["Funding (USD)"].sum()

    # Determine the subplot's row and column
    row = i // ncols + 1
    col = i % ncols + 1

    # Add a bar chart trace for this subfield
    fig_breakdown_country.add_trace(
        go.Bar(
            x=agg_country["Research Country"],
            y=agg_country["Funding (USD)"],
            marker_color=px.colors.qualitative.Pastel[0]
        ),
        row=row, col=col
    )

    fig_breakdown_country.update_xaxes(title_text="Country", row=row, col=col)
    fig_breakdown_country.update_yaxes(title_text="Funding (USD)", row=row, col=col)


fig_breakdown_country.update_layout(
    template="plotly_white",
    showlegend=False,
    height=500 * nrows,  # Adjust height based on the number of rows.
    width=1400,
    plot_bgcolor='white',
    paper_bgcolor='white'
)

# fig_breakdown_country.show()


fig_choropleth = px.choropleth(
    agg_funding_country,
    locations="Research Country",        # Country names or ISO codes
    locationmode="country names",          # or "ISO-3" if you have country codes
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

fig_choropleth.update_geos(
    projection_type="robinson"
)

# fig_choropleth.show()

# Save graphs
# fig_funder.write_image("/Users/lbs1678/Desktop/Bibliometrics/FundingData/Graphs/aggregated_funding_by_funder.png", scale=2)
# fig_sub.write_image("/Users/lbs1678/Desktop/Bibliometrics/FundingData/Graphs/aggregated_funding_by_subfield.png", scale=2)
# fig_breakdown.write_image("/Users/lbs1678/Desktop/Bibliometrics/FundingData/Graphs/breakdown_by_subfield_and_funder.jpeg", scale=2)
# fig_country.write_image("/Users/lbs1678/Desktop/Bibliometrics/FundingData/Graphs/funding_by_research_country.png", scale=2)
# fig_choropleth.write_image("/Users/lbs1678/Desktop/Bibliometrics/FundingData/Graphs/map_funding_by_country.png", scale=2)


# Get a sorted list of unique subfields
unique_subfields = sorted(filtered_data["Fields of research"].unique())

for subfield in unique_subfields:
    # Filter the data for the current subfield
    df_sub = filtered_data[filtered_data["Fields of research"] == subfield].copy()

    # Sort the data by funding in descending order and take the top X grants
    df_top20 = df_sub.sort_values("Funding (USD)", ascending=False).head(20)

    # Create a table with columns for Title, Funder, and Funding (USD)
    fig_top = go.Figure(data=[go.Table(
        header=dict(
            values=["<b>Title</b>", "<b>Abstract</b>", "<b>Funder</b>", "<b>Funding (USD)</b>"],
            fill_color="paleturquoise",
            align="left"
        ),
        cells=dict(
            values=[df_top20["Title"], df_top20["Abstract"], df_top20["Funder"], df_top20["Funding (USD)"]],
            fill_color="lavender",
            align="left"
        )
    )])

    # Update layout with a global title for the table
    fig_top.update_layout(
        title_text=f"Top 20 Grants for Subfield: {subfield}",
        title_x=0.5,
        template="plotly_white"
    )

    # Display the table
    # fig_top.show()

# Build Dash app
app = dash.Dash(__name__)
server = app.server  # for deployment

# App layout with multiple tabs
app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label="Overview", children=[
            html.Div([
                html.H3("Aggregated Funding by Funder"),
                dcc.Graph(figure=fig_funder)
            ], style={'padding': '20px'})
        ]),
        dcc.Tab(label="Breakdown by Subfield & Funder", children=[
            html.Div([
                html.H3("Breakdown of Research Funding by Subfield for Every Funder"),
                dcc.Graph(figure=fig_breakdown),
                html.H3("Aggregated Funding by Research Subfield"),
                dcc.Graph(figure=fig_sub)
            ], style={'padding': '20px'})
        ]),
        dcc.Tab(label="Geographical Analysis", children=[
            html.Div([
                html.H3("Aggregated Funding by Research Country"),
                dcc.Graph(figure=fig_country),
                html.H3("Aggregated Funding by Research Country (Choropleth)"),
                dcc.Graph(figure=fig_choropleth),
                html.H3("Aggregated Funding by Research Country (By subfield)"),
                dcc.Graph(figure=fig_breakdown_country),
            ], style={'padding': '20px'})
        ]),
        dcc.Tab(label="Top Grants by Subfield", children=[
            html.Div([
                html.H3("Top 20 Grants by Subfield"),
                html.Label("Select Subfield:"),
                dcc.Dropdown(
                    id="subfield-dropdown",
                    options=[{'label': sub, 'value': sub} for sub in unique_subfields],
                    value=unique_subfields[0],
                    clearable=False
                ),
                dcc.Graph(id="top20-table")
            ], style={'padding': '20px'})
        ])
    ])
])


# Callback for updating the Top 20 Grants Table by Subfield
@app.callback(
    Output("top20-table", "figure"),
    Input("subfield-dropdown", "value")
)
def update_top20_table(selected_subfield):
    # Filter data for the selected subfield
    df_sub = filtered_data[filtered_data["Fields of research"] == selected_subfield].copy()
    # Sort by funding in descending order and take top 20
    df_top20 = df_sub.sort_values("Funding (USD)", ascending=False).head(20)

    table_fig = go.Figure(data=[go.Table(
        header=dict(
            values=["<b>Title</b>", "<b>Abstract</b>", "<b>Funder</b>", "<b>Funding (USD)</b>"],
            fill_color="paleturquoise",
            align="left",
            font=dict(size=18)
        ),
        cells=dict(
            values=[
                df_top20["Title"],
                df_top20["Abstract"],
                df_top20["Funder"],
                df_top20["Funding (USD)"]
            ],
            fill_color="lavender",
            align="left",
            font=dict(size=16)
        )
    )])

    # Update layout to increase the table's size
    table_fig.update_layout(
        title_text=f"Top 20 Grants for Subfield: {selected_subfield}",
        title_x=0.5,
        template="plotly_white",
        height=1000   # Increase height as needed
    )

    return table_fig


# Run the App
if __name__ == '__main__':
    # For local development, you might want to use debug mode.
    app.run_server(debug=True)
