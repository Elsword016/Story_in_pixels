import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns 
import matplotlib.pyplot as plt  
import matplotlib.dates as mdates
from plotly.subplots import make_subplots
import plotly.express as px
st.set_page_config(
    page_title="A/B Test Dashboard",
    page_icon="📊", # You can use an emoji or a URL to an image
    layout="wide" # "wide" is often good for dashboards
)
st.title("📊 A/B Test Campaign analysis of T-Shirt Website")

@st.cache_data # Use st.cache_data for data, st.cache_resource for ML models etc.
def load_data(control_path, test_path,kpi_summary_path):
    ctrl_grp = pd.read_csv(control_path, delimiter=';')
    test_grp = pd.read_csv(test_path, delimiter=';')
    kpi_summary = pd.read_csv(kpi_summary_path, delimiter=',')
    # --- Perform your initial data cleaning and feature engineering here ---
    # Example: Convert 'Date' to datetime, handle NaNs, calculate daily metrics
    ctrl_grp['Date'] = pd.to_datetime(ctrl_grp['Date'], format='%d.%m.%Y')
    test_grp['Date'] = pd.to_datetime(test_grp['Date'], format='%d.%m.%Y')
    # ... (your existing data processing for ctrl_grp and test_grp) ...
    # ... (calculate CTR, CR, CPA, etc. for each group if not already done) ...
    return ctrl_grp, test_grp,kpi_summary

@st.cache_data
def load_combined_data(filepath):
    data = pd.read_csv(filepath)
    # Convert 'Date' column to datetime objects
    # The format '%d.%m.%Y' matches "1.08.2019"
    data['Date'] = pd.to_datetime(data['Date'], format='%d.%m.%Y')
    # Your CSV already has CTR, CR, etc. as fractions.
    # If you want to display them as percentages in the plot (e.g., 0.08 -> 8%),
    # you might multiply by 100 here or adjust labels accordingly.
    # For simplicity, let's assume the function will plot the raw fractional values,
    # and the y-axis labels will indicate if they are percentages.
    # Example: If CTR is 0.08, metric1_label='CTR (%)' implies the axis shows 0.08 as 8%.
    # If you want the axis to show 8, then:
    # data['CTR'] = data['CTR'] * 100
    # data['CR'] = data['CR'] * 100
    return data

def create_conversion_sankey(ctrl_grp, test_grp):
    # Calculate means for each step
    ctrl_means = {
        'Impressions': ctrl_grp['# of Impressions'].mean(),
        'Clicks': ctrl_grp['# of Website Clicks'].mean(),
        'Searches': ctrl_grp['# of Searches'].mean(),
        'Views': ctrl_grp['# of View Content'].mean(),
        'Cart': ctrl_grp['# of Add to Cart'].mean(),
        'Purchase': ctrl_grp['# of Purchase'].mean()
    }
    
    test_means = {
        'Impressions': test_grp['# of Impressions'].mean(),
        'Clicks': test_grp['# of Website Clicks'].mean(),
        'Searches': test_grp['# of Searches'].mean(),
        'Views': test_grp['# of View Content'].mean(),
        'Cart': test_grp['# of Add to Cart'].mean(),
        'Purchase': test_grp['# of Purchase'].mean()
    }

    # Calculate conversion rates for labels
    ctrl_rates = {
        'CTR': (ctrl_means['Clicks'] / ctrl_means['Impressions']) * 100,
        'Click_to_Search': (ctrl_means['Searches'] / ctrl_means['Clicks']) * 100,
        'Search_to_View': (ctrl_means['Views'] / ctrl_means['Searches']) * 100,
        'View_to_Cart': (ctrl_means['Cart'] / ctrl_means['Views']) * 100,
        'Cart_to_Purchase': (ctrl_means['Purchase'] / ctrl_means['Cart']) * 100
    }
    
    test_rates = {
        'CTR': (test_means['Clicks'] / test_means['Impressions']) * 100,
        'Click_to_Search': (test_means['Searches'] / test_means['Clicks']) * 100,
        'Search_to_View': (test_means['Views'] / test_means['Searches']) * 100,
        'View_to_Cart': (test_means['Cart'] / test_means['Views']) * 100,
        'Cart_to_Purchase': (test_means['Purchase'] / test_means['Cart']) * 100
    }

    # Define nodes
    labels = [
        f'Impressions\nControl\n({ctrl_means["Impressions"]:,.0f})',
        f'Clicks\nControl\n({ctrl_means["Clicks"]:,.0f})',
        f'Searches\nControl\n({ctrl_means["Searches"]:,.0f})',
        f'Views\nControl\n({ctrl_means["Views"]:,.0f})',
        f'Cart\nControl\n({ctrl_means["Cart"]:,.0f})',
        f'Purchase\nControl\n({ctrl_means["Purchase"]:,.0f})',
        f'Impressions\nTest\n({test_means["Impressions"]:,.0f})',
        f'Clicks\nTest\n({test_means["Clicks"]:,.0f})',
        f'Searches\nTest\n({test_means["Searches"]:,.0f})',
        f'Views\nTest\n({test_means["Views"]:,.0f})',
        f'Cart\nTest\n({test_means["Cart"]:,.0f})',
        f'Purchase\nTest\n({test_means["Purchase"]:,.0f})'
    ]

    # Define links
    source = [
        # Control group links
        0, 1, 2, 3, 4,
        # Test group links
        6, 7, 8, 9, 10
    ]
    
    target = [
        # Control group links
        1, 2, 3, 4, 5,
        # Test group links
        7, 8, 9, 10, 11
    ]
    
    value = [
        # Control group flows
        ctrl_means['Clicks'],
        ctrl_means['Searches'],
        ctrl_means['Views'],
        ctrl_means['Cart'],
        ctrl_means['Purchase'],
        # Test group flows
        test_means['Clicks'],
        test_means['Searches'],
        test_means['Views'],
        test_means['Cart'],
        test_means['Purchase']
    ]

    # Create hover text with conversion rates
    hover_text = [
        f"Control: {ctrl_rates['CTR']:.1f}%",
        f"Control: {ctrl_rates['Click_to_Search']:.1f}%",
        f"Control: {ctrl_rates['Search_to_View']:.1f}%",
        f"Control: {ctrl_rates['View_to_Cart']:.1f}%",
        f"Control: {ctrl_rates['Cart_to_Purchase']:.1f}%",
        f"Test: {test_rates['CTR']:.1f}%",
        f"Test: {test_rates['Click_to_Search']:.1f}%",
        f"Test: {test_rates['Search_to_View']:.1f}%",
        f"Test: {test_rates['View_to_Cart']:.1f}%",
        f"Test: {test_rates['Cart_to_Purchase']:.1f}%"
    ]

    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=["rgba(31, 119, 180, 0.8)"]*6 + ["rgba(255, 127, 14, 0.8)"]*6  # Blue for Control, Orange for Test
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_text,
            color=["rgba(31, 119, 180, 0.3)"]*5 + ["rgba(255, 127, 14, 0.3)"]*5
        )
    )])

    # Update layout
    fig.update_layout(
        title_text="Conversion Funnel: Control vs Test Campaign",
        font_size=10,
        height=600,
        width=1200,
        showlegend=False
    )

    return fig

ctrl_grp_df, test_grp_df,kpi_summary_df = load_data('AB_Testing/control_group.csv', 'AB_testing/test_group.csv','AB_testing/kpi_summary_row1.csv')
#combined_data = pd.concat([ctrl_grp_df, test_grp_df], ignore_index=True)
combined_data = load_combined_data('AB_Testing/combined_data.csv')

#st.markdown("**Date:** October 26, 2023")
#st.markdown("**To:** Marketing Leadership, Sales Team")
#st.markdown("**From:** Data Analytics Team")
#st.markdown("**Subject:** Key Findings & Recommendations from Recent A/B Campaign Test")

#st.markdown("---")

st.subheader("Executive Summary")
st.markdown("Our recent A/B test reveals a critical divergence in campaign performance: the **Test Campaign significantly boosts ad engagement (CTR +66.5%) but results in lower actual conversions (CR -12.1%)** compared to the Control Campaign. This suggests the Test Campaign is excellent at attracting attention but struggles to convert that attention into desired actions.")
st.markdown("**Immediate action should focus on optimizing the post-click experience for the Test Campaign (landing page alignment) and refining its audience targeting to improve conversion quality.**")
#link the pdf file
st.markdown("[Download the static dashboard](D:/portfolio/end-to-end/AB_testing/data_dashboard.pdf)") 
st.markdown("---") 

st.subheader("Key Performance Indicators (KPIs)") 
col1,col2,col3 = st.columns(3) 

with col1:
    # Filter for the 'Conversion Rate (CR)' row
    cr_data = kpi_summary_df[kpi_summary_df['Metric'] == 'Conversion Rate (CR)']
    if not cr_data.empty:
        cr_test_value = cr_data['Test'].iloc[0]
        cr_lift_value = cr_data['Lift (%)'].iloc[0]
        # Ensure lift is a string and ends with '%' for st.metric's delta, or convert to float for calculation
        # For st.metric, delta usually expects a numerical value for comparison or a string.
        # If lift is already like "-12.06%", we can use it.
        # If it's a number, we might format it. Let's assume it's a string like in the CSV.

        st.metric(
            label="Conversion Rate (CR) - Test",
            value=cr_test_value, # Should be like "8.64%"
            delta=f"{cr_lift_value}", # Should be like "-12.06%"
            delta_color="inverse" # "inverse" because a negative lift is bad for CR
        )
        # Optionally, display the control value and p-value
        control_val = cr_data['Control'].iloc[0]
        #p_val = cr_data['P-value'].iloc[0]
        significant = cr_data['Significant? (✓/✗)'].iloc[0]
        st.caption(f"Control: {control_val} | Statistically Significant: {significant}")
    else:
        st.metric(label="Conversion Rate (CR) - Test", value="N/A")

with col2:
    # Filter for the 'Conversion Rate (CR)' row
    ctr_data = kpi_summary_df[kpi_summary_df['Metric'] == 'Click-Through Rate (CTR)']
    if not ctr_data.empty:
        ctr_test_value = ctr_data['Test'].iloc[0]
        ctr_lift_value = ctr_data['Lift (%)'].iloc[0]
        # Ensure lift is a string and ends with '%' for st.metric's delta, or convert to float for calculation
        # For st.metric, delta usually expects a numerical value for comparison or a string.
        # If lift is already like "-12.06%", we can use it.
        # If it's a number, we might format it. Let's assume it's a string like in the CSV.

        st.metric(
            label="Click-Through Rate (CTR) - Test",
            value=ctr_test_value, # Should be like "8.64%"
            delta=f"{ctr_lift_value}", # Should be like "-12.06%"
            delta_color="inverse" # "inverse" because a negative lift is bad for CR
        )
        # Optionally, display the control value and p-value
        control_val = ctr_data['Control'].iloc[0]
        #p_val = cr_data['P-value'].iloc[0]
        significant = ctr_data['Significant? (✓/✗)'].iloc[0]
        st.caption(f"Control: {control_val} | Statistically Significant: {significant}")
    else:
        st.metric(label="Conversion Rate (CR) - Test", value="N/A")

with col3:
    # Filter for the 'Conversion Rate (CR)' row
    cpa_data = kpi_summary_df[kpi_summary_df['Metric'] == 'Cost Per Acq. (CPA)']
    if not cpa_data.empty:
        cpa_test_value = cpa_data['Test'].iloc[0]
        cpa_lift_value = cpa_data['Lift (%)'].iloc[0]
        # Ensure lift is a string and ends with '%' for st.metric's delta, or convert to float for calculation
        # For st.metric, delta usually expects a numerical value for comparison or a string.
        # If lift is already like "-12.06%", we can use it.
        # If it's a number, we might format it. Let's assume it's a string like in the CSV.

        st.metric(
            label="Cost Per Action (CPA) - Test",
            value=cpa_test_value, # Should be like "8.64%"
            delta=f"{cpa_lift_value}", # Should be like "-12.06%"
            delta_color="inverse" # "inverse" because a negative lift is bad for CR
        )
        # Optionally, display the control value and p-value
        control_val = cpa_data['Control'].iloc[0]
        #p_val = cr_data['P-value'].iloc[0]
        significant = cpa_data['Significant? (✓/✗)'].iloc[0]
        st.caption(f"Control: {control_val} | Statistically Significant: {significant}")
    else:
        st.metric(label="Cost Per Action (CPA) - Test", value="N/A")
st.markdown("---") 

st.subheader("KPIs Analysis")
st.markdown("The A/B test ran through September 2019, comparing the 'Control Campaign' against the 'Test Campaign.' Statistical analysis of key performance indicators (KPIs) shows:")
st.markdown("#### 1. Click-Through Rate (CTR) - *Getting Attention*")
st.markdown("**Test Campaign:** **8.09% CTR**")
st.markdown("**Control Campaign:** 4.86% CTR")
st.markdown("**Lift:** **+66.54%** in favor of the Test Campaign.")
st.markdown("**Significance:** This improvement is statistically significant (p < 0.001).")
st.markdown("*Implication:* The Test Campaign's ad creative and messaging are substantially more effective at capturing user interest and driving clicks.")

st.markdown("#### 2. Conversion Rate (CR) - *Driving Action*")
st.markdown("**Test Campaign:** 8.64% CR")
st.markdown("**Control Campaign:** 9.83% CR")
st.markdown("**Lift:** -12.06% for the Test Campaign.")
st.markdown("**Significance:** This decrease is statistically significant (p < 0.001).")
st.markdown("*Implication:* Despite attracting more clicks, the Test Campaign is less effective at converting those clicks into actual purchases or desired outcomes.")
st.markdown("---")
def plot_funnel_dropoff_rates(ctrl_grp, test_grp):
    # Calculate funnel metrics for both groups
    def calculate_rates(data):
        # Calculate mean for impressions, clicks, etc.
        impressions = data['# of Impressions'].sum() # Or .mean() depending on your aggregation logic
        clicks = data['# of Website Clicks'].sum()
        searches = data['# of Searches'].sum()
        views = data['# of View Content'].sum()
        cart = data['# of Add to Cart'].sum()
        purchase = data['# of Purchase'].sum()

        return {
            'CTR': (clicks / impressions) * 100 if impressions > 0 else 0,
            'Click_to_Search': (searches / clicks) * 100 if clicks > 0 else 0,
            'Search_to_View': (views / searches) * 100 if searches > 0 else 0,
            'View_to_Cart': (cart / views) * 100 if views > 0 else 0,
            'Cart_to_Purchase': (purchase / cart) * 100 if cart > 0 else 0
        }

    # Get rates for both groups
    ctrl_rates_dict = calculate_rates(ctrl_grp)
    test_rates_dict = calculate_rates(test_grp)

    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        'Step': list(ctrl_rates_dict.keys()) * 2,
        'Rate': list(ctrl_rates_dict.values()) + list(test_rates_dict.values()),
        'Campaign': ['Control'] * len(ctrl_rates_dict) + ['Test'] * len(test_rates_dict)
    })

    # Create the plot using an explicit figure and axes
    fig, ax = plt.subplots(figsize=(12, 7)) # Changed: Create fig and ax
    
    # Create grouped bar chart
    sns.barplot(x='Step', y='Rate', hue='Campaign', data=plot_data,
                palette=['blue', 'orange'], ax=ax) # Changed: Pass ax

    # Customize the plot
    ax.set_title('Funnel Step Conversion Rates: Control vs Test', pad=20, fontsize=14) # Changed: Use ax
    ax.set_xlabel('Funnel Step', fontsize=12) # Changed: Use ax
    ax.set_ylabel('Conversion Rate (%)', fontsize=12) # Changed: Use ax
    
    # Rotate x-axis labels for better readability
    ax.tick_params(axis='x', rotation=45, labelsize=10) # Changed: Use ax
    ax.tick_params(axis='y', labelsize=10)
    
    # Add value labels on bars
    for container in ax.containers: # Changed: Use ax.containers
        ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=9) # Changed: Use ax.bar_label

    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7) # Changed: Use ax
    
    # Adjust layout to prevent label cutoff
    fig.tight_layout() # Changed: Use fig.tight_layout()
    
    # Remove plt.show() as Streamlit handles rendering
    # plt.show() # Removed

    # Print detailed comparison (this will print to console, not the Streamlit app)
    # If you want this in the app, return it and use st.text or st.dataframe
    # print("\nDetailed Funnel Step Comparison:")
    # print("-" * 60)
    # print(f"{'Step':<20} {'Control':>10} {'Test':>10} {'Difference':>12}")
    # print("-" * 60)
    
    # for step in ctrl_rates_dict.keys():
    #     diff = test_rates_dict[step] - ctrl_rates_dict[step]
    #     print(f"{step:<20} {ctrl_rates_dict[step]:>9.1f}% {test_rates_dict[step]:>9.1f}% {diff:>11.1f}%")

    return fig

st.subheader("Funnel performance analysis")
funnel_col1, funnel_col2 = st.columns(2) 

with funnel_col1:
    st.markdown("#### Conversion Funnel Sankey Diagram")
    fig = create_conversion_sankey(ctrl_grp_df, test_grp_df) 
    st.plotly_chart(fig,use_container_width=True ) 

with funnel_col2: 
    st.markdown("#### Funnel Step Conversion Rate")
    funnel_bar_fig = plot_funnel_dropoff_rates(ctrl_grp_df, test_grp_df) 
    st.pyplot(funnel_bar_fig) 

st.markdown("The Sankey diagram visually confirms the CTR and CR disparity, showing a wider initial flow for **Clicks** in the Test group but a proportionally larger drop-off at subsequent stages leading to **Purchase**.")
st.markdown("The Funnel Step Conversion Rate bar chart is the most diagnostic for identifying where to focus optimization efforts for the Test campaign. The Test campaign is excellent at the very top of the funnel (getting clicks).It experiences significant leakage in the middle of the funnel, particularly from Search to View and, most critically, from View Content to Add to Cart.However, if users from the Test campaign reach the cart, they convert to purchase at a higher rate than Control.")

def plot_combined_metrics_plotly(combined_data: pd.DataFrame, metric1_name: str, metric2_name: str, title: str, metric1_label: str = None, metric2_label: str = None):
    """
    Plots two metrics over time for Control and Test campaigns on a dual-axis chart using Plotly.

    Args:
        combined_data (pd.DataFrame): DataFrame containing 'Date', 'Campaign Name',
                                      metric1_name, and metric2_name columns.
                                      'Date' should be datetime objects.
        metric1_name (str): Column name of the first metric.
        metric2_name (str): Column name of the second metric.
        title (str): Title of the plot.
        metric1_label (str, optional): Display label for the first metric's y-axis. Defaults to metric1_name.
        metric2_label (str, optional): Display label for the second metric's y-axis. Defaults to metric2_name.

    Returns:
        plotly.graph_objects.Figure: The Plotly figure object.
    """
    if metric1_label is None:
        metric1_label = metric1_name
    if metric2_label is None:
        metric2_label = metric2_name

    # Ensure 'Date' is datetime
    if not pd.api.types.is_datetime64_any_dtype(combined_data['Date']):
        try:
            combined_data['Date'] = pd.to_datetime(combined_data['Date'], dayfirst=True)
        except ValueError:
            combined_data['Date'] = pd.to_datetime(combined_data['Date'], format='%d.%m.%Y', errors='coerce')
        except Exception as e:
            print(f"Error converting 'Date' column in plot_combined_metrics_plotly: {e}")
            # Potentially return an empty figure or raise error if Date is crucial and unparseable
            return go.Figure()


    # Filter data for Control and Test campaigns and sort by date
    control_data = combined_data[combined_data['Campaign Name'] == 'Control Campaign'].sort_values(by='Date')
    test_data = combined_data[combined_data['Campaign Name'] == 'Test Campaign'].sort_values(by='Date')

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Define colors for metrics (can be customized)
    color_metric1_control = '#4287f5' # Bright blue that works well on dark backgrounds
    color_metric1_test = 'deepskyblue'    # Example color
    color_metric2_control = 'red'  # Example color
    color_metric2_test = 'salmon'     # Example color


    # --- Plot Metric 1 ---
    # Control
    fig.add_trace(
        go.Scatter(
            x=control_data['Date'],
            y=control_data[metric1_name],
            name=f'{metric1_label} (Control)',
            mode='lines',
            line=dict(color=color_metric1_control, width=2)
        ),
        secondary_y=False,
    )
    # Test
    fig.add_trace(
        go.Scatter(
            x=test_data['Date'],
            y=test_data[metric1_name],
            name=f'{metric1_label} (Test)',
            mode='lines',
            line=dict(color=color_metric1_test, width=2, dash='dash')
        ),
        secondary_y=False,
    )

    # --- Plot Metric 2 ---
    # Control
    fig.add_trace(
        go.Scatter(
            x=control_data['Date'],
            y=control_data[metric2_name],
            name=f'{metric2_label} (Control)',
            mode='lines',
            line=dict(color=color_metric2_control, width=2)
        ),
        secondary_y=True,
    )
    # Test
    fig.add_trace(
        go.Scatter(
            x=test_data['Date'],
            y=test_data[metric2_name],
            name=f'{metric2_label} (Test)',
            mode='lines',
            line=dict(color=color_metric2_test, width=2, dash='dash')
        ),
        secondary_y=True,
    )

    # --- Update layout ---
    fig.update_layout(
        title_text=title,
        title_x=0.5, # Center title
        xaxis_title='Date',
        legend_title_text='Metrics',
        legend=dict(
            orientation="h", # Horizontal legend
            yanchor="bottom",
            y=1.02, # Position above plot
            xanchor="right",
            x=1
        ),
        autosize=True,
        # height=500 # Optionally set height
    )

    # Set y-axes titles
    fig.update_yaxes(title_text=metric1_label, secondary_y=False, title_font=dict(color=color_metric1_control))
    fig.update_yaxes(title_text=metric2_label, secondary_y=True, title_font=dict(color=color_metric2_control))
    
    # Optional: Format x-axis ticks for dates (Plotly often does this well automatically)
    fig.update_xaxes(tickformat='%Y-%m-%d', tickangle=30)

    # Optional: Add gridlines if desired (Plotly has them by default but can be customized)
    # fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', secondary_y=False)
    # fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', secondary_y=True)


    return fig

st.markdown("---")

st.subheader("Performance Over Time")
time_col1,time_col2 = st.columns(2)  
#combined_data['CTR'] = (combined_data['# of Website Clicks'] / combined_data['# of Impressions'])
with time_col1:
    st.markdown("#### Engagement & Conversion Trends")
    # Ensure the metric names 'CTR' and 'CR' exist as columns in combined_data_df
    # The values in your CSV for CTR and CR are fractions (e.g., 0.0848).
    # The labels will indicate they represent percentages.
    engagement_fig = plot_combined_metrics_plotly(
        combined_data,
        metric1_name='CTR', # This column exists in your CSV
        metric2_name='CR',  # This column exists in your CSV
        title='Daily CTR & Conversion Rate Trends',
        metric1_label='CTR', # Or 'CTR (%)' if you multiply by 100
        metric2_label='CR' # Or 'CR (%)'
    )
    st.plotly_chart(engagement_fig,use_container_width=True)

with time_col2: 
    st.markdown("#### Cost Efficiency Trends")
    # Ensure the metric names 'CPC' and 'CPA' exist as columns in combined_data_df
    # and that the 'Date' column is valid.
    if 'CPC' in combined_data.columns and 'CPA' in combined_data.columns and \
       'Date' in combined_data.columns and pd.api.types.is_datetime64_any_dtype(combined_data['Date']) and \
       not combined_data['Date'].isnull().all():

        cost_fig = plot_combined_metrics_plotly(
            combined_data, # Use the same combined DataFrame
            metric1_name='CPC',  # Plot Cost Per Click on the primary y-axis
            metric2_name='CPA',  # Plot Cost Per Acquisition on the secondary y-axis
            title='Daily CPC & CPA Trends',
            metric1_label='Cost Per Click (USD)',
            metric2_label='Cost Per Acquisition (USD)'
        )
        st.plotly_chart(cost_fig, use_container_width=True)
    else:
        st.warning("Could not plot cost efficiency trends due to missing or invalid data (Date, CPC, or CPA). Check console for details from data loading.")

st.markdown("Test campaign appears significantly more effective at generating initial user engagement, consistently achieving a higher Click-Through Rate (CTR) and often a lower Cost Per Click (CPC) than the Control campaign. ")
st.markdown("---")

def plot_daily_funnel_step_ratios(df: pd.DataFrame, ratio_columns: list, title_prefix="Daily"):
    """
    Plots daily funnel step ratios over time for Control and Test campaigns.
    Each ratio gets its own subplot.

    Args:
        df (pd.DataFrame): DataFrame containing 'Date', 'Campaign Name', and ratio columns.
        ratio_columns (list): List of column names for the funnel ratios to plot.
                               e.g., ['Click_to_Search_Ratio', 'Search_to_View_Content_Ratio', ...]
        title_prefix (str): Prefix for the plot titles.

    Returns:
        list: A list of Plotly figure objects, one for each ratio.
    """
    figs = []
    control_df = df[df['Campaign Name'] == 'Control Campaign'].sort_values(by='Date')
    test_df = df[df['Campaign Name'] == 'Test Campaign'].sort_values(by='Date')

    for ratio_col in ratio_columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=control_df['Date'],
            y=control_df[ratio_col],
            mode='lines+markers',
            name=f'Control - {ratio_col.replace("_", " ")}',
            line=dict(color='#4287f5')
        ))
        fig.add_trace(go.Scatter(
            x=test_df['Date'],
            y=test_df[ratio_col],
            mode='lines+markers',
            name=f'Test - {ratio_col.replace("_", " ")}',
            line=dict(color='red', dash='dash')
        ))
        fig.update_layout(
            title_text=f'{title_prefix} {ratio_col.replace("_", " ")} Over Time',
            xaxis_title='Date',
            yaxis_title='Ratio',
            legend_title_text='Campaign',
            title_x=0.5
        )
        figs.append(fig)
    return figs

combined_data['Date'] = pd.to_datetime(combined_data['Date'], dayfirst=True, errors='coerce')


st.header("Funnel Step Performance Over Time")
funnel_ratio_cols = [
    'Click_to_Search_Ratio',
    'Search_to_View_Content_Ratio',
    'View_Content_to_Add_to_Cart_Ratio',
    'Add_to_Cart_to_Purchase_Ratio'
]
funnel_step_figs = plot_daily_funnel_step_ratios(combined_data, funnel_ratio_cols)
for fig_fs in funnel_step_figs:
    st.plotly_chart(fig_fs, use_container_width=True)

st.markdown("---") 

def plot_cumulative_performance(df: pd.DataFrame, metrics_to_plot: list, metric_labels: dict = None):
    """
    Plots cumulative performance metrics over time for Control and Test campaigns.

    Args:
        df (pd.DataFrame): DataFrame with 'Date', 'Campaign Name', and metric columns.
        metrics_to_plot (list): List of column names for metrics to plot cumulatively
                                (e.g., ['# of Purchase', 'Spend [USD]']).
        metric_labels (dict, optional): Dictionary to map metric column names to display labels.

    Returns:
        list: A list of Plotly figure objects, one for each metric.
    """
    figs = []
    df_sorted = df.sort_values(by=['Campaign Name', 'Date'])

    if metric_labels is None:
        metric_labels = {metric: metric.replace("_", " ") for metric in metrics_to_plot}


    for metric in metrics_to_plot:
        fig = go.Figure()
        for campaign_name, campaign_color, line_dash in [('Control Campaign', 'blue', 'solid'), ('Test Campaign', 'red', 'dash')]:
            campaign_df = df_sorted[df_sorted['Campaign Name'] == campaign_name].copy()
            campaign_df[f'Cumulative_{metric}'] = campaign_df[metric].cumsum()
            fig.add_trace(go.Scatter(
                x=campaign_df['Date'],
                y=campaign_df[f'Cumulative_{metric}'],
                mode='lines',
                name=f'{campaign_name} - Cumulative {metric_labels.get(metric, metric)}',
                line=dict(color=campaign_color, dash=line_dash)
            ))
        fig.update_layout(
            title_text=f'Cumulative {metric_labels.get(metric, metric)} Over Time',
            xaxis_title='Date',
            yaxis_title=f'Cumulative {metric_labels.get(metric, metric)}',
            legend_title_text='Campaign',
            title_x=0.5
        )
        figs.append(fig)

    # Special case for Cumulative CPA
    if '# of Purchase' in df.columns and 'Spend [USD]' in df.columns:
        fig_cpa = go.Figure()
        for campaign_name, campaign_color, line_dash in [('Control Campaign', 'blue', 'solid'), ('Test Campaign', 'red', 'dash')]:
            campaign_df = df_sorted[df_sorted['Campaign Name'] == campaign_name].copy()
            campaign_df['Cumulative_# of Purchase'] = campaign_df['# of Purchase'].cumsum()
            campaign_df['Cumulative_Spend [USD]'] = campaign_df['Spend [USD]'].cumsum()
            # Avoid division by zero; start CPA calculation when purchases > 0
            campaign_df['Cumulative_CPA'] = campaign_df.apply(
                lambda row: row['Cumulative_Spend [USD]'] / row['Cumulative_# of Purchase'] if row['Cumulative_# of Purchase'] > 0 else None,
                axis=1
            )
            campaign_df.dropna(subset=['Cumulative_CPA'], inplace=True) # Remove rows where CPA couldn't be calculated yet

            fig_cpa.add_trace(go.Scatter(
                x=campaign_df['Date'],
                y=campaign_df['Cumulative_CPA'],
                mode='lines',
                name=f'{campaign_name} - Cumulative CPA',
                line=dict(color=campaign_color, dash=line_dash)
            ))
        fig_cpa.update_layout(
            title_text='Cumulative Cost Per Acquisition (CPA) Over Time',
            xaxis_title='Date',
            yaxis_title='Cumulative CPA (USD)',
            legend_title_text='Campaign',
            title_x=0.5
        )
        figs.append(fig_cpa)

    return figs 

st.header("Cumulative Performance Trends")
cumulative_metrics = ['# of Purchase', 'Spend [USD]']
# You can add more like '# of Website Clicks' if desired
cumulative_figs = plot_cumulative_performance(combined_data, cumulative_metrics)
for fig_cum in cumulative_figs:
    st.plotly_chart(fig_cum, use_container_width=True)

st.markdown("---") 
def plot_daily_metric_distributions(df: pd.DataFrame, metrics_to_plot: list, metric_labels: dict = None):
    """
    Plots box plots for the distribution of daily key metrics, comparing Control and Test.

    Args:
        df (pd.DataFrame): DataFrame with 'Date', 'Campaign Name', and metric columns.
        metrics_to_plot (list): List of column names for metrics.
                                (e.g., ['CR', 'CPA', '# of Purchase']).
        metric_labels (dict, optional): Dictionary to map metric column names to display labels.

    Returns:
        list: A list of Plotly figure objects, one for each metric.
    """
    figs = []
    if metric_labels is None:
        metric_labels = {metric: metric.replace("_", " ") for metric in metrics_to_plot}

    for metric in metrics_to_plot:
        # Ensure metric column is numeric
        if not pd.api.types.is_numeric_dtype(df[metric]):
            print(f"Warning: Metric '{metric}' is not numeric and will be skipped for distribution plot.")
            continue

        fig = px.box(
            df,
            x='Campaign Name',
            y=metric,
            color='Campaign Name',
            notched=True, # Optional: for a "notched" box plot
            points="all", # Optional: show all data points
            title=f'Distribution of Daily {metric_labels.get(metric, metric)}',
            labels={metric: metric_labels.get(metric, metric), 'Campaign Name': 'Campaign'}
        )
        fig.update_layout(title_x=0.5)
        figs.append(fig)
    return figs

st.header("Distribution of Daily Key Metrics")
distribution_metrics = ['CR', 'CPA', '# of Purchase'] # Make sure these columns exist and are numeric
# You might need to convert CR to percentage if it's a fraction for better y-axis scaling in the plot
# e.g., combined_data_df['CR_Percent'] = combined_data_df['CR'] * 100
# then use 'CR_Percent' in distribution_metrics

# Ensure data types are correct before plotting distributions
for col in distribution_metrics:
    if col in combined_data.columns:
        combined_data[col] = pd.to_numeric(combined_data[col], errors='coerce')
    else:
        st.warning(f"Column {col} not found for distribution plot.")
        distribution_metrics.remove(col) # Remove if not found to avoid errors

combined_data_cleaned = combined_data.dropna(subset=distribution_metrics) # Drop rows where key metrics might be NaN after conversion

if not combined_data_cleaned.empty and distribution_metrics:
    dist_figs = plot_daily_metric_distributions(combined_data_cleaned, distribution_metrics)
    for fig_dist in dist_figs:
        st.plotly_chart(fig_dist, use_container_width=True)
else:
    st.warning("Not enough data to plot daily metric distributions after cleaning.")


st.markdown("---") 

st.subheader("The Core Paradox: High Engagement, Lower Conversion")
st.markdown("The data clearly indicates that while the Test Campaign excels at the top of the funnel (attracting clicks), there's a significant drop-off in performance further down the funnel. This suggests a potential mismatch between what the ad promises (or who it attracts) and what the user experiences post-click.")

st.markdown("---") 

st.subheader("Business Implications")
st.markdown("This performance divergence has several important business implications:")
st.markdown("1.  **Potentially Higher Cost Per Acquisition (CPA) for Test Campaign:** While not explicitly calculated with significance here, a lower CR despite higher spend (due to more clicks) often leads to a less efficient CPA.")
st.markdown("2.  **Risk of Wasted Ad Spend:** The high volume of clicks on the Test Campaign that don't convert represents potentially inefficient ad spend.")
st.markdown("3.  **Opportunity Cost:** We are successfully grabbing attention with the Test Campaign but failing to capitalize on it fully. Improving the conversion rate of this high-CTR campaign could yield substantial gains.")

st.markdown("---") 

st.subheader("Recommendations")

st.markdown("To address these findings and capitalize on the Test Campaign's strengths while mitigating its weaknesses, we recommend the following:")

st.markdown("*   ### Immediate Actions (Next 1-2 Weeks):")
st.markdown("1.  **Optimize Landing Page for Test Campaign Traffic:**")
st.markdown("*   **Action:** Review and revise the landing page content, design, and call-to-action (CTA) specifically for users arriving from the Test Campaign ads.")
st.markdown("*   **Goal:** Ensure a seamless transition from ad to landing page, reinforcing the ad's message and value proposition to improve conversion.")
st.markdown("*   **Consider:** A/B testing variations of the landing page for the Test Campaign traffic.")
st.markdown("2.  **Refine Audience Targeting for Test Campaign:**")
st.markdown("*   **Action:** Analyze the demographic and behavioral data of users who clicked on the Test Campaign ads versus those who converted.")
st.markdown("*   **Goal:** Identify if the Test Campaign is attracting a less qualified audience and adjust targeting parameters to attract users with higher purchase intent.")

st.markdown("*   ### Medium-Term Actions (Next 1-3 Months):")
st.markdown("1.  **Iterate on Test Campaign Creatives:**")
st.markdown("*   **Action:** While the CTR is high, explore subtle modifications to ad copy or visuals that might pre-qualify users better without significantly sacrificing click volume.")
st.markdown("*   **Goal:** Maintain high engagement while attracting a more conversion-ready audience.")
st.markdown("2.  **Analyze the Full Funnel for Drop-off Points:**")
st.markdown("*   **Action:** Beyond the initial click-to-conversion, examine intermediate steps (e.g., add-to-cart, checkout initiation) for both campaigns to pinpoint specific friction points in the Test Campaign's user journey.")
st.markdown("*   **Goal:** Identify and address bottlenecks throughout the conversion funnel.")

st.subheader("Next Steps & Monitoring")

st.markdown("*   **Responsibility:**")
st.markdown("*   Marketing Team: Lead landing page optimization and audience targeting adjustments.")
st.markdown("*   Analytics Team: Support with data analysis for audience segmentation and monitor post-change performance.")
st.markdown("*   **Timeline:** Implement immediate actions within the next two weeks.")
st.markdown("*   **Monitoring:** We will track CR, CTR, CPA, and overall ROAS for both campaigns closely following these changes. A follow-up analysis will be provided [e.g., 4 weeks post-implementation].")

st.markdown("---")
st.subheader("Conclusion")

st.markdown("The Test Campaign demonstrates significant potential in capturing user attention. By strategically addressing the post-click experience and refining audience targeting, we can work towards translating this high engagement into improved conversion rates and, ultimately, greater business impact. We are optimistic that these targeted interventions will enhance the overall effectiveness of our marketing efforts.")
