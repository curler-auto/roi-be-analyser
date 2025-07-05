import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO

st.set_page_config(layout="wide", page_title="ROI & Break-Even Calculator for Asset Finance")

# --- Default Parameters ---
def get_default_params():
    # Store all parameters in a flat dictionary for easier session state management
    params = {
        # Initial Costs
        'initial_dev_team_salaries': 1_500_000,
        'initial_cloud_setup_costs': 50_000,
        'initial_marketing_launch_costs': 200_000,
        'other_initial_one_time_costs': 25_000,

        # Annual Operating Costs
        'annual_core_platform_team_salaries': 1_000_000,
        'annual_baseline_cloud_costs': 75_000,
        'annual_customer_support_team_salaries': 200_000,
        'annual_sales_marketing_budget': 400_000,
        'annual_research_innovation_budget': 150_000,
        'annual_compliance_audit_costs': 50_000,
        'annual_internal_software_tools_licenses': 30_000,
        'other_annual_operating_expenses': 20_000,

        # Customer Growth & Projection
        'annual_new_cust_growth_rate': 0.15, # 15%
        'num_years_projection': 5,
    }

    # Segment-specific data as a DataFrame for easier handling
    customer_segments_df = pd.DataFrame({
        'Segment': ['Small', 'Medium', 'Enterprise'],
        'ARR_per_Customer': [25000, 50000, 100000],
        'CAC_per_Customer': [10000, 25000, 50000],
        'Churn_Rate': [0.15, 0.10, 0.05], # 15%, 10%, 5%
        'Expansion_Rate': [0.05, 0.07, 0.10], # 5%, 7%, 10%
        'Variable_Cost_to_Serve': [1000, 2500, 5000],
        'New_Customers_Year1': [20, 10, 3]
    })
    
    return params, customer_segments_df

# --- Initialize Session State (Crucial for Slider Fix and Data Persistence) ---
if 'initialized' not in st.session_state:
    st.session_state.params, st.session_state.customer_segments_df = get_default_params()
    st.session_state.data_source_option = "Use Default Values" # Default selected option
    st.session_state.uploaded_file_data = False # To track if data was successfully uploaded
    st.session_state.initialized = True

# --- Calculation Logic ---
@st.cache_data(show_spinner=False) # Cache the calculation function for performance
def calculate_financials(params, customer_segments_df):
    years = list(range(0, params['num_years_projection'] + 1))
    
    # Initialize DataFrame
    df = pd.DataFrame(index=years, columns=[
        'Total Customers (EoY)', 'New Customers Acquired (Total)', 'Churned Customers (Total)',
        'Small Customers (EoY)', 'Medium Customers (EoY)', 'Enterprise Customers (EoY)',
        'New Small Customers (Acquired)', 'New Medium Customers (Acquired)', 'New Enterprise Customers (Acquired)',
        'Annual Recurring Revenue (ARR)', 'Total Variable Cost to Serve', 'Net Revenue (ARR - VCS)',
        'Total Customer Acquisition Cost (CAC)', 'Total Annual Operating Costs',
        'Total Annual Costs (Operating + CAC)', 'Annual Net Profit / (Loss)',
        'Cumulative Net Profit / (Loss)', 'Cumulative Total Investment'
    ])

    # Initial Year (Year 0) Costs
    total_initial_investment = params['initial_dev_team_salaries'] + \
                               params['initial_cloud_setup_costs'] + \
                               params['initial_marketing_launch_costs'] + \
                               params['other_initial_one_time_costs']
                               
    df.loc[0, 'Cumulative Total Investment'] = total_initial_investment
    df.loc[0, 'Annual Net Profit / (Loss)'] = -total_initial_investment
    df.loc[0, 'Cumulative Net Profit / (Loss)'] = -total_initial_investment
    
    # Initialize customer counts for Year 0
    df.loc[0, ['Small Customers (EoY)', 'Medium Customers (EoY)', 'Enterprise Customers (EoY)']] = 0
    df.loc[0, 'Total Customers (EoY)'] = 0
    df.loc[0, ['New Small Customers (Acquired)', 'New Medium Customers (Acquired)', 'New Enterprise Customers (Acquired)']] = 0

    # Loop through projection years
    for year in range(1, params['num_years_projection'] + 1):
        prev_year = year - 1

        # Determine new customers for the current year
        if year == 1:
            new_small = customer_segments_df.loc[customer_segments_df['Segment'] == 'Small', 'New_Customers_Year1'].iloc[0]
            new_medium = customer_segments_df.loc[customer_segments_df['Segment'] == 'Medium', 'New_Customers_Year1'].iloc[0]
            new_enterprise = customer_segments_df.loc[customer_segments_df['Segment'] == 'Enterprise', 'New_Customers_Year1'].iloc[0]
        else:
            # Use previously calculated new customers from last year for growth rate application
            new_small = round(df.loc[prev_year, 'New Small Customers (Acquired)'] * (1 + params['annual_new_cust_growth_rate']))
            new_medium = round(df.loc[prev_year, 'New Medium Customers (Acquired)'] * (1 + params['annual_new_cust_growth_rate']))
            new_enterprise = round(df.loc[prev_year, 'New Enterprise Customers (Acquired)'] * (1 + params['annual_new_cust_growth_rate']))

        df.loc[year, 'New Small Customers (Acquired)'] = new_small
        df.loc[year, 'New Medium Customers (Acquired)'] = new_medium
        df.loc[year, 'New Enterprise Customers (Acquired)'] = new_enterprise
        df.loc[year, 'New Customers Acquired (Total)'] = new_small + new_medium + new_enterprise

        # Calculate Churn for current year (based on previous year's EOY customers)
        churn_small = round(df.loc[prev_year, 'Small Customers (EoY)'] * customer_segments_df.loc[customer_segments_df['Segment'] == 'Small', 'Churn_Rate'].iloc[0])
        churn_medium = round(df.loc[prev_year, 'Medium Customers (EoY)'] * customer_segments_df.loc[customer_segments_df['Segment'] == 'Medium', 'Churn_Rate'].iloc[0])
        churn_enterprise = round(df.loc[prev_year, 'Enterprise Customers (EoY)'] * customer_segments_df.loc[customer_segments_df['Segment'] == 'Enterprise', 'Churn_Rate'].iloc[0])
        df.loc[year, 'Churned Customers (Total)'] = churn_small + churn_medium + churn_enterprise

        # Calculate Customers at End of Year (ensure no negative customers)
        df.loc[year, 'Small Customers (EoY)'] = max(0, df.loc[prev_year, 'Small Customers (EoY)'] + new_small - churn_small)
        df.loc[year, 'Medium Customers (EoY)'] = max(0, df.loc[prev_year, 'Medium Customers (EoY)'] + new_medium - churn_medium)
        df.loc[year, 'Enterprise Customers (EoY)'] = max(0, df.loc[prev_year, 'Enterprise Customers (EoY)'] + new_enterprise - churn_enterprise)
        df.loc[year, 'Total Customers (EoY)'] = df.loc[year, 'Small Customers (EoY)'] + df.loc[year, 'Medium Customers (EoY)'] + df.loc[year, 'Enterprise Customers (EoY)']

        # Calculate ARR (including expansion from retained customers)
        # Revenue from retained customers (prev year's revenue * (1-churn) * (1+expansion))
        # This is a bit more nuanced. Simplified here to apply expansion to all active customers for simplicity in this general framework.
        # For full accuracy, you'd track cohort revenue.
        arr_small_seg = df.loc[year, 'Small Customers (EoY)'] * customer_segments_df.loc[customer_segments_df['Segment'] == 'Small', 'ARR_per_Customer'].iloc[0] * (1 + customer_segments_df.loc[customer_segments_df['Segment'] == 'Small', 'Expansion_Rate'].iloc[0])
        arr_medium_seg = df.loc[year, 'Medium Customers (EoY)'] * customer_segments_df.loc[customer_segments_df['Segment'] == 'Medium', 'ARR_per_Customer'].iloc[0] * (1 + customer_segments_df.loc[customer_segments_df['Segment'] == 'Medium', 'Expansion_Rate'].iloc[0])
        arr_enterprise_seg = df.loc[year, 'Enterprise Customers (EoY)'] * customer_segments_df.loc[customer_segments_df['Segment'] == 'Enterprise', 'ARR_per_Customer'].iloc[0] * (1 + customer_segments_df.loc[customer_segments_df['Segment'] == 'Enterprise', 'Expansion_Rate'].iloc[0])
        df.loc[year, 'Annual Recurring Revenue (ARR)'] = arr_small_seg + arr_medium_seg + arr_enterprise_seg

        # Calculate Variable Cost to Serve
        vc_serve_small = df.loc[year, 'Small Customers (EoY)'] * customer_segments_df.loc[customer_segments_df['Segment'] == 'Small', 'Variable_Cost_to_Serve'].iloc[0]
        vc_serve_medium = df.loc[year, 'Medium Customers (EoY)'] * customer_segments_df.loc[customer_segments_df['Segment'] == 'Medium', 'Variable_Cost_to_Serve'].iloc[0]
        vc_serve_enterprise = df.loc[year, 'Enterprise Customers (EoY)'] * customer_segments_df.loc[customer_segments_df['Segment'] == 'Enterprise', 'Variable_Cost_to_Serve'].iloc[0]
        df.loc[year, 'Total Variable Cost to Serve'] = vc_serve_small + vc_serve_medium + vc_serve_enterprise

        # Calculate Net Revenue
        df.loc[year, 'Net Revenue (ARR - VCS)'] = df.loc[year, 'Annual Recurring Revenue (ARR)'] - df.loc[year, 'Total Variable Cost to Serve']

        # Calculate CAC
        cac_total = (new_small * customer_segments_df.loc[customer_segments_df['Segment'] == 'Small', 'CAC_per_Customer'].iloc[0]) + \
                    (new_medium * customer_segments_df.loc[customer_segments_df['Segment'] == 'Medium', 'CAC_per_Customer'].iloc[0]) + \
                    (new_enterprise * customer_segments_df.loc[customer_segments_df['Segment'] == 'Enterprise', 'CAC_per_Customer'].iloc[0])
        df.loc[year, 'Total Customer Acquisition Cost (CAC)'] = cac_total

        # Total Annual Operating Costs (fixed costs for the year)
        df.loc[year, 'Total Annual Operating Costs'] = params['annual_core_platform_team_salaries'] + \
                                                      params['annual_baseline_cloud_costs'] + \
                                                      params['annual_customer_support_team_salaries'] + \
                                                      params['annual_sales_marketing_budget'] + \
                                                      params['annual_research_innovation_budget'] + \
                                                      params['annual_compliance_audit_costs'] + \
                                                      params['annual_internal_software_tools_licenses'] + \
                                                      params['other_annual_operating_expenses']

        # Total Annual Costs (Operating Costs + CAC) for the current year
        df.loc[year, 'Total Annual Costs (Operating + CAC)'] = df.loc[year, 'Total Annual Operating Costs'] + df.loc[year, 'Total Customer Acquisition Cost (CAC)']

        # Annual Net Profit / (Loss)
        df.loc[year, 'Annual Net Profit / (Loss)'] = df.loc[year, 'Net Revenue (ARR - VCS)'] - df.loc[year, 'Total Annual Operating Costs'] - df.loc[year, 'Total Customer Acquisition Cost (CAC)']

        # Cumulative Net Profit / (Loss)
        df.loc[year, 'Cumulative Net Profit / (Loss)'] = df.loc[prev_year, 'Cumulative Net Profit / (Loss)'] + df.loc[year, 'Annual Net Profit / (Loss)']

        # Cumulative Total Investment (Initial investment + all annual costs)
        df.loc[year, 'Cumulative Total Investment'] = df.loc[prev_year, 'Cumulative Total Investment'] + df.loc[year, 'Total Annual Costs (Operating + CAC)']

    return df.fillna(0) # Fill any potential NaNs from non-calculated cells with 0

# --- App Title and Description ---
st.title("💰 Asset Finance Platform: ROI & Break-Even Calculator")
st.markdown("""
This application helps the Program Director for the Platform Team (building an end-to-end asset finance solution)
to determine the Break-Even Point and financial ROI.
""")

# --- Left Pane: Control Measures (Sidebar) ---
st.sidebar.header("⚙️ Control Measures")

# Data Source Selection
st.sidebar.subheader("1. Data Source Selection")
st.session_state.data_source_option = st.sidebar.radio(
    "Choose Input Data:",
    ("Use Default Values", "Upload My Own Data")
)

# User Data Upload Interface
if st.session_state.data_source_option == "Upload My Own Data":
    st.sidebar.markdown("Upload two Excel sheets/CSV files:")
    st.sidebar.markdown("- `Input_Parameters` (for general costs/growth)")
    st.sidebar.markdown("- `Customer_Segments_Data` (for per-segment data)")
    
    uploaded_files = st.sidebar.file_uploader(
        "Select your Excel/CSV files (.xlsx, .csv)",
        type=["xlsx", "csv"],
        accept_multiple_files=True,
        help="Ensure correct file names and column headers as per documentation."
    )

    if uploaded_files:
        params_file = None
        customer_file = None

        for file in uploaded_files:
            if "Input_Parameters" in file.name:
                params_file = file
            elif "Customer_Segments_Data" in file.name:
                customer_file = file

        if params_file and customer_file:
            try:
                # Read Input_Parameters
                if params_file.name.endswith('.xlsx'):
                    df_params = pd.read_excel(params_file, sheet_name='Input_Parameters')
                else:
                    df_params = pd.read_csv(params_file)
                
                uploaded_params = {row['Parameter_Name']: row['Value'] for _, row in df_params.iterrows()}

                # Read Customer_Segments_Data
                if customer_file.name.endswith('.xlsx'):
                    uploaded_customer_df = pd.read_excel(customer_file, sheet_name='Customer_Segments_Data')
                else:
                    uploaded_customer_df = pd.read_csv(customer_file)

                # Update session state with uploaded data
                st.session_state.params.update(uploaded_params) # Update existing dict
                st.session_state.customer_segments_df = uploaded_customer_df
                st.session_state.uploaded_file_data = True
                st.sidebar.success("Custom data loaded successfully! Sliders are now disabled.")
                st.experimental_rerun() # Rerun to apply changes and disable sliders
            except Exception as e:
                st.sidebar.error(f"Error reading files. Please check format. Error: {e}")
                st.session_state.uploaded_file_data = False
        else:
            st.sidebar.warning("Please upload BOTH 'Input_Parameters' and 'Customer_Segments_Data' files.")
            st.session_state.uploaded_file_data = False
    else: # If no file is currently in the uploader, ensure flag is reset if "upload" was chosen
        if st.session_state.data_source_option == "Upload My Own Data" and st.session_state.uploaded_file_data:
            st.session_state.uploaded_file_data = False
            # This rerun ensures sliders re-enable if user clears uploaded files
            st.experimental_rerun() 
else: # If using default values, ensure uploaded_file_data is False
    if st.session_state.uploaded_file_data:
        st.session_state.uploaded_file_data = False
        st.experimental_rerun() # Rerun to re-enable sliders


# Determine if sliders should be disabled
sliders_disabled = (st.session_state.data_source_option == "Upload My Own Data" and st.session_state.uploaded_file_data)

# --- Render Sliders (Control Measures) ---
st.sidebar.subheader("2. Adjust Parameters")
st.sidebar.markdown("*(Disabled if 'Upload My Own Data' is selected)*")

with st.sidebar.expander("Initial (Year 0) Costs"):
    st.session_state.params['initial_dev_team_salaries'] = st.slider("Initial Dev Team Salaries ($)", 
                                                                 min_value=500_000, max_value=5_000_000, step=100_000, 
                                                                 value=int(st.session_state.params['initial_dev_team_salaries']),
                                                                 disabled=sliders_disabled, key="initial_dev_team_salaries_slider")
    st.session_state.params['initial_cloud_setup_costs'] = st.slider("Initial Cloud Setup Costs ($)", 
                                                                 min_value=10_000, max_value=200_000, step=10_000, 
                                                                 value=int(st.session_state.params['initial_cloud_setup_costs']),
                                                                 disabled=sliders_disabled, key="initial_cloud_setup_costs_slider")
    st.session_state.params['initial_marketing_launch_costs'] = st.slider("Initial Marketing & Launch ($)", 
                                                                      min_value=50_000, max_value=1_000_000, step=50_000, 
                                                                      value=int(st.session_state.params['initial_marketing_launch_costs']),
                                                                      disabled=sliders_disabled, key="initial_marketing_launch_costs_slider")
    st.session_state.params['other_initial_one_time_costs'] = st.slider("Other Initial One-Time Costs ($)", 
                                                                    min_value=0, max_value=200_000, step=10_000, 
                                                                    value=int(st.session_state.params['other_initial_one_time_costs']),
                                                                    disabled=sliders_disabled, key="other_initial_one_time_costs_slider")

with st.sidebar.expander("Annual Operating Costs"):
    st.session_state.params['annual_core_platform_team_salaries'] = st.slider("Annual Core Platform Team Salaries ($)", 
                                                                           min_value=500_000, max_value=3_000_000, step=100_000, 
                                                                           value=int(st.session_state.params['annual_core_platform_team_salaries']),
                                                                           disabled=sliders_disabled, key="annual_core_platform_team_salaries_slider")
    st.session_state.params['annual_baseline_cloud_costs'] = st.slider("Annual Baseline Cloud Costs ($)", 
                                                                    min_value=20_000, max_value=200_000, step=10_000, 
                                                                    value=int(st.session_state.params['annual_baseline_cloud_costs']),
                                                                    disabled=sliders_disabled, key="annual_baseline_cloud_costs_slider")
    st.session_state.params['annual_customer_support_team_salaries'] = st.slider("Annual Customer Support Team Salaries ($)", 
                                                                              min_value=100_000, max_value=1_000_000, step=50_000, 
                                                                              value=int(st.session_state.params['annual_customer_support_team_salaries']),
                                                                              disabled=sliders_disabled, key="annual_customer_support_team_salaries_slider")
    st.session_state.params['annual_sales_marketing_budget'] = st.slider("Annual Sales & Marketing Budget ($)", 
                                                                     min_value=200_000, max_value=2_000_000, step=100_000, 
                                                                     value=int(st.session_state.params['annual_sales_marketing_budget']),
                                                                     disabled=sliders_disabled, key="annual_sales_marketing_budget_slider")
    st.session_state.params['annual_research_innovation_budget'] = st.slider("Annual Research & Innovation Budget ($)", 
                                                                         min_value=0, max_value=500_000, step=25_000, 
                                                                         value=int(st.session_state.params['annual_research_innovation_budget']),
                                                                         disabled=sliders_disabled, key="annual_research_innovation_budget_slider")
    st.session_state.params['annual_compliance_audit_costs'] = st.slider("Annual Compliance & Audit Costs ($)", 
                                                                     min_value=0, max_value=200_000, step=10_000, 
                                                                     value=int(st.session_state.params['annual_compliance_audit_costs']),
                                                                     disabled=sliders_disabled, key="annual_compliance_audit_costs_slider")
    st.session_state.params['annual_internal_software_tools_licenses'] = st.slider("Annual Internal Software/Tools Licenses ($)", 
                                                                               min_value=0, max_value=100_000, step=5_000, 
                                                                               value=int(st.session_state.params['annual_internal_software_tools_licenses']),
                                                                               disabled=sliders_disabled, key="annual_internal_software_tools_licenses_slider")
    st.session_state.params['other_annual_operating_expenses'] = st.slider("Other Annual Operating Expenses ($)", 
                                                                       min_value=0, max_value=100_000, step=5_000, 
                                                                       value=int(st.session_state.params['other_annual_operating_expenses']),
                                                                       disabled=sliders_disabled, key="other_annual_operating_expenses_slider")

with st.sidebar.expander("Customer & Revenue Dynamics"):
    
    # Use helper function to get/set values from DataFrame
    def get_df_value(segment, column):
        return st.session_state.customer_segments_df.loc[st.session_state.customer_segments_df['Segment'] == segment, column].iloc[0]

    def update_df_value(segment, column, value):
        st.session_state.customer_segments_df.loc[st.session_state.customer_segments_df['Segment'] == segment, column] = value

    st.subheader("Small Business Segment")
    update_df_value('Small', 'ARR_per_Customer', st.number_input("ARR per Customer (Small) ($)", value=int(get_df_value('Small', 'ARR_per_Customer')), min_value=1000, disabled=sliders_disabled, key="arr_small_input"))
    update_df_value('Small', 'CAC_per_Customer', st.number_input("CAC per Customer (Small) ($)", value=int(get_df_value('Small', 'CAC_per_Customer')), min_value=1000, disabled=sliders_disabled, key="cac_small_input"))
    update_df_value('Small', 'Churn_Rate', st.slider("Annual Churn Rate (Small) (%)", min_value=0, max_value=30, value=int(get_df_value('Small', 'Churn_Rate')*100), disabled=sliders_disabled, key="churn_small_slider") / 100)
    update_df_value('Small', 'Expansion_Rate', st.slider("Expansion Rate (Small) (%)", min_value=0, max_value=20, value=int(get_df_value('Small', 'Expansion_Rate')*100), disabled=sliders_disabled, key="expansion_small_slider") / 100)
    update_df_value('Small', 'Variable_Cost_to_Serve', st.number_input("Variable Cost to Serve (Small) ($)", value=int(get_df_value('Small', 'Variable_Cost_to_Serve')), min_value=0, disabled=sliders_disabled, key="vc_serve_small_input"))
    update_df_value('Small', 'New_Customers_Year1', st.number_input("New Small Customers (Year 1)", value=int(get_df_value('Small', 'New_Customers_Year1')), min_value=0, disabled=sliders_disabled, key="new_cust_small_y1_input"))

    st.subheader("Medium Business Segment")
    update_df_value('Medium', 'ARR_per_Customer', st.number_input("ARR per Customer (Medium) ($)", value=int(get_df_value('Medium', 'ARR_per_Customer')), min_value=5000, disabled=sliders_disabled, key="arr_medium_input"))
    update_df_value('Medium', 'CAC_per_Customer', st.number_input("CAC per Customer (Medium) ($)", value=int(get_df_value('Medium', 'CAC_per_Customer')), min_value=5000, disabled=sliders_disabled, key="cac_medium_input"))
    update_df_value('Medium', 'Churn_Rate', st.slider("Annual Churn Rate (Medium) (%)", min_value=0, max_value=30, value=int(get_df_value('Medium', 'Churn_Rate')*100), disabled=sliders_disabled, key="churn_medium_slider") / 100)
    update_df_value('Medium', 'Expansion_Rate', st.slider("Expansion Rate (Medium) (%)", min_value=0, max_value=20, value=int(get_df_value('Medium', 'Expansion_Rate')*100), disabled=sliders_disabled, key="expansion_medium_slider") / 100)
    update_df_value('Medium', 'Variable_Cost_to_Serve', st.number_input("Variable Cost to Serve (Medium) ($)", value=int(get_df_value('Medium', 'Variable_Cost_to_Serve')), min_value=0, disabled=sliders_disabled, key="vc_serve_medium_input"))
    update_df_value('Medium', 'New_Customers_Year1', st.number_input("New Medium Customers (Year 1)", value=int(get_df_value('Medium', 'New_Customers_Year1')), min_value=0, disabled=sliders_disabled, key="new_cust_medium_y1_input"))

    st.subheader("Enterprise Business Segment")
    update_df_value('Enterprise', 'ARR_per_Customer', st.number_input("ARR per Customer (Enterprise) ($)", value=int(get_df_value('Enterprise', 'ARR_per_Customer')), min_value=10000, disabled=sliders_disabled, key="arr_enterprise_input"))
    update_df_value('Enterprise', 'CAC_per_Customer', st.number_input("CAC per Customer (Enterprise) ($)", value=int(get_df_value('Enterprise', 'CAC_per_Customer')), min_value=10000, disabled=sliders_disabled, key="cac_enterprise_input"))
    update_df_value('Enterprise', 'Churn_Rate', st.slider("Annual Churn Rate (Enterprise) (%)", min_value=0, max_value=30, value=int(get_df_value('Enterprise', 'Churn_Rate')*100), disabled=sliders_disabled, key="churn_enterprise_slider") / 100)
    update_df_value('Enterprise', 'Expansion_Rate', st.slider("Expansion Rate (Enterprise) (%)", min_value=0, max_value=20, value=int(get_df_value('Enterprise', 'Expansion_Rate')*100), disabled=sliders_disabled, key="expansion_enterprise_slider") / 100)
    update_df_value('Enterprise', 'Variable_Cost_to_Serve', st.number_input("Variable Cost to Serve (Enterprise) ($)", value=int(get_df_value('Enterprise', 'Variable_Cost_to_Serve')), min_value=0, disabled=sliders_disabled, key="vc_serve_enterprise_input"))
    update_df_value('Enterprise', 'New_Customers_Year1', st.number_input("New Enterprise Customers (Year 1)", value=int(get_df_value('Enterprise', 'New_Customers_Year1')), min_value=0, disabled=sliders_disabled, key="new_cust_enterprise_y1_input"))

    st.session_state.params['annual_new_cust_growth_rate'] = st.slider("Annual New Customer Growth Rate (from Year 2) (%)", 
                                                               min_value=0, max_value=50, value=int(st.session_state.params['annual_new_cust_growth_rate']*100), disabled=sliders_disabled, key="annual_new_cust_growth_rate_slider") / 100
    st.session_state.params['num_years_projection'] = st.slider("Number of Years for Projection", min_value=1, max_value=10, 
                                                       value=int(st.session_state.params['num_years_projection']), disabled=sliders_disabled, key="num_years_projection_slider")

# --- Perform Calculations with current session state values ---
# This will rerun automatically when sliders are moved or input is changed
financial_df = calculate_financials(st.session_state.params, st.session_state.customer_segments_df)

# --- Right Pane: Performance Measures & What-If Analysis Results ---
st.header("✨ Performance Measures & What-If Analysis Results")

# --- Key Metrics ---
col_be, col_roi = st.columns(2)

# Calculate Break-Even Point
be_year_idx_mask = (financial_df['Cumulative Net Profit / (Loss)'] >= 0)
if be_year_idx_mask.any() and financial_df.loc[financial_df.index[-1], 'Cumulative Net Profit / (Loss)'] >= 0:
    be_year_idx = financial_df.index[be_year_idx_mask].min() # Get the first year where it's non-negative
    if be_year_idx == 0:
        break_even_point_years = 0.00
    else:
        prev_cum_profit = financial_df.loc[be_year_idx - 1, 'Cumulative Net Profit / (Loss)']
        current_annual_profit = financial_df.loc[be_year_idx, 'Annual Net Profit / (Loss)'] # Use annual profit for interpolation

        if current_annual_profit > 0:
            fractional_year = abs(prev_cum_profit) / current_annual_profit
            break_even_point_years = be_year_idx - 1 + fractional_year
        else:
             # If annual profit is 0 or negative, but cumulative became positive
             # due to previous years, then break-even is at the start of this year.
            break_even_point_years = float(be_year_idx)
else:
    break_even_point_years = "Not Reached"

# Calculate ROI
total_cumulative_profit = financial_df.loc[st.session_state.params['num_years_projection'], 'Cumulative Net Profit / (Loss)']
total_cumulative_investment = financial_df.loc[st.session_state.params['num_years_projection'], 'Cumulative Total Investment']

if total_cumulative_investment != 0:
    roi_percent = (total_cumulative_profit / total_cumulative_investment) * 100
else:
    roi_percent = 0.00

with col_be:
    if isinstance(break_even_point_years, (int, float)):
        st.metric(label="📊 Break-Even Point", value=f"{break_even_point_years:.2f} Years")
    else:
        st.metric(label="📊 Break-Even Point", value=break_even_point_years)
with col_roi:
    st.metric(label=f"📈 Total ROI (at Year {st.session_state.params['num_years_projection']})", value=f"{roi_percent:.2f}%")

# --- Graphs ---
st.subheader("Visualizing Performance")

# Graph 1: Cumulative Net Profit / (Loss)
fig_cashflow = go.Figure()
fig_cashflow.add_trace(go.Scatter(x=financial_df.index, y=financial_df['Cumulative Net Profit / (Loss)'], 
                                  mode='lines+markers', name='Cumulative Net Profit / (Loss)',
                                  line=dict(color='#1f77b4', width=3), marker=dict(size=8)))
fig_cashflow.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-Even Line", annotation_position="top left")
fig_cashflow.update_layout(
    xaxis_title="Year",
    yaxis_title="Cumulative Net Profit / (Loss) ($)",
    hovermode="x unified",
    title_text="Cumulative Financial Performance Over Time",
    height=400
)
st.plotly_chart(fig_cashflow, use_container_width=True)

# Graph 2: Annual Revenue vs. Total Costs
fig_revenue_cost = go.Figure()
fig_revenue_cost.add_trace(go.Bar(x=financial_df.index, y=financial_df['Annual Recurring Revenue (ARR)'], name='Annual Revenue', marker_color='#2ca02c'))
fig_revenue_cost.add_trace(go.Bar(x=financial_df.index, y=financial_df['Total Annual Costs (Operating + CAC)'], name='Total Annual Costs', marker_color='#d62728'))
fig_revenue_cost.update_layout(
    barmode='group',
    xaxis_title="Year",
    yaxis_title="Amount ($)",
    hovermode="x unified",
    title_text="Annual Revenue vs. Total Costs",
    height=400
)
st.plotly_chart(fig_revenue_cost, use_container_width=True)

# Graph 3: Customers by Segment Growth
fig_customers = go.Figure()
fig_customers.add_trace(go.Bar(x=financial_df.index, y=financial_df['Small Customers (EoY)'], name='Small Customers'))
fig_customers.add_trace(go.Bar(x=financial_df.index, y=financial_df['Medium Customers (EoY)'], name='Medium Customers'))
fig_customers.add_trace(go.Bar(x=financial_df.index, y=financial_df['Enterprise Customers (EoY)'], name='Enterprise Customers'))
fig_customers.update_layout(
    barmode='stack',
    xaxis_title="Year",
    yaxis_title="Number of Customers (EoY)",
    hovermode="x unified",
    title_text="Customer Base Growth by Segment",
    height=400
)
st.plotly_chart(fig_customers, use_container_width=True)


# --- Detailed Data Table ---
st.subheader("Detailed Annual Financials (Scroll to view)")
st.dataframe(financial_df.round(0).astype(int), use_container_width=True, height=350)

st.markdown("---")
st.info("💡 **What-If Analysis Tip:** Change values in the left sidebar to instantly see the impact on Break-Even, ROI, and the graphs on the right.")

st.markdown("---")
st.markdown("### ℹ️ Data Upload Format Guide:")
st.markdown("""
To upload your own data, ensure you have two separate Excel sheets (or CSV files) with the following exact names and column headers:

**1. `Input_Parameters`**
   - **Columns:** `Parameter_Category`, `Parameter_Name`, `Value`, `Unit`
   - **Example Rows:**
     `Initial Costs, initial_dev_team_salaries, 1500000, USD`
     `Annual Operating Costs, annual_core_platform_team_salaries, 1000000, USD`
     `Growth Projections, num_years_projection, 5, Years`
     *(All parameters from the "Costs & Overheads" and "Customer Growth & Projection" sections in the sidebar's default values should be included here.)*

**2. `Customer_Segments_Data`**
   - **Columns:** `Segment`, `ARR_per_Customer`, `CAC_per_Customer`, `Churn_Rate`, `Expansion_Rate`, `Variable_Cost_to_Serve`, `New_Customers_Year1`
   - **Example Rows:**
     `Small, 25000, 10000, 0.15, 0.05, 1000, 20`
     `Medium, 50000, 25000, 0.10, 0.07, 2500, 10`
     `Enterprise, 100000, 50000, 0.05, 0.10, 5000, 3`

**Important:** For rates (Churn_Rate, Expansion_Rate, annual_new_cust_growth_rate), input them as decimals (e.g., 0.15 for 15%).
""")