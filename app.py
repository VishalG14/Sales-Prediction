import streamlit as st
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression

# Initialize session state for storing product and sales data
if 'products' not in st.session_state:
    st.session_state['products'] = []

if 'sales' not in st.session_state:
    st.session_state['sales'] = []

def prepare_data(data):
    data['Sale Date'] = pd.to_datetime(data['Sale Date'])
    data['DayOfYear'] = data['Sale Date'].dt.dayofyear
    data['Year'] = data['Sale Date'].dt.year
    return data

def train_model(data):
    # Prepare data
    data = prepare_data(data)
    X = data[['DayOfYear', 'Year']]
    y = data['Earnings']
    
    # Check for empty DataFrame
    if X.empty or y.empty:
        return None

    # Train model using all data if not enough samples for a split
    if len(data) < 5:
        model = LinearRegression()
        model.fit(X, y)
    else:
        # Train-test split
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        except ValueError as e:
            st.error(f"ValueError during train_test_split: {e}")
            return None

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

    return model

def predict_earnings_simple(today_earnings, days_ahead):
    daily_earnings_rate = today_earnings
    total_predicted_earnings = daily_earnings_rate * days_ahead
    return total_predicted_earnings

def calculate_financials(data, sales_data):
    today = datetime.today().strftime("%Y-%m-%d")
    today_sales = pd.DataFrame(sales_data)
    today_sales['Date'] = pd.to_datetime(today_sales['Date'])
    today_data = today_sales[today_sales['Date'] == today]
    today_data['Profit'] = today_data['Quantity Sold'] * (today_data['Selling Price'] - data.set_index('Name').loc[today_data['Product Name']]['Cost Price'].values)

    total_profit = today_data[today_data['Profit'] > 0]['Profit'].sum()
    total_loss = today_data[today_data['Profit'] < 0]['Profit'].sum()
    total_earnings = total_profit + total_loss

    product_earnings = today_data.groupby('Product Name')['Profit'].sum().reset_index()

    return total_profit, total_loss, total_earnings, product_earnings

# Display image and title side by side
col1, col2 = st.columns([1, 3])
with col1:
    st.image("sales.jpg", use_column_width=True)
with col2:
    st.title("Product Sales Analysis")

# Add Product Data Button
add_product_button = st.button('Add Product Data')

if add_product_button:
    st.session_state['add_product'] = True

# Product addition form
if 'add_product' in st.session_state and st.session_state['add_product']:
    st.header('Add Product Details')
    
    product_id = st.number_input('Product ID', min_value=1, step=1)
    product_name = st.text_input('Product Name')
    product_description = st.text_area('Product Description')
    quantity_type = st.selectbox('Quantity Type', ['Unit'])
    sku = st.text_input('SKU')
    quantity = st.number_input('Quantity', min_value=0, step=1)
    product_cost = st.number_input('Product Cost (in rupees)', min_value=0, step=1)
    sell_price = st.number_input('Sell Price (in rupees)', min_value=0, step=1)
    selected_date = st.date_input('Select Date', datetime.today())
    
    save_details_button = st.button('Save Details')
    
    if save_details_button:
        # Save the product details to session state
        new_product = {
            'ID': product_id,
            'Name': product_name,
            'Description': product_description,
            'Quantity Type': quantity_type,
            'SKU': sku,
            'Quantity': quantity,
            'Cost Price': product_cost,
            'Selling Price': sell_price,
            'Date': selected_date.strftime("%Y-%m-%d")
        }
        st.session_state['products'].append(new_product)
        st.session_state['add_product'] = False
        st.success('Product details saved successfully!')

# Add Sales Data Button
add_sales_button = st.button('Add Sales Data')

if add_sales_button:
    st.session_state['add_sales'] = True

# Sales addition form
if 'add_sales' in st.session_state and st.session_state['add_sales']:
    st.header('Add Sales Details')

    product_names = [product['Name'] for product in st.session_state['products']]
    product_name = st.selectbox('Product Name', product_names)
    
    selected_product = next((product for product in st.session_state['products'] if product['Name'] == product_name), None)
    if selected_product:
        max_quantity = selected_product['Quantity']
        quantity_sold = st.number_input('Quantity Sold', min_value=0, max_value=max_quantity, step=1)
        
        if quantity_sold > max_quantity:
            st.warning(f"You can't enter a quantity higher than the actual quantity ({max_quantity}).")
        
        save_sales_button = st.button('Save Sales')
        
        if save_sales_button:
            new_sale = {
                'Product Name': product_name,
                'Quantity Sold': quantity_sold,
                'Selling Price': selected_product['Selling Price'],
                'Date': datetime.today().strftime("%Y-%m-%d")
            }
            st.session_state['sales'].append(new_sale)
            st.session_state['add_sales'] = False
            st.success('Sales details saved successfully!')

# Display products and generate report button
if st.session_state['products']:
    st.header('Products Added')
    for i, product in enumerate(st.session_state['products']):
        with st.expander(f"Product {i + 1}: {product['Name']}"):
            st.markdown(f"""
            <div style="background-color: #f9f9f9; padding: 10px; border-radius: 5px; color: black;">
                <strong>ID:</strong> {product['ID']}<br>
                <strong>Name:</strong> {product['Name']}<br>
                <strong>Description:</strong> {product['Description']}<br>
                <strong>Quantity Type:</strong> {product['Quantity Type']}<br>
                <strong>SKU:</strong> {product['SKU']}<br>
                <strong>Quantity:</strong> {product['Quantity']}<br>
                <strong>Cost Price:</strong> ₹{product['Cost Price']}<br>
                <strong>Selling Price:</strong> ₹{product['Selling Price']}<br>
                <strong>Date:</strong> {product['Date']}
            </div>
            """, unsafe_allow_html=True)
            
            # Show sales related to this product
            product_sales = [sale for sale in st.session_state['sales'] if sale['Product Name'] == product['Name']]
            if product_sales:
                st.markdown("<strong>Sales:</strong>", unsafe_allow_html=True)
                for sale in product_sales:
                    st.markdown(f"""
                    <div style="background-color: #e0f7fa; padding: 10px; border-radius: 5px; margin-top: 10px; color: black;">
                        <strong>Quantity Sold:</strong> {sale['Quantity Sold']}<br>
                        <strong>Selling Price:</strong> ₹{sale['Selling Price']}<br>
                        <strong>Date:</strong> {sale['Date']}
                    </div>
                    """, unsafe_allow_html=True)
        
    generate_report_button = st.button('Generate Report')
    
    if generate_report_button:
        # Combine products and sales data
        df_products = pd.DataFrame(st.session_state['products'])
        df_sales = pd.DataFrame(st.session_state['sales'])
        
        # Calculate combined earnings
        df_sales['Earnings'] = df_sales['Quantity Sold'] * df_sales['Selling Price']
        
        # Combine products and sales data for model training
        df_combined = df_sales[['Date', 'Earnings']].rename(columns={'Date': 'Sale Date'})
        df_combined['Sale Date'] = pd.to_datetime(df_combined['Sale Date'])
        df_combined = df_combined.resample('D', on='Sale Date').sum().reset_index()
        
        # Train the model and make predictions
        model = train_model(df_combined)
        
        # Today's financials
        total_profit, total_loss, total_earnings, product_earnings = calculate_financials(df_products, st.session_state['sales'])
        
        # Predictions using today's earnings
        if model:
            earnings_month = predict_earnings_simple(total_earnings, 30)
            earnings_year = predict_earnings_simple(total_earnings, 365)
            
            # Display the results in a styled format
            st.markdown("""
            <style>
            .report-section {
                background-color: #162447;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                transition: all 0.3s ease;
                color: white;
                margin-bottom: 20px;
                text-align: center;
            }
            .report-section:hover {
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
            }
            .report-section h3 {
                color: #ffab40;
                font-size: 24px;
            }
            .report-section p {
                color: white;
                font-size: 18px;
            }
            .table-section {
                background-color: #1b1b2f;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                transition: all 0.3s ease;
                color: white;
                margin-bottom: 20px;
            }
            .table-section:hover {
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
            }
            .table-section table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }
            .table-section th, .table-section td {
                padding: 15px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            .table-section th {
                background-color: #162447;
                color: #ffab40;
            }
            .table-section td {
                background-color: #1b1b2f;
                color: white;
            }
            .table-section tr:nth-child(even) {
                background-color: #1b1b2f;
            }
            .table-section tr:hover {
                background-color: #162447;
            }
            .section-title {
                font-size: 26px;
                color: #ffab40;
                margin-bottom: 20px;
                text-align: center;
            }
            </style>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="report-section">
                <h3>Sales Prediction</h3>
                <p><strong>Sales after a month:</strong> ₹{earnings_month:.2f}</p>
                <p><strong>Sales after a year:</strong> ₹{earnings_year:.2f}</p>
            </div>
            <div class="report-section">
                <h3>Financials</h3>
                <p><strong>Today's Total Profit:</strong> ₹{total_profit:.2f}</p>
                <p><strong>Today's Total Loss:</strong> ₹{total_loss:.2f}</p>
                <p><strong>Today's Total Earnings:</strong> ₹{total_earnings:.2f}</p>
            </div>
            <div class="table-section">
                <div class="section-title">Top Rated Products & Customer Satisfaction (Top 5 Products)</div>
                <table>
                    <thead>
                        <tr>
                            <th>Product Name</th>
                            <th>Profit (₹)</th>
                        </tr>
                    </thead>
                    <tbody>
            """, unsafe_allow_html=True)

            for index, row in product_earnings.iterrows():
                st.markdown(f"""
                <tr>
                    <td>{row['Product Name']}</td>
                    <td>₹{row['Profit']:.2f}</td>
                </tr>
                """, unsafe_allow_html=True)

            st.markdown("""
                    </tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)