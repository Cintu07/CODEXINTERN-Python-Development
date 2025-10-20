import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Sample dataset creation (replace with your CSV file)
# To use your own CSV: df = pd.read_csv('your_file.csv')
data = {
    'Product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones', 
                'Webcam', 'Speaker', 'Tablet', 'Phone', 'Charger'],
    'Sales': [45000, 1200, 2500, 15000, 3500, 4500, 2800, 28000, 35000, 800],
    'Quantity': [50, 150, 100, 40, 80, 60, 70, 45, 55, 200],
    'Rating': [4.5, 4.2, 4.0, 4.7, 4.3, 3.9, 4.1, 4.6, 4.8, 3.8],
    'Category': ['Electronics', 'Accessories', 'Accessories', 'Electronics', 
                 'Accessories', 'Accessories', 'Electronics', 'Electronics', 
                 'Electronics', 'Accessories']
}

df = pd.DataFrame(data)

# Save sample data to CSV (optional)
df.to_csv('sample_data.csv', index=False)

print("=" * 60)
print("BASIC DATA ANALYSIS")
print("=" * 60)

# Display first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Basic statistics
print("\n\nDataset Information:")
print(df.info())

print("\n\nStatistical Summary:")
print(df.describe())

# Calculate averages
print("\n\nAVERAGE CALCULATIONS:")
print(f"Average Sales: ${df['Sales'].mean():,.2f}")
print(f"Average Quantity: {df['Quantity'].mean():.2f}")
print(f"Average Rating: {df['Rating'].mean():.2f}")

# Additional analysis
print(f"\nTotal Sales: ${df['Sales'].sum():,.2f}")
print(f"Highest Sales: ${df['Sales'].max():,.2f} ({df.loc[df['Sales'].idxmax(), 'Product']})")
print(f"Lowest Sales: ${df['Sales'].min():,.2f} ({df.loc[df['Sales'].idxmin(), 'Product']})")

# Category-wise analysis
print("\n\nCATEGORY-WISE ANALYSIS:")
category_sales = df.groupby('Category')['Sales'].agg(['sum', 'mean', 'count'])
print(category_sales)

# VISUALIZATIONS
fig = plt.figure(figsize=(16, 10))

# 1. Bar Chart - Sales by Product
plt.subplot(2, 3, 1)
plt.bar(df['Product'], df['Sales'], color='steelblue', edgecolor='black')
plt.xlabel('Product', fontsize=10, fontweight='bold')
plt.ylabel('Sales ($)', fontsize=10, fontweight='bold')
plt.title('Sales by Product', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# 2. Scatter Plot - Sales vs Rating
plt.subplot(2, 3, 2)
colors = ['red' if cat == 'Electronics' else 'green' for cat in df['Category']]
plt.scatter(df['Rating'], df['Sales'], s=df['Quantity']*3, c=colors, alpha=0.6, edgecolors='black')
plt.xlabel('Rating', fontsize=10, fontweight='bold')
plt.ylabel('Sales ($)', fontsize=10, fontweight='bold')
plt.title('Sales vs Rating (Size = Quantity)', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='red', label='Electronics'),
                   Patch(facecolor='green', label='Accessories')]
plt.legend(handles=legend_elements, loc='upper left')

# 3. Heatmap - Correlation Matrix
plt.subplot(2, 3, 3)
numeric_df = df[['Sales', 'Quantity', 'Rating']]
correlation = numeric_df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap', fontsize=12, fontweight='bold')

# 4. Pie Chart - Sales by Category
plt.subplot(2, 3, 4)
category_total = df.groupby('Category')['Sales'].sum()
plt.pie(category_total, labels=category_total.index, autopct='%1.1f%%', 
        startangle=90, colors=['#ff9999', '#66b3ff'])
plt.title('Sales Distribution by Category', fontsize=12, fontweight='bold')

# 5. Bar Chart - Quantity by Product
plt.subplot(2, 3, 5)
plt.barh(df['Product'], df['Quantity'], color='coral', edgecolor='black')
plt.xlabel('Quantity', fontsize=10, fontweight='bold')
plt.ylabel('Product', fontsize=10, fontweight='bold')
plt.title('Quantity by Product', fontsize=12, fontweight='bold')
plt.grid(axis='x', alpha=0.3)

# 6. Line Chart - Rating Distribution
plt.subplot(2, 3, 6)
df_sorted = df.sort_values('Rating')
plt.plot(df_sorted['Product'], df_sorted['Rating'], marker='o', 
         color='green', linewidth=2, markersize=8)
plt.xlabel('Product', fontsize=10, fontweight='bold')
plt.ylabel('Rating', fontsize=10, fontweight='bold')
plt.title('Product Ratings', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data_analysis_visualizations.png', dpi=300, bbox_inches='tight')
print("\n\nVisualizations saved as 'data_analysis_visualizations.png'")
plt.show()

# INSIGHTS AND OBSERVATIONS
print("\n" + "=" * 60)
print("INSIGHTS AND OBSERVATIONS")
print("=" * 60)

print("\n1. SALES PERFORMANCE:")
print(f"   - Laptops have the highest sales (${df[df['Product']=='Laptop']['Sales'].values[0]:,})")
print(f"   - Electronics category dominates with {category_total['Electronics']/df['Sales'].sum()*100:.1f}% of total sales")

print("\n2. CORRELATION ANALYSIS:")
print(f"   - Sales and Rating correlation: {correlation.loc['Sales', 'Rating']:.3f}")
if correlation.loc['Sales', 'Rating'] > 0:
    print("     (Positive correlation suggests higher-rated products tend to have better sales)")
else:
    print("     (Little to no correlation between rating and sales)")

print("\n3. PRODUCT PERFORMANCE:")
top_3 = df.nlargest(3, 'Sales')
print("   Top 3 Products by Sales:")
for idx, row in top_3.iterrows():
    print(f"   - {row['Product']}: ${row['Sales']:,} (Rating: {row['Rating']})")

print("\n4. RATING INSIGHTS:")
print(f"   - Average rating across all products: {df['Rating'].mean():.2f}/5.0")
print(f"   - Highest rated: {df.loc[df['Rating'].idxmax(), 'Product']} ({df['Rating'].max()})")
print(f"   - Lowest rated: {df.loc[df['Rating'].idxmin(), 'Product']} ({df['Rating'].min()})")

print("\n5. RECOMMENDATIONS:")
print("   - Focus marketing on high-performing electronics")
print("   - Improve ratings for lower-rated accessories")
print("   - Consider bundling low-sales items with popular products")

print("\n" + "=" * 60)