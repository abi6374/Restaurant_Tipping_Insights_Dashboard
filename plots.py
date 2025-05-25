import seaborn as sns
import matplotlib.pyplot as plt

# Function to create a box plot
def create_box_plot(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='gender', y='tip', data=data)
    plt.title('Box Plot of Tips by Gender')
    plt.savefig('box_plot.png')
    plt.close()

# Function to create a scatter plot
def create_scatter_plot(data):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='total_bill', y='tip', hue='time', data=data)
    plt.title('Scatter Plot of Total Bill vs Tip')
    plt.savefig('scatter_plot.png')
    plt.close()

# Function to create a heatmap
def create_heatmap(data):
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True)
    plt.title('Heatmap of Correlations')
    plt.savefig('heatmap.png')
    plt.close()