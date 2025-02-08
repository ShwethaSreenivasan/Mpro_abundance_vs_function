#Written by Shwetha Sreenivasan, Swint-Kruse Laboratory, KUMC
#27th August 2024
#Python 3.10.6

import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import sys

def generate_roc_curve(input_file, output_image):
    """
    Generates a ROC curve from an Excel file and saves it as an image.

    Parameters:
    input_file (str): Path to the Excel file containing true labels and predicted scores.
    output_image (str): Path to save the generated ROC curve image.
    """
    # Read the Excel file into a DataFrame
    data = pd.read_excel(input_file)

    # Extracting the true labels and predicted scores from the first two columns
    true_label = data.iloc[:, 0]
    predicted_score = data.iloc[:, 1]

    # Generate the false positive rate (fpr), true positive rate (tpr), and thresholds for the ROC curve
    fpr, tpr, thresholds = roc_curve(true_label, predicted_score)
    
    # Calculate the Area Under the Curve (AUC) which provides a measure of the overall performance of the model
    roc_auc = auc(fpr, tpr)

    # Plotting the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Plot the diagonal line representing a random classifier
    plt.xlim([0.0, 1.0])  # Set the limits of the x-axis
    plt.ylim([0.0, 1.05]) # Set the limits of the y-axis
    plt.xlabel('False Positive Rate')  # Label for the x-axis
    plt.ylabel('True Positive Rate')  # Label for the y-axis
    plt.title('Receiver Operating Characteristic')  # Title of the plot
    plt.legend(loc="lower right")  # Display legend in the lower right corner

    # Save the figure to the specified output file
    plt.savefig(output_image)
    plt.close()  # Close the plot to free up memory

    print(f"ROC curve saved as {output_image}")

if __name__ == "__main__":
    # Check if the correct number of arguments are provided
    if len(sys.argv) != 3:
        # Print usage statement if the arguments are incorrect
        print("Usage: python generate_roc.py <input_excel_file> <output_image_file>")
        sys.exit(1)  # Exit the program with an error code

    # Get the input Excel file and output image file name from the command-line arguments
    input_file = sys.argv[1]
    output_image = sys.argv[2]

    # Call the function to generate the ROC curve and save the image
    generate_roc_curve(input_file, output_image)
