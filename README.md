# Customer Churn Prediction

This project focuses on predicting customer churn using machine learning techniques. The goal is to identify customers who are likely to churn, allowing businesses to take proactive measures to retain them.

## Project Files

* `app.py`: This is likely the main application file, possibly a Flask or Streamlit application, that serves the churn prediction model.
* `Customer Churn new.csv`: This CSV file probably contains the primary dataset used for training and evaluating the churn prediction model. It's expected to have various customer-related features and a target variable indicating churn.
* `customer_churn_with_added_features.csv`: This CSV file suggests that some feature engineering or additional data has been incorporated into the original `Customer Churn new.csv` dataset, potentially leading to improved model performance.
* `requirements.txt`: This file lists all the Python libraries and their versions required to run the `app.py` and other scripts in this project. It ensures that the project environment can be replicated.

## Getting Started

To run this project, you'll need to set up your Python environment and install the necessary dependencies.

### Prerequisites

* Python 3.x

### Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <your-repository-url>
    cd <your-project-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

After installing the dependencies, you can run the main application:

```bash
python app.py
