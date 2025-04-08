# NHS HealthSign

NHS HealthSign is an analytical toolkit designed to process NHS reports from PDF documents, screen for major diseases, analyze vital signs, and monitor timeline trends. Leveraging curated clinical read codes and significance-based analysis, the system provides data-driven insights for potential health risks in patient records. The project also integrates Docling to parse and convert PDF reports into a processable Markdown format.

## Features

- **PDF Document Parsing**  
  Utilizes [Docling](https://github.com/docling-project/docling) (or your custom implementation) to parse PDF files containing NHS reports, converting them into Markdown for further analysis. This ensures that even non-digital report formats can be ingested and processed efficiently.

- **NHS Report Analysis**  
  Processes structured NHS report data to extract relevant clinical entries and vital signs.

- **Disease Screening**  
  Leverages curated clinical read codes to screen for major disease groups:
  - **Cancer** (e.g., breast, prostate, lung, colon, melanoma, leukemia)  
    
  - **Heart Disease** (e.g., myocardial infarction, heart failure, atrial fibrillation)
  - **Respiratory Conditions** (e.g., COPD, pulmonary fibrosis, lung cancer)
  - **Diabetes-related Conditions**
  - **Stroke and Cerebrovascular Events**

- **Vitals Analysis**  
  Provides comprehensive analysis of various vital sign measurements, including:
  - Anthropometric metrics (Height, Weight, BMI)
  - Cardiovascular indices (Blood Pressure, Heart Rate)
  - Metabolic panels (Cholesterol, HbA1c, Fasting Glucose)  
    
  - Additional tests related to Renal, Liver, Inflammatory, Endocrine, and Respiratory functions

- **Timeline Monitoring**  
  Tracks historical data trends and highlights significant changes in patient records to support early intervention.

- **Interactive Interface**  
  A Streamlit-powered dashboard allows healthcare professionals to visualize trends, explore data interactively, and investigate clinical histories.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/tamil-phy/NHS-HealthSign.git
    ```
2. **Navigate to the project directory:**
    ```bash
    cd NHS-HealthSign
    ```
3. **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
4. **Install dependencies:**
    ```bash
    pip install openai fuzzywuzzy tqdm docling
    ```

## Usage

- **Parse NHS PDF Reports:**  
  Use the PDF parsing script to convert PDF documents into Markdown:
    ```bash
    python pdf2md.py path/to/your/report.pdf
    ```
  This step uses Docling to extract text from PDF files, making the data ready for further analysis.

- **Run the Core Analysis:**  
  Execute the primary analysis process:
  
    ```bash
    python main.py
    ```
  
- **Launch the Interactive UI:**  
  Visualize and interact with the processed data via Streamlit:
    ```bash
    streamlit run streamlit-ui.py
    ```

## Project Structure

- **main.py:**  
  Contains the core logic for processing parsed NHS reports, screening for diseases, and calculating significance based on historical timelines.

- **pdf2md.py:**  
  Implements the PDF-to-Markdown conversion using Docling, enabling the ingestion of PDF-formatted NHS reports.

- **streamlit-ui.py:**  
  Provides an interactive front-end using Streamlit for data visualization and clinical trend exploration.

- **Configuration Files:**
  - `INTERESTED_READCODES.json`:  
    Contains clinical read codes for disease screening across various categories such as cancer, heart disease, respiratory conditions, diabetes, and stroke.  
    
  - `INTERESTED_VITALS.json`:  
    Defines vital sign tests, metrics, and their measurement units, covering parameters from anthropometrics to metabolic, renal, liver, inflammatory, endocrine, and respiratory tests.  
    
  - `red_flag_read_codes_subset.json`:  
    A subset of critical read codes for fast-track referrals and confirmed diagnoses to flag high-risk conditions.  
    
  - `secrets.json`:  
    Holds API keys and sensitive configurations (ensure this file is secured and not exposed publicly).  
