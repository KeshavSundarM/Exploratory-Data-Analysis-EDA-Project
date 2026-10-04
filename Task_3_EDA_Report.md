# Task 3 – Exploratory Data Analysis (EDA) Project

## Employee Performance and Workforce Analysis

### Objective
Analyze employee data to discover patterns, relationships, trends, and factors that may influence performance and attrition.

### Dataset
The project contains Age, Experience, Department, Education, Monthly Income, Job Satisfaction, Work Hours per Week, Performance Score, Annual Leave Days, and Attrition. Missing values and duplicate records are intentionally included to demonstrate data cleaning.

### Data Cleaning
1. Load data using Pandas.
2. Inspect missing values and duplicates.
3. Remove duplicate rows.
4. Fill missing numerical values with the median.
5. Convert Attrition to a numeric flag for correlation analysis.

### Exploratory Analysis
The project performs descriptive statistics, department-wise analysis, education-wise analysis, correlation analysis, and six visualizations.

### Key Findings
- Performance varies between departments.
- Education level is associated with differences in average income.
- Job satisfaction has a relationship with performance in the sample.
- Work hours show an association with attrition.
- Income, experience, satisfaction, workload, and education should be considered together when studying workforce outcomes.

### Visualizations
1. Income distribution
2. Average performance by department
3. Average income by education
4. Attrition rate by department
5. Job satisfaction vs performance
6. Correlation matrix

### Conclusion
The EDA provides a structured view of employee data and identifies useful patterns for further investigation. These are exploratory associations, not causal conclusions. The project can be extended into predictive modeling for employee attrition or performance.

### Technologies
Python, Pandas, NumPy, Matplotlib, Jupyter Notebook.
