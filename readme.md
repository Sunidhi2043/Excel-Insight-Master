# AI Query Processor for Excel Sheets

This project is an AI-powered tool that allows users to upload Excel files, input queries, and receive processed results. The system leverages Natural Language Processing (NLP) for advanced query understanding, enabling users to interact with their Excel sheets using free-form text queries. It processes the queries, performs the necessary operations on the Excel data, and generates results, including charts.

## Features
- **File Upload:** Users can upload Excel files.
- **Query Processing:** Users can input queries to extract or manipulate data from the Excel file.
- **Chart Generation:** The system can generate visual charts based on the data processed from the query.
- **NLP Integration:** Advanced query understanding powered by spaCy to handle complex, free-form queries.
- **Machine Learning Model Integration:** A trained model (without Keras) is used to interpret queries at an advanced level.
- **Web Interface:** The project includes a simple web interface where users can upload files and input queries.

## Prerequisites
To run this project locally, you need to have the following installed:
- Python 3.12 (or a compatible version)
- VS Code (optional but recommended for development)
- pip (Python package installer)

## Setup

### 1. Clone the repository
```bash
git clone <repository_url>
cd ai-query-processor
