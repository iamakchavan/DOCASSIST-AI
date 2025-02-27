# DocAssist AI - Medical Report Analysis System

![DocAssist AI](frontend/public/assets/logo.png)

DocAssist AI is a sophisticated medical report analysis system that leverages machine learning to analyze blood test reports and provide intelligent medical recommendations. The system can process both PDF reports and manually entered blood test values to deliver comprehensive medical insights.

## Features

- 🔍 **PDF Report Analysis**: Automatically extract medical values from uploaded PDF reports
- 📊 **Manual Data Entry**: Input blood test values manually for instant analysis
- 🏥 **Disease Pattern Detection**: Identify potential diseases based on blood parameter patterns
- 📈 **Abnormal Value Detection**: Highlight and explain abnormal blood test results
- 💊 **Treatment Recommendations**: Provide detailed treatment plans and monitoring guidelines
- 📱 **Modern UI/UX**: Clean, responsive interface with real-time updates
- 🔒 **Secure Processing**: Local processing of medical data with no external storage

## Tech Stack

### Frontend
- HTML5/CSS3/JavaScript
- Modern UI components with shadcn-inspired styling
- Responsive design for all devices
- Chart.js for data visualization

### Backend
- Python 3.8+
- Flask for API server
- PyPDF2 for PDF processing
- NumPy/Pandas for data processing
- Scikit-learn for ML predictions

## Prerequisites

Before running the application, ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package manager)
- Git

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/realranjan/DOCASSIST-AI.git
   cd DOCASSIST-AI
   ```

2. Set up the Python virtual environment:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Start the Flask backend server:
   ```bash
   cd backend
   python app.py
   ```
   The backend server will start on `http://localhost:5000`

2. Open the frontend:
   - Navigate to the `frontend` directory
   - Open `index.html` in your web browser
   - For the best experience, use a modern web browser (Chrome, Firefox, Edge)

## Usage

1. **PDF Analysis**:
   - Click the "Upload PDF" button
   - Select a blood test report PDF
   - Wait for the analysis results

2. **Manual Entry**:
   - Navigate to the "Manual Entry" tab
   - Fill in the blood test parameters
   - Click "Analyze" for instant results

3. **View Results**:
   - Review the comprehensive medical report
   - Check abnormal values and their implications
   - Review disease patterns if detected
   - Follow recommended treatments and monitoring plans

## Development

To contribute to the project:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## API Documentation

The backend provides several REST endpoints:

- `POST /predict/file`: Analyze PDF reports
- `POST /predict`: Process manual entries
- `GET /api/dashboard-data`: Get dashboard statistics

For detailed API documentation, refer to the [API Documentation](docs/API.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Medical reference ranges and disease patterns are based on standard medical guidelines
- UI design inspired by modern healthcare applications
- Special thanks to all contributors and the medical professionals who provided domain expertise

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.

## Authors

- Ranjan Kumar - Initial work - [realranjan](https://github.com/realranjan)

---
Made with ❤️ by the DocAssist AI Team 