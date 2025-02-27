from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import FunctionTransformer
import PyPDF2
import re
import os
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Add these constants at the top of the file
BLOOD_RANGES = {
    'HAEMATOCRIT': {
        'low': {'value': 29, 'unit': '%'},
        'high': {'value': 66, 'unit': '%'},
        'conditions': {
            'low': 'Possible anemia',
            'high': 'Possible polycythemia or dehydration'
        }
    },
    'HAEMOGLOBINS': {
        'low': {'value': 12, 'unit': 'g/dL'},
        'high': {'value': 18, 'unit': 'g/dL'},
        'conditions': {
            'low': 'Anemia or blood loss',
            'high': 'Polycythemia or dehydration'
        }
    },
    'ERYTHROCYTE': {
        'low': {'value': 4.0, 'unit': 'M/µL'},
        'high': {'value': 6.2, 'unit': 'M/µL'},
        'conditions': {
            'low': 'Decreased red blood cell production or increased destruction',
            'high': 'Polycythemia or bone marrow disorder'
        }
    },
    'LEUCOCYTE': {
        'low': {'value': 4.0, 'unit': 'K/µL'},
        'high': {'value': 11.0, 'unit': 'K/µL'},
        'conditions': {
            'low': 'Weakened immune system or bone marrow problems',
            'high': 'Infection, inflammation, or leukemia'
        }
    },
    'THROMBOCYTE': {
        'low': {'value': 150, 'unit': 'K/µL'},
        'high': {'value': 450, 'unit': 'K/µL'},
        'conditions': {
            'low': 'Increased bleeding risk',
            'high': 'Increased clotting risk'
        }
    },
    'MCH': {
        'low': {'value': 27, 'unit': 'pg'},
        'high': {'value': 32, 'unit': 'pg'},
        'conditions': {
            'low': 'Iron deficiency',
            'high': 'Possible vitamin B12 deficiency'
        }
    },
    'MCHC': {
        'low': {'value': 32, 'unit': 'g/dL'},
        'high': {'value': 36, 'unit': 'g/dL'},
        'conditions': {
            'low': 'Iron deficiency anemia',
            'high': 'Hereditary spherocytosis'
        }
    },
    'MCV': {
        'low': {'value': 80, 'unit': 'fL'},
        'high': {'value': 100, 'unit': 'fL'},
        'conditions': {
            'low': 'Microcytic anemia',
            'high': 'Macrocytic anemia'
        }
    }
}

DISEASE_PATTERNS = {
    'IRON DEFICIENCY ANEMIA': {
        'conditions': {
            'HAEMOGLOBINS': {'condition': 'low', 'importance': 'primary'},
            'MCV': {'condition': 'low', 'importance': 'primary'},
            'MCH': {'condition': 'low', 'importance': 'secondary'},
            'HAEMATOCRIT': {'condition': 'low', 'importance': 'secondary'}
        },
        'treatments': [
            "Primary Treatment:\n" +
            "- Ferrous sulfate 325mg oral tablet twice daily\n" +
            "- Vitamin C 500mg with iron tablets to enhance absorption\n" +
            "- Folic acid 1mg daily",
            
            "Precautions:\n" +
            "- Take iron on empty stomach\n" +
            "- Avoid antacids, calcium supplements\n" +
            "- May cause dark stools (normal side effect)",
            
            "Monitoring:\n" +
            "- CBC every 2-3 weeks until hemoglobin normalizes\n" +
            "- Ferritin levels monthly\n" +
            "- Iron studies in 3 months"
        ]
    },
    
    'MEGALOBLASTIC ANEMIA': {
        'conditions': {
            'HAEMOGLOBINS': {'condition': 'low', 'importance': 'primary'},
            'MCV': {'condition': 'high', 'importance': 'primary'},
            'MCH': {'condition': 'high', 'importance': 'secondary'}
        },
        'treatments': [
            "Primary Treatment:\n" +
            "- Vitamin B12 1000mcg IM injection weekly for 4 weeks\n" +
            "- Folic acid 5mg daily\n" +
            "- Then monthly B12 injections for maintenance",
            
            "Precautions:\n" +
            "- Monitor neurological symptoms\n" +
            "- Avoid nitrous oxide anesthesia\n" +
            "- Report any numbness/tingling",
            
            "Monitoring:\n" +
            "- CBC weekly for first month\n" +
            "- B12 and folate levels monthly\n" +
            "- Neurological assessment every visit"
        ]
    },
    
    'ACUTE INFECTION': {
        'conditions': {
            'LEUCOCYTE': {'condition': 'high', 'importance': 'primary'},
            'THROMBOCYTE': {'condition': 'high', 'importance': 'secondary'}
        },
        'treatments': [
            "Primary Treatment:\n" +
            "- Broad-spectrum antibiotics based on likely source:\n" +
            "- Amoxicillin-clavulanate 875/125mg twice daily\n" +
            "- Alternative: Azithromycin 500mg day 1, then 250mg days 2-5",
            
            "Precautions:\n" +
            "- Complete full course of antibiotics\n" +
            "- Monitor for allergic reactions\n" +
            "- Report fever > 101°F or worsening symptoms",
            
            "Monitoring:\n" +
            "- Daily temperature checks\n" +
            "- CBC with differential in 3-5 days\n" +
            "- CRP and ESR to track inflammation"
        ]
    },
    
    'SEVERE THROMBOCYTOPENIA': {
        'conditions': {
            'THROMBOCYTE': {'condition': 'low', 'importance': 'primary'}
        },
        'treatments': [
            "Primary Treatment:\n" +
            "- If autoimmune: Prednisone 1mg/kg/day\n" +
            "- Platelet transfusion if count < 10,000 or bleeding\n" +
            "- IVIG 1g/kg if severe autoimmune cause",
            
            "Precautions:\n" +
            "- Avoid aspirin and NSAIDs\n" +
            "- No contact sports or activities with bleeding risk\n" +
            "- Use soft toothbrush, electric razor only\n" +
            "- Report any unusual bruising or bleeding",
            
            "Monitoring:\n" +
            "- Daily platelet counts until stable\n" +
            "- Bleeding time and coagulation studies\n" +
            "- Regular blood pressure checks"
        ]
    },
    
    'POLYCYTHEMIA': {
        'conditions': {
            'HAEMATOCRIT': {'condition': 'high', 'importance': 'primary'},
            'HAEMOGLOBINS': {'condition': 'high', 'importance': 'primary'},
            'ERYTHROCYTE': {'condition': 'high', 'importance': 'secondary'}
        },
        'treatments': [
            "Primary Treatment:\n" +
            "- Therapeutic phlebotomy 500mL weekly\n" +
            "- Hydroxyurea 500mg twice daily if indicated\n" +
            "- Low-dose aspirin 81mg daily for clot prevention",
            
            "Precautions:\n" +
            "- Maintain adequate hydration\n" +
            "- Avoid smoking and alcohol\n" +
            "- Report headaches or visual changes\n" +
            "- Avoid high altitudes",
            
            "Monitoring:\n" +
            "- CBC weekly until stable\n" +
            "- Iron studies monthly\n" +
            "- JAK2 mutation testing\n" +
            "- Regular blood pressure monitoring"
        ]
    },
    
    'PANCYTOPENIA': {
        'conditions': {
            'HAEMOGLOBINS': {'condition': 'low', 'importance': 'primary'},
            'LEUCOCYTE': {'condition': 'low', 'importance': 'primary'},
            'THROMBOCYTE': {'condition': 'low', 'importance': 'primary'}
        },
        'treatments': [
            "Primary Treatment:\n" +
            "- Immediate hematology consultation\n" +
            "- Blood product support as needed\n" +
            "- G-CSF if neutropenic\n" +
            "- Bone marrow evaluation required",
            
            "Precautions:\n" +
            "- Strict infection precautions\n" +
            "- Avoid crowds and sick contacts\n" +
            "- No invasive procedures without coverage\n" +
            "- Bleeding precautions as for thrombocytopenia",
            
            "Monitoring:\n" +
            "- Daily CBC with differential\n" +
            "- Fever monitoring every 4 hours\n" +
            "- Weekly bone marrow recovery assessment\n" +
            "- Regular blood product support assessment"
        ]
    }
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_medical_values(text):
    """Extract medical values from text using regex patterns"""
    patterns = {
        'HAEMATOCRIT': r'(?i)h[ae]matocrit.*?(\d+\.?\d*)',
        'HAEMOGLOBINS': r'(?i)h[ae]moglobin.*?(\d+\.?\d*)',
        'ERYTHROCYTE': r'(?i)erythrocyte.*?(\d+\.?\d*)',
        'LEUCOCYTE': r'(?i)leucocyte.*?(\d+\.?\d*)',
        'THROMBOCYTE': r'(?i)thrombocyte.*?(\d+\.?\d*)',
        'MCH': r'(?i)MCH.*?(\d+\.?\d*)',
        'MCHC': r'(?i)MCHC.*?(\d+\.?\d*)',
        'MCV': r'(?i)MCV.*?(\d+\.?\d*)',
        'AGE': r'(?i)age.*?(\d+)',
        'SEX': r'(?i)(?:sex|gender).*?(male|female|m|f)'
    }
    
    results = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1)
            if key == 'SEX':
                # Convert sex to binary (0 for female, 1 for male)
                value = 1 if value.lower() in ['male', 'm'] else 0
            else:
                value = float(value)
            results[key] = value
    
    return results

# Feature engineering function exactly as used in training
def feature_engineering(df):
    df['THROMBOCYTE_LEUCOCYTE_RATIO'] = df['THROMBOCYTE'] / (df['LEUCOCYTE'] + 1e-6)
    df['ERYTHROCYTE_LEUCOCYTE'] = df['ERYTHROCYTE'] * df['LEUCOCYTE']
    return df

# Define model path
MODEL_PATH = r'C:\Users\ranja\OneDrive\Documents\upgrad\capstone\docassist\models\final_model_pipeline.pkl'

# Global variable for model
model = None

def load_model():
    global model
    try:
        # Register the feature engineering function
        globals()['feature_engineering'] = feature_engineering
        # Load the model pipeline
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print(f"Please ensure your model file exists at: {MODEL_PATH}")
        return False

# Load model when starting the server
if not load_model():
    raise RuntimeError("Failed to load the model. Please check the model path and file.")

@app.route('/predict/file', methods=['POST'])
def predict_from_file():
    if 'file' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No file uploaded'
        }), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({
            'status': 'error',
            'message': 'No file selected'
        }), 400
        
    if not allowed_file(file.filename):
        return jsonify({
            'status': 'error',
            'message': 'Invalid file type. Only PDF files are allowed.'
        }), 400
        
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract text from PDF
        text = ""
        with open(filepath, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        
        # Extract values from text
        extracted_data = extract_medical_values(text)
        
        # Check if all required fields were found
        required_fields = ['HAEMATOCRIT', 'HAEMOGLOBINS', 'ERYTHROCYTE', 'LEUCOCYTE',
                         'THROMBOCYTE', 'MCH', 'MCHC', 'MCV', 'AGE', 'SEX']
        missing_fields = [field for field in required_fields if field not in extracted_data]
        
        if missing_fields:
            return jsonify({
                'status': 'error',
                'message': f'Missing required fields in PDF: {", ".join(missing_fields)}'
            }), 400
        
        # Create DataFrame for prediction
        input_data = {
            'HAEMATOCRIT': extracted_data['HAEMATOCRIT'],
            'HAEMOGLOBINS': extracted_data['HAEMOGLOBINS'],
            'ERYTHROCYTE': extracted_data['ERYTHROCYTE'],
            'LEUCOCYTE': extracted_data['LEUCOCYTE'],
            'THROMBOCYTE': extracted_data['THROMBOCYTE'],
            'MCH': extracted_data['MCH'],
            'MCHC': extracted_data['MCHC'],
            'MCV': extracted_data['MCV'],
            'AGE': extracted_data['AGE'],
            'SEX_ENCODED': extracted_data['SEX']
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Make prediction using the pipeline
        prediction = model.predict(input_df)
        
        # Convert prediction to meaningful response
        result = "Inpatient" if prediction[0] == 1 else "Outpatient"
        
        # Clean up - remove uploaded file
        os.remove(filepath)
        
        return jsonify({
            'status': 'success',
            'prediction': result,
            'prediction_code': int(prediction[0]),
            'extracted_values': extracted_data
        })
        
    except Exception as e:
        # Clean up in case of error
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

def check_disease_patterns(values):
    """Check if values match known disease patterns"""
    detected_diseases = {}
    
    for disease, pattern in DISEASE_PATTERNS.items():
        matches = {'primary': 0, 'secondary': 0}
        required = {'primary': 0, 'secondary': 0}
        
        for param, criteria in pattern['conditions'].items():
            if criteria['importance'] == 'primary':
                required['primary'] += 1
            else:
                required['secondary'] += 1
                
            if param in values:
                ranges = BLOOD_RANGES[param]
                value = values[param]
                
                if criteria['condition'] == 'low' and value < ranges['low']['value']:
                    matches[criteria['importance']] += 1
                elif criteria['condition'] == 'high' and value > ranges['high']['value']:
                    matches[criteria['importance']] += 1
        
        # Disease is detected if all primary conditions are met and at least half of secondary
        if (matches['primary'] == required['primary'] and 
            (required['secondary'] == 0 or matches['secondary'] >= required['secondary'] / 2)):
            detected_diseases[disease] = pattern['treatments']
    
    return detected_diseases

def format_medical_report(prediction, values, detected_diseases, abnormal_values):
    """Generate a formatted medical report with clean styling"""
    report = []
    severity = get_severity_level(abnormal_values)
    
    # Header
    report.append("<div class='report-header'>")
    report.append("Medical Laboratory Report")
    report.append(f"{datetime.now().strftime('%B %d, %Y')}")
    report.append("</div>")
    
    # Patient Information
    report.append("<div class='section'>")
    report.append("<h2>Patient Information</h2>")
    report.append(f"Age: {int(values['AGE'])} years")
    report.append(f"Sex: {'Male' if values['SEX_ENCODED'] == 1 else 'Female'}")
    report.append("</div>")
    
    # Blood Analysis Results
    report.append("<div class='section'>")
    report.append("<h2>Blood Analysis Results</h2>")
    report.append("<div class='results-table'>")
    report.append("<table>")
    report.append("<tr><th>Parameter</th><th>Value</th><th>Normal Range</th><th>Status</th></tr>")
    
    for param, value in values.items():
        if param in BLOOD_RANGES:
            ranges = BLOOD_RANGES[param]
            severity_text = get_parameter_severity(param, value)
            value_text = f"<span class='abnormal'>{value:.1f}</span>" if severity_text != "Normal" else f"{value:.1f}"
            normal_range = f"{ranges['low']['value']}-{ranges['high']['value']} {ranges['low']['unit']}"
            report.append(f"<tr><td>{param}</td><td>{value_text}</td><td>{normal_range}</td><td>{severity_text}</td></tr>")
    
    report.append("</table></div></div>")
    
    # Clinical Interpretation
    report.append("<div class='section'>")
    report.append("<h2>Clinical Interpretation</h2>")
    
    # List all abnormal findings
    abnormal_findings = []
    for param, value in values.items():
        if param in BLOOD_RANGES:
            ranges = BLOOD_RANGES[param]
            if value < ranges['low']['value']:
                abnormal_findings.append(f"• {param} is Low ({value:.1f} {ranges['low']['unit']}): {ranges['conditions']['low']}")
            elif value > ranges['high']['value']:
                abnormal_findings.append(f"• {param} is High ({value:.1f} {ranges['high']['unit']}): {ranges['conditions']['high']}")
    
    if abnormal_findings:
        report.append("<div class='findings'>")
        report.append("<strong>Abnormal Findings:</strong>")
        for finding in abnormal_findings:
            report.append(f"<div class='finding'>{finding}</div>")
        report.append("</div>")
    
    # Overall interpretation
    if prediction == 1:
        report.append("<div class='warning'>")
        report.append("CRITICAL CONDITION DETECTED - REQUIRES HOSPITALIZATION")
        report.append("</div>")
        if detected_diseases:
            for disease in detected_diseases:
                report.append(f"<div class='finding'>Severe {disease.lower()} requiring immediate attention</div>")
    else:
        if detected_diseases:
            for disease in detected_diseases:
                report.append(f"<div class='finding'>{disease.title()} detected—manageable with outpatient care</div>")
        else:
            report.append("<div class='normal-finding'>All parameters are within normal range or show minor variations</div>")
    report.append("</div>")
    
    # Recommended Action
    report.append("<div class='section'>")
    report.append("<h2>Recommended Action</h2>")
    if prediction == 1:
        report.append("<div class='urgent'>IMMEDIATE HOSPITAL ADMISSION RECOMMENDED</div>")
        
        # Specialist Consultations
        report.append("<h3>Required Consultations</h3>")
        report.append("<ul>")
        if detected_diseases:
            for disease in detected_diseases:
                if 'ANEMIA' in disease:
                    report.append("<li>Hematologist</li>")
                elif 'INFECTION' in disease:
                    report.append("<li>Infectious Disease Specialist</li>")
                elif 'THROMBOCYTOPENIA' in disease:
                    report.append("<li>Hematologist</li>")
                elif 'POLYCYTHEMIA' in disease:
                    report.append("<li>Hematologist/Oncologist</li>")
        report.append("<li>Internal Medicine</li>")
        report.append("</ul>")
    
    if detected_diseases:
        # Medications and Treatment
        for disease, treatments in detected_diseases.items():
            report.append(f"<h3>Treatment Plan for {disease}</h3>")
            report.append("<ul class='treatment-list'>")
            
            # Medications
            medications = []
            for treatment in treatments:
                if "Primary Treatment:" in treatment:
                    medications.extend([item.strip() for item in treatment.split('\n') if item.strip() and item.strip() != "Primary Treatment:"])
            if medications:
                report.append("<li><strong>Medications:</strong></li>")
                for med in medications:
                    if med.startswith('-'):
                        report.append(f"<li class='treatment-item'>{med[1:].strip()}</li>")
                    else:
                        report.append(f"<li class='treatment-item'>{med}</li>")
            
            # Precautions
            precautions = []
            for treatment in treatments:
                if "Precautions:" in treatment:
                    precautions.extend([item.strip() for item in treatment.split('\n') if item.strip() and item.strip() != "Precautions:"])
            if precautions:
                report.append("<li><strong>Precautions:</strong></li>")
                for precaution in precautions:
                    if precaution.startswith('-'):
                        report.append(f"<li class='treatment-item'>{precaution[1:].strip()}</li>")
                    else:
                        report.append(f"<li class='treatment-item'>{precaution}</li>")
            
            # Follow-up Plan
            monitoring = []
            for treatment in treatments:
                if "Monitoring:" in treatment:
                    monitoring.extend([item.strip() for item in treatment.split('\n') if item.strip() and item.strip() != "Monitoring:"])
            if monitoring:
                report.append("<li><strong>Follow-up Plan:</strong></li>")
                for plan in monitoring:
                    if plan.startswith('-'):
                        report.append(f"<li class='treatment-item'>{plan[1:].strip()}</li>")
                    else:
                        report.append(f"<li class='treatment-item'>{plan}</li>")
            
            report.append("</ul>")
    else:
        report.append("<h3>Recommendations</h3>")
        report.append("<ul>")
        report.append("<li>Continue routine health maintenance</li>")
        report.append("<li>Regular exercise and balanced diet</li>")
        report.append("<li>Annual health check-up</li>")
        report.append("</ul>")
    report.append("</div>")
    
    # Footer
    report.append("<div class='report-footer'>")
    if prediction == 1:
        report.append("<div class='urgent-notice'>URGENT MEDICAL ATTENTION IS NECESSARY</div>")
    else:
        if detected_diseases:
            report.append("<div class='notice'>Condition is manageable with outpatient care</div>")
        else:
            report.append("<div class='notice'>Overall health status is satisfactory</div>")
    report.append("<div class='generated-by'>Report Generated by AI-Powered Medical Decision Support System</div>")
    report.append("</div>")
    
    return "\n".join(report)

def get_severity_level(abnormal_values):
    """Determine the severity level based on abnormal values"""
    if not abnormal_values:
        return "No"
    
    severe_count = 0
    for param, details in abnormal_values.items():
        if is_severe_abnormality(param, details['value']):
            severe_count += 1
    
    if severe_count >= 2:
        return "Severe"
    elif severe_count == 1:
        return "Moderate"
    return "Mild"

def get_parameter_severity(param, value):
    """Get severity description for a parameter"""
    ranges = BLOOD_RANGES[param]
    if value < ranges['low']['value']:
        if value < ranges['low']['value'] * 0.8:
            return f"Severe {ranges['conditions']['low']}"
        return ranges['conditions']['low']
    elif value > ranges['high']['value']:
        if value > ranges['high']['value'] * 1.2:
            return f"Severe {ranges['conditions']['high']}"
        return ranges['conditions']['high']
    return "Normal"

def is_severe_abnormality(param, value):
    """Check if the abnormality is severe"""
    ranges = BLOOD_RANGES[param]
    return (value < ranges['low']['value'] * 0.8 or 
            value > ranges['high']['value'] * 1.2)

# Modify the predict endpoint to include ranges and recommendations
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded'
        }), 500
        
    try:
        data = request.json
        
        # Create DataFrame with user input
        input_data = {
            'HAEMATOCRIT': float(data['HAEMATOCRIT']),
            'HAEMOGLOBINS': float(data['HAEMOGLOBINS']),
            'ERYTHROCYTE': float(data['ERYTHROCYTE']),
            'LEUCOCYTE': float(data['LEUCOCYTE']),
            'THROMBOCYTE': float(data['THROMBOCYTE']),
            'MCH': float(data['MCH']),
            'MCHC': float(data['MCHC']),
            'MCV': float(data['MCV']),
            'AGE': float(data['AGE']),
            'SEX_ENCODED': int(data['SEX'])
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model.predict(input_df)
        
        # Get recommendations
        recommendations = format_medical_report(prediction[0], input_data, check_disease_patterns(input_data), {})
        
        # Convert prediction to meaningful response
        result = "Inpatient" if prediction[0] == 1 else "Outpatient"
        
        return jsonify({
            'status': 'success',
            'prediction': result,
            'prediction_code': int(prediction[0]),
            'recommendations': recommendations,
            'blood_ranges': BLOOD_RANGES
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

# Add this new endpoint
@app.route('/api/dashboard-data', methods=['GET'])
def get_dashboard_data():
    try:
        # This would normally come from your database
        return jsonify({
            'parameters': {
                'HAEMATOCRIT': 42.5,
                'HAEMOGLOBINS': 14.2,
                'ERYTHROCYTE': 4.8,
                'LEUCOCYTE': 7.2,
                'THROMBOCYTE': 250,
                'MCH': 29.5,
                'MCHC': 33.5,
                'MCV': 88.5
            },
            'health_score': 92,
            'summary': "Your complete blood count shows optimal hematological health. All red blood cell indices (MCV, MCH, MCHC) are within normal ranges, indicating healthy red blood cell production and function. Your oxygen-carrying capacity is excellent, and both white blood cells and platelets are at ideal levels."
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 